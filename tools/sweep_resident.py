"""FAST-PATH SWEEP runner: multi-view stochastic training on the full-resident + device-loss path
(the 2.74x trainer), with camera-as-device-buffers (one trace set serves all views, keystone-verified in
cambuf_test) + per-view gt resident on device + held-out eval + .ply. This is bh_native_sweep's
methodology on the FAST trainer instead of the host-bound hybrid.

Per iter: pick a train view -> set camera buffers + copy that view's gt into y_d (device) -> geom-fwd
trace -> (every-N) host binning -> rend-fwd trace -> device loss-grad trace (gC on device) -> rend-bwd
trace -> host scatter -> geom-bwd trace -> device Adam. Eval renders the test split (traces, no bwd).

Outputs conform to the canonical contract: eval.json (evalcard), bh_fast_rollup.json, summary.txt,
and GT|render held-out panels on full runs.

Run (one G per process):
    podman-compose --profile hw run --rm hw python3 tools/sweep_resident.py --G 1000 --iters 3000
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import torch
import ttnn
from PIL import Image

from spike import data, metrics, sh, plyio, evalcard
from spike.model import GaussianModel
from spike.train import DEFAULT_LR
from m4_train_binned import TileMap, assign_bins, K_POLY
from geom_device import device_fwd_core, device_bwd_core, A, M
from traced_fwd import gather_theta_buf
from resident_traced import render_bwd, theta_bwd, T3, PN, B1, B2, EPS
from loss_manual import gauss_1d, band_matrix, C1 as L_C1, C2 as L_C2, LAMBDA as L_LAM
from bin_device import bin_to_buffers, build_inv_device, make_bin_ctx
from m9_scatter_oracle import build_inv

C0 = sh.C0
DEV = None
CG = None
DT = ttnn.float32
BF = ttnn.bfloat16


def u(t, dt=DT):
    return ttnn.from_torch(t.reshape(-1, 1).contiguous() if t.dim() == 1 else t.contiguous(),
                           dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV)


def uh(t, dt=DT):   # HOST tensor (valid source for copy_host_to_device_tensor)
    return ttnn.from_torch(t.reshape(-1, 1).contiguous() if t.dim() == 1 else t.contiguous(),
                           dtype=dt, layout=ttnn.TILE_LAYOUT)


def dn(t):
    return ttnn.to_torch(t).float()


def setbuf(buf, t, dt=BF, layout=ttnn.TILE_LAYOUT):
    t2 = t.reshape(-1, 1) if t.dim() == 1 else t
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(t2.contiguous(), dtype=dt, layout=layout), buf)


def _save_img(path, img):
    arr = (img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_panel(path, gt, render):
    """Save GT (left) | render (right) side-by-side panel as uint8 PNG."""
    gt_arr = (gt.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
    rend_arr = (render.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
    panel = np.concatenate([gt_arr, rend_arr], axis=1)
    Image.fromarray(panel).save(path)


def main():
    global DEV, CG
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--scene-name", default="ficus")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--G", type=int, default=1000)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bin-every", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--out", default="outputs/bh_sweep_fast")
    ap.add_argument("--method", default="bh_fast_resident",
                    help="algo label for canonical output paths (eval.json/rollup/panels)")
    ap.add_argument("--no-ply", action="store_true")
    ap.add_argument("--no-views", action="store_true",
                    help="skip GT|render panel saves even on full runs")
    # MCMC density control (ported from spike/mcmc.py). 0 disables.
    ap.add_argument("--lam-o", type=float, default=0.01)        # opacity-L1 reg
    ap.add_argument("--lam-s", type=float, default=0.01)        # scale-L1 reg (kills runaway-scale spikes)
    ap.add_argument("--relocate-every", type=int, default=100)  # 0 disables relocation
    ap.add_argument("--dead-thr", type=float, default=0.005)
    # depth-weight lever (arm B). off -> reproduces arm-A exactly.
    ap.add_argument("--depth-weight", action="store_true")
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--tau0", type=float, default=4.0)          # refined from z readback at warmup
    ap.add_argument("--lr-bt", type=float, default=0.02)        # Adam LR for beta/tau scalars
    # arm-softmin lever (arm SM / candidate #2): rho=exp(-(mcz-zref)/tau), z ATTACHED (C3 z-force).
    # Mutually exclusive with --depth-weight.
    ap.add_argument("--arm-softmin", action="store_true")
    ap.add_argument("--smtau0", type=float, default=1.5,
                    help="softmin tau init (default 1.5; warmup sets zref to median visible mcz)")
    ap.add_argument("--lr-smtau", type=float, default=0.02,    # Adam LR for tau scalar
                    help="Adam LR for softmin tau scalar")
    # arm-pairwise lever (arm PW / candidate #5 redux): per-tile [K,K] soft-compare GEMM,
    # OIT-over composite (replaces render_fwd_cache/render_bwd entirely for this arm).
    # Mutually exclusive with --depth-weight/--arm-softmin.
    ap.add_argument("--arm-pairwise", action="store_true",
                    help="per-tile pairwise soft-occlusion (arm PW): OIT-over composite")
    ap.add_argument("--pwtau0", type=float, default=0.01,
                    help="PW pairwise soft-compare tau init (GPU bake-off winner value)")
    ap.add_argument("--lr-pwtau", type=float, default=0.02,
                    help="Adam LR for the pw_tau scalar")
    # random-background training (gsplat-style honest-opacity; WSR bakes bg into the normalized avg
    # otherwise -> grey/white empty-space gaussians). per-iter random bg, c_B=bg; eval stays white.
    ap.add_argument("--random-bg", action="store_true")
    ap.add_argument("--colmap", default=None, help="path to a COLMAP scene root (real scene); overrides blender")
    ap.add_argument("--downscale", type=int, default=4, help="COLMAP image downscale factor")
    # device-resident fast-path levers (opt-in; default off = current host behaviour). Ported from
    # resident_traced.py, which already validated each against the host oracle.
    ap.add_argument("--device-binning", action="store_true",
                    help="compute binning idx/valid ON DEVICE (bin_to_buffers: v2 masked-dist+topk); "
                         "removes the bin-every host sync + mu2d/keep download")
    ap.add_argument("--device-scatter", action="store_true",
                    help="run the bin->gaussian grad scatter ON DEVICE (traced gather-reduce over sinv_u); "
                         "removes the per-iter rend-bwd sync + host index_add. arm-PW packs a 10th (gmcz_occ) channel")
    ap.add_argument("--fused-adam", action="store_true",
                    help="batch the 14 params into one [G,14] Adam update (~150 ops -> ~10); kills the "
                         "untraced-Adam dispatch cost (was ~35-54%% of per-iter time)")
    args = ap.parse_args()
    assert sum([args.depth_weight, args.arm_softmin, args.arm_pairwise]) <= 1, \
        "--depth-weight, --arm-softmin, --arm-pairwise are mutually exclusive"
    if args.colmap:
        args.random_bg = False                # real scenes have full backgrounds (no alpha) -> no random-bg
    res, G, K = args.res, args.G, args.K
    out_dir = os.path.join(args.out, f"{args.scene_name}_G{G}")
    os.makedirs(out_dir, exist_ok=True)

    tr_pm, tr_tr = None, None
    if args.colmap:                                    # real scene: COLMAP poses + held-out split + point-init
        cams_all, imgs_all = data.load_colmap(args.colmap, downscale=args.downscale)
        Hc, Wc = (cams_all[0].H // 16) * 16, (cams_all[0].W // 16) * 16     # crop to whole 16x16 tiles
        from spike.camera import Camera
        def _crop(c, im):
            return Camera(c.R_v, c.t_v, c.fx, c.fy, c.cx, c.cy, Hc, Wc), im[:Hc, :Wc, :].contiguous()
        test_set = set(range(0, len(cams_all), 8))     # every 8th view held out (standard 3DGS protocol)
        tr, te = [], []
        for i in range(len(cams_all)):
            (te if i in test_set else tr).append(_crop(cams_all[i], imgs_all[i]))
        tr_c, tr_i = [c for c, _ in tr], [im for _, im in tr]
        te_c, te_i = [c for c, _ in te], [im for _, im in te]
        res = max(Hc, Wc)
        tmap = TileMap(Hc, Wc); T, H, W = tmap.T, Hc, Wc
        torch.manual_seed(0)
        m = GaussianModel(G, seed=0)
        pxyz, prgb = data.load_colmap_points(args.colmap)
        m.init_from_points(torch.from_numpy(pxyz), torch.from_numpy(prgb))
        print(f"[colmap] {args.colmap} ds={args.downscale} -> {Hc}x{Wc} | {len(tr_c)} train, {len(te_c)} test "
              f"| G={G} init {len(pxyz)} pts", flush=True)
    else:
        tr_c, tr_i = data.load_blender(args.scene, "train", res=res, n=args.n_train, stride=max(1, 100 // args.n_train))
        te_c, te_i = data.load_blender(args.scene, "test", res=res)
        # random-bg: keep per-view premultiplied object color + transmittance (1-a) for per-iter compositing.
        if args.random_bg:
            _, tr_rgba = data.load_blender(args.scene, "train", res=res, n=args.n_train,
                                           stride=max(1, 100 // args.n_train), keep_alpha=True)
            tr_pm = [(im[..., :3] * im[..., 3:4]).permute(2, 0, 1).contiguous() for im in tr_rgba]   # [3,H,W]
            tr_tr = [(1.0 - im[..., 3:4]).expand(-1, -1, 3).permute(2, 0, 1).contiguous() for im in tr_rgba]
        tmap = TileMap(res, res); T, H, W = tmap.T, res, res
        torch.manual_seed(0)
        m = GaussianModel(G, extent=1.5, seed=0)
    Ntr = len(tr_c)
    print(f"[fast] {args.scene_name} G={G} K={K} res={H}x{W} iters={args.iters} | {Ntr} train, {len(te_c)} test", flush=True)

    # Canonical paths for eval.json / panels / rollup (uses same res as training)
    paths = evalcard.run_paths(args.method, args.scene_name, G, res,
                               iters=args.iters, K=K, root=args.out)
    os.makedirs(paths["dir"], exist_ok=True)

    LR = {**{k: DEFAULT_LR["means"] for k in ("mx", "my", "mz")},
          **{k: DEFAULT_LR["quats"] for k in ("qw", "qx", "qy", "qz")},
          **{k: DEFAULT_LR["scales"] for k in ("lx", "ly", "lz")},
          **{k: DEFAULT_LR["color"] for k in ("cr", "cg", "cb")}, "op": DEFAULT_LR["opacity"]}
    w_b = float(torch.nn.functional.softplus(m.w_b_raw))

    DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        import resident_traced as mrt
        mrt.CG = CG                     # render_bwd/theta_bwd use resident_traced's module-global CG
        # ===== (A) buffers =====
        P = dict(mx=u(m.means3d[:, 0]), my=u(m.means3d[:, 1]), mz=u(m.means3d[:, 2]),
                 qw=u(m.quats[:, 0]), qx=u(m.quats[:, 1]), qy=u(m.quats[:, 2]), qz=u(m.quats[:, 3]),
                 lx=u(m.log_scales[:, 0]), ly=u(m.log_scales[:, 1]), lz=u(m.log_scales[:, 2]),
                 cr=u(m.color_dc[:, 0]), cg=u(m.color_dc[:, 1]), cb=u(m.color_dc[:, 2]), op=u(m.opacity_raw))
        mom = {k: u(torch.zeros(G)) for k in PN}; vom = {k: u(torch.zeros(G)) for k in PN}
        gacc = {k: u(torch.zeros(G)) for k in PN}
        tmp1 = {k: u(torch.zeros(G)) for k in PN}; tmp2 = {k: u(torch.zeros(G)) for k in PN}
        # fused Adam: merged [G,14] moments + [1,14] per-param LR row (batch 14 params -> 1 update)
        mom_m = u(torch.zeros(G, len(PN))); vom_m = u(torch.zeros(G, len(PN)))
        LR_vec = u(torch.tensor([[LR[k] for k in PN]], dtype=torch.float32))   # [1,14], broadcast over G
        idx_u = ttnn.from_torch(torch.zeros(T, K, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        valid6 = u(torch.zeros(T, 6, K), BF); vf = u(torch.zeros(T, K, 1), BF)
        origins_t = u(tmap.origins[:, None, :].expand(T, K, 2).contiguous(), BF)
        Phi = u(tmap.Phi.unsqueeze(0).expand(T, 256, 6).contiguous(), BF)
        bias = u((w_b * m.c_b)[None, None, :].expand(T, 256, 3).contiguous(), BF)
        wb_buf = u(torch.full((T, 256, 1), w_b), BF)
        bump = torch.zeros(T, K, 6); bump[..., 5] = 1.0; bump_buf = u(bump, BF)
        gC_buf = u(torch.zeros(T, 256, 3), BF)
        gcon_buf = {k: u(torch.zeros(G)) for k in ("a", "b", "c", "mx", "my")}
        gco_buf = [u(torch.zeros(G)) for _ in range(3)]; gocl_buf = u(torch.zeros(G))
        # device-scatter (bin->gaussian grad gather-reduce on device): inv[G,SMAX] slot table + zero
        # sentinel row. arm-PW packs a 10th channel (gmcz_occ z-force); else 9 (a,b,c,mx,my,col0-2,op).
        SMAX = (2 * 1 + 1) ** 2                                   # R=1 stencil -> <=9 valid slots per gaussian
        NCH = 10 if args.arm_pairwise else 9
        sinv_u = zrow = None
        if args.device_scatter:
            sinv_u = ttnn.from_torch(torch.full((G, SMAX), T * K, dtype=torch.int32),
                                     dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
            zrow = u(torch.zeros(1, NCH), BF)                    # grad_pad sentinel (gathered by inv padding)
        # device-binning CONSTANTS preallocated ONCE (no per-bin-every from_torch -> trace/alloc safe).
        # Shared by bin_to_buffers + build_inv_device (both only run in the --device-binning path).
        bin_ctx = make_bin_ctx(tmap, G, K, DEV) if args.device_binning else None
        # depth-weight lever: rho=sigmoid(beta*(tau-z)), sort-free, multiplies o.
        # beta/tau are global learnable scalars (broadcast to [G,1] device buffers, host-Adam updated per-iter).
        beta_buf = u(torch.full((G,), args.beta0)); tau_buf = u(torch.full((G,), args.tau0))
        gbeta_buf = u(torch.zeros(G)); gtau_buf = u(torch.zeros(G))
        bt = {"beta": args.beta0, "tau": args.tau0}
        bt_m = {"beta": 0.0, "tau": 0.0}; bt_v = {"beta": 0.0, "tau": 0.0}
        # arm-softmin lever: rho=exp(-(mcz-zref)/tau), sort-free, z ATTACHED (C3 z-force).
        # inv_tau=1/tau, inv_tau2=1/tau^2 stored as [G,1] buffers (broadcast scalars);
        # zref_buf = warmup median visible mcz, refreshed every ~500 iters.
        if args.arm_softmin:
            _sm_tau0 = args.smtau0
            inv_tau_buf = u(torch.full((G,), 1.0 / _sm_tau0))
            inv_tau2_buf = u(torch.full((G,), 1.0 / (_sm_tau0 ** 2)))
            zref_buf = u(torch.zeros(G))                         # filled at warmup
            gsmtau_buf = u(torch.zeros(G))
            sm = {"tau": _sm_tau0}
            sm_m = {"tau": 0.0}; sm_v = {"tau": 0.0}
        # arm-pairwise lever: per-tile [K,K] soft-compare GEMM S[g,h]=sigmoid((zw_g-zw_h)/tau),
        # OIT-over composite. tau buffers fully replicated to consumption shape (not [1,1]
        # broadcast, mirrors the beta/tau/inv_tau full-replicate convention above).
        if args.arm_pairwise:
            NEAR_PW, FAR_PW = 0.5, 8.0
            _pw_tau0 = args.pwtau0
            pw_inv_tau_buf = u(torch.full((T, K, K), 1.0 / _pw_tau0), DT)       # for u_arg = d_raw * inv_tau
            pw_inv_tau_k1_buf = u(torch.full((T, K, 1), 1.0 / _pw_tau0), DT)    # for gzw = (rowsum-colsum) * inv_tau
            pw_inv_tau2_buf = u(torch.full((T, 1, 1), 1.0 / (_pw_tau0 ** 2)), DT)  # for gtau coefficient
            one_minus_I_buf = u((1.0 - torch.eye(K))[None].expand(T, K, K).contiguous(), DT)  # constant, never updated
            cb_bcast = u(m.c_b[None, None, :].expand(T, 256, 3).contiguous(), BF)  # constant c_b (not w_b-scaled)
            gmcz_occ_buf = u(torch.zeros(G))          # [G,1] per-gaussian z-force grad accumulator (host-scattered)
            gpwtau_buf = u(torch.zeros(T, 1, 1), DT)  # [T,1,1] per-tile tau-grad, host-summed over T
            pw = {"tau": _pw_tau0}
            pw_m = {"tau": 0.0}; pw_v = {"tau": 0.0}
        # camera buffers [G,1]
        Rvb = [[u(torch.zeros(G)) for _ in range(3)] for _ in range(3)]
        tvb = [u(torch.zeros(G)) for _ in range(3)]
        fxb, fyb, cxb, cyb = u(torch.zeros(G)), u(torch.zeros(G)), u(torch.zeros(G)), u(torch.zeros(G))
        # loss consts
        g1d = gauss_1d(); Mh_np, Mw_np = band_matrix(H, g1d), band_matrix(W, g1d)
        Hout, Win = Mh_np.shape[0], Mw_np.shape[1]
        Ns_loss = float(3 * Mh_np.shape[0] * Mh_np.shape[0]); N_loss = float(3 * H * W)
        Mh_b = u(Mh_np.unsqueeze(0).expand(3, *Mh_np.shape).contiguous())
        MwT_b = u(Mw_np.t().unsqueeze(0).expand(3, Win, Mw_np.shape[0]).contiguous())
        MhT_b = u(Mh_np.t().unsqueeze(0).expand(3, Mh_np.shape[1], Hout).contiguous())
        Mw_b = u(Mw_np.unsqueeze(0).expand(3, *Mw_np.shape).contiguous())
        y_d = u(torch.zeros(3, H, W))                                  # current-view gt (device), set per iter
        inv = torch.empty(H * W, dtype=torch.long); inv[tmap.gidx] = torch.arange(T * 256)
        gidx_u = ttnn.from_torch(tmap.gidx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        inv_u = ttnn.from_torch(inv.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        # per-view gt resident (device), computed once
        gt_res = [u(tr_i[v].permute(2, 0, 1).contiguous()) for v in range(Ntr)]

        def setcam(camlist, v):
            cam = camlist[v]
            Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
            tv = [float(cam.t_v[i]) for i in range(3)]
            for i in range(3):
                for j in range(3):
                    ttnn.copy_host_to_device_tensor(uh(torch.full((G,), Rv[i][j])), Rvb[i][j])
                ttnn.copy_host_to_device_tensor(uh(torch.full((G,), tv[i])), tvb[i])
            for buf, val in ((fxb, cam.fx), (fyb, cam.fy), (cxb, cam.cx), (cyb, cam.cy)):
                ttnn.copy_host_to_device_tensor(uh(torch.full((G,), float(val))), buf)

        def _setcam_direct(cam):
            """Set camera buffers from a camera object directly (not from a list)."""
            Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
            tv = [float(cam.t_v[i]) for i in range(3)]
            for i in range(3):
                for j in range(3):
                    ttnn.copy_host_to_device_tensor(uh(torch.full((G,), Rv[i][j])), Rvb[i][j])
                ttnn.copy_host_to_device_tensor(uh(torch.full((G,), tv[i])), tvb[i])
            for buf, val in ((fxb, cam.fx), (fyb, cam.fy), (cxb, cam.cx), (cyb, cam.cy)):
                ttnn.copy_host_to_device_tensor(uh(torch.full((G,), float(val))), buf)

        def geom_fwd():
            sx, sy, sz = ttnn.exp(P["lx"]), ttnn.exp(P["ly"]), ttnn.exp(P["lz"])
            cols = (P["mx"], P["my"], P["mz"], P["qw"], P["qx"], P["qy"], P["qz"], sx, sy, sz)
            conic, mu2d, cache = device_fwd_core(cols, Rvb, tvb, fxb, fyb, cxb, cyb)
            keep = cache["zmask"]
            o = ttnn.sigmoid(P["op"])
            if args.depth_weight:
                # depth = camera-space mcz (NOT cache["z"], which is the normalized quaternion z-comp).
                # depth-detached: rho re-weights o by depth but we do not backprop rho->mcz->means.
                rho = ttnn.sigmoid(M(beta_buf, A(tau_buf, ttnn.neg(cache["mcz"]))))   # sigmoid(beta*(tau-depth))
                cache["rho"] = rho
                keo = M(M(keep, o), rho)
            elif args.arm_softmin:
                # rho_sm = exp(-(mcz - zref) / tau); z ATTACHED (C3) -> z-force in geom_bwd.
                # zref_buf is a [G,1] broadcast scalar (median visible mcz, refreshed ~500 iters).
                rho_sm = ttnn.exp(M(A(cache["mcz"], ttnn.neg(zref_buf)), ttnn.neg(inv_tau_buf)))
                cache["rho"] = rho_sm
                keo = M(M(keep, o), rho_sm)
            else:
                keo = M(keep, o)
            color = [ttnn.relu(A(M(P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
            color_o = ttnn.concat([M(keo, color[0]), M(keo, color[1]), M(keo, color[2])], dim=-1)
            return conic, mu2d, cache, keep, keo, o, color, color_o

        conic, mu2d, cache, keep, keo, o, color, color_o = geom_fwd()

        def render_fwd_cache(thU, col, oc):
            relu_Q = ttnn.relu(ttnn.matmul(Phi, thU, core_grid=CG))
            w = ttnn.square(relu_Q)
            den = ttnn.add(ttnn.matmul(w, oc, core_grid=CG), wb_buf)
            num = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)
            return ttnn.div(num, den), (relu_Q, w, den, num)

        def render_fwd_pw(thU, col, oc, mcz):
            """PW forward (candidate #5 redux, per-tile [T,256,K]): pairwise soft-occlusion via a
            [K,K] soft-compare GEMM on the true per-(pixel,gaussian) absorbance. REPLACES
            render_fwd_cache entirely for this arm -- no WSR den/div; OIT-over composite with
            residual background transmittance T_bg*c_b. z ATTACHED through zw_t (C3)."""
            relu_Q = ttnn.relu(ttnn.matmul(Phi, thU, core_grid=CG))
            w = ttnn.square(relu_Q)                                            # [T,256,K] w_geo
            oc_row = T3(oc)                                                    # [T,1,K]
            alpha_raw = ttnn.mul(w, oc_row)                                    # [T,256,K] broadcast dim-2
            alpha_pg = ttnn.clamp(alpha_raw, 1e-6, 1.0 - 1e-4)
            alpha_gate = ttnn.mul(ttnn.gtz(ttnn.add(alpha_raw, -1e-6)),
                                  ttnn.gtz(ttnn.add(ttnn.neg(alpha_raw), 1.0 - 1e-4)))
            one_minus_alpha = ttnn.add(ttnn.neg(alpha_pg), 1.0)
            a_pg = ttnn.neg(ttnn.log(one_minus_alpha))                         # [T,256,K] -log(1-alpha)

            # ttnn.embedding requires BFLOAT16 weights -- one unavoidable bf16 round-trip on the
            # gather itself, but immediately upcast back to fp32 so the tau=0.01-sharp sigmoid
            # compare below doesn't compound further bf16 rounding on top of it.
            z_t = ttnn.typecast(ttnn.embedding(idx_u, ttnn.typecast(mcz, BF)), DT)  # [T,K,1]
            zw_raw = ttnn.mul(ttnn.add(z_t, -NEAR_PW), 1.0 / (FAR_PW - NEAR_PW))
            zw_t = ttnn.clamp(zw_raw, 0.0, 1.0)                                # [T,K,1]
            zw_gate = ttnn.mul(ttnn.gtz(zw_raw), ttnn.gtz(ttnn.add(ttnn.neg(zw_raw), 1.0)))
            zw_row = T3(zw_t)                                                  # [T,1,K]
            d_raw = ttnn.add(zw_t, ttnn.neg(zw_row))                           # [T,K,K] d_raw[g,h]=zw[g]-zw[h]
            u_arg = ttnn.mul(d_raw, pw_inv_tau_buf)                            # [T,K,K]
            S_raw = ttnn.sigmoid(u_arg)
            S = ttnn.mul(S_raw, one_minus_I_buf)                               # [T,K,K] diagonal zeroed

            S_bf = ttnn.typecast(S, BF)
            logT_raw = ttnn.neg(ttnn.matmul(a_pg, T3(S_bf), core_grid=CG))     # [T,256,K]
            logT_gate = ttnn.gtz(ttnn.add(logT_raw, 30.0))                     # 1 where logT_raw > -30
            logT = ttnn.clamp(logT_raw, -30.0, 0.0)                            # upper bound 0.0 exact (a>=0,S>=0)
            Tt = ttnn.exp(logT)                                                # [T,256,K]
            wT = ttnn.mul(w, Tt)                                               # [T,256,K]
            num = ttnn.matmul(wT, col, core_grid=CG)                           # [T,256,3]
            a_sum = ttnn.sum(a_pg, dim=-1, keepdim=True)                       # [T,256,1]
            T_bg = ttnn.exp(ttnn.neg(a_sum))                                   # [T,256,1]
            C = ttnn.add(num, ttnn.mul(T_bg, cb_bcast))                        # [T,256,3]

            cache = dict(w=w, Tt=Tt, wT=wT, a_pg=a_pg, S=S, d_raw=d_raw, T_bg=T_bg,
                        alpha_gate=alpha_gate, logT_gate=logT_gate, zw_gate=zw_gate,
                        relu_Q=relu_Q, col=col, oc_row=oc_row)
            return C, cache

        def render_bwd_pw(gC, cache):
            """PW backward. Returns (gthU, gcol, goc, gmcz_occ_t) -- gthU/gcol/goc feed the existing
            theta_bwd / gco_buf / gocl_buf machinery unchanged; gmcz_occ_t[T,K,1] is NEW and is
            host-scattered to gmcz_occ_buf[G,1] before geom_bwd runs (the C3 z-force)."""
            w, Tt, wT = cache["w"], cache["Tt"], cache["wT"]
            a_pg, S, d_raw, T_bg = cache["a_pg"], cache["S"], cache["d_raw"], cache["T_bg"]
            alpha_gate, logT_gate = cache["alpha_gate"], cache["logT_gate"]
            relu_Q, col = cache["relu_Q"], cache["col"]

            gT_bg = ttnn.sum(ttnn.mul(gC, cb_bcast), dim=-1, keepdim=True)     # [T,256,1]
            gwT = ttnn.matmul(gC, T3(col), core_grid=CG)                       # [T,256,K]
            gcol = ttnn.matmul(T3(wT), gC, core_grid=CG)                       # [T,K,3]
            gw_num = ttnn.mul(gwT, Tt)                                         # [T,256,K]
            gTt = ttnn.mul(gwT, w)                                             # [T,256,K]
            glogT = ttnn.mul(gTt, ttnn.mul(Tt, logT_gate))                     # [T,256,K]
            S_bf = ttnn.typecast(S, BF)
            ga_logT = ttnn.neg(ttnn.matmul(glogT, S_bf, core_grid=CG))         # [T,256,K] (no transpose on S)
            gS_raw = ttnn.neg(ttnn.matmul(T3(glogT), a_pg, core_grid=CG))      # [T,K,K] (sums over the 256 pixels)
            gS = ttnn.typecast(gS_raw, DT)

            ga_bg = ttnn.mul(ttnn.neg(gT_bg), T_bg)                            # [T,256,1]
            ga = ttnn.add(ga_logT, ga_bg)                                      # [T,256,K] broadcast last-dim
            galpha = ttnn.mul(ttnn.mul(ga, ttnn.exp(a_pg)), alpha_gate)        # ga*1/(1-alpha)*gate (1/(1-a)=exp(a_pg))
            gw_alpha = ttnn.mul(galpha, cache["oc_row"])                       # [T,256,K]
            goc_pg = ttnn.mul(galpha, w)                                       # [T,256,K]
            goc = T3(ttnn.sum(goc_pg, dim=1, keepdim=True))                    # [T,K,1]
            gw = ttnn.add(gw_num, gw_alpha)                                    # [T,256,K]

            gu = ttnn.mul(gS, ttnn.mul(S, ttnn.add(ttnn.neg(S), 1.0)))         # [T,K,K] gS*S*(1-S)
            rowsum = ttnn.sum(gu, dim=-1, keepdim=True)                        # [T,K,1]
            colsum = T3(ttnn.sum(gu, dim=1, keepdim=True))                     # [T,K,1]
            gzw = ttnn.mul(ttnn.add(rowsum, ttnn.neg(colsum)), pw_inv_tau_k1_buf)   # [T,K,1]
            gmcz_occ_t = ttnn.mul(ttnn.mul(gzw, cache["zw_gate"]), 1.0 / (FAR_PW - NEAR_PW))  # [T,K,1]

            gu_d = ttnn.mul(gu, d_raw)                                         # [T,K,K]
            gtau_step1 = ttnn.sum(gu_d, dim=-1, keepdim=True)                  # [T,K,1]
            gtau_step2 = ttnn.sum(gtau_step1, dim=1, keepdim=True)             # [T,1,1]
            ttnn.copy(ttnn.mul(gtau_step2, ttnn.neg(pw_inv_tau2_buf)), gpwtau_buf)

            gthU = ttnn.matmul(T3(Phi), ttnn.mul(gw, ttnn.mul(relu_Q, 2.0)), core_grid=CG)
            return gthU, gcol, goc, gmcz_occ_t

        def rend_fwd():
            theta, col_t, oc_t = gather_theta_buf(conic, mu2d, color_o, keo, idx_u, valid6, vf, origins_t, bump_buf, T, K)
            if args.arm_pairwise:
                Cc, rc = render_fwd_pw(theta, col_t, oc_t, cache["mcz"])
            else:
                Cc, rc = render_fwd_cache(theta, col_t, oc_t)
            return Cc, rc, theta, col_t, oc_t

        C, rc, theta, col_t, oc_t = rend_fwd()

        def filt(t):   return ttnn.matmul(ttnn.matmul(Mh_b, t), MwT_b)
        def filt_T(t): return ttnn.matmul(ttnn.matmul(MhT_b, t), Mw_b)

        def loss_grad_dev():
            Cf = ttnn.reshape(C, (T * 256, 3))
            img = ttnn.embedding(inv_u, Cf)
            img = ttnn.reshape(ttnn.transpose(img, -2, -1), (3, H, W))
            x = ttnn.typecast(img, DT)
            fx, fx2, fxy = filt(x), filt(ttnn.mul(x, x)), filt(ttnn.mul(x, y_d))
            fy = filt(y_d); muy = fy; muy2 = ttnn.mul(fy, fy); sy = ttnn.sub(filt(ttnn.mul(y_d, y_d)), muy2)
            mux = fx; mux2 = ttnn.mul(mux, mux)
            sx = ttnn.sub(fx2, mux2); sxy = ttnn.sub(fxy, ttnn.mul(mux, muy))
            A1 = ttnn.add(ttnn.mul(ttnn.mul(mux, muy), 2.0), L_C1)
            A2 = ttnn.add(ttnn.mul(sxy, 2.0), L_C2)
            Bd1 = ttnn.add(ttnn.add(mux2, muy2), L_C1); Bd2 = ttnn.add(ttnn.add(sx, sy), L_C2)
            D = ttnn.mul(Bd1, Bd2); S = ttnn.div(ttnn.mul(A1, A2), D)
            dS_dfx2 = ttnn.neg(ttnn.div(S, Bd2)); dS_dfxy = ttnn.div(ttnn.mul(A1, 2.0), D)
            term = ttnn.sub(ttnn.mul(muy, ttnn.sub(A2, A1)), ttnn.mul(ttnn.mul(S, mux), ttnn.sub(Bd2, Bd1)))
            dS_dfx = ttnn.mul(ttnn.div(term, D), 2.0)
            dmeanS = ttnn.div(ttnn.add(ttnn.add(filt_T(dS_dfx), ttnn.mul(ttnn.mul(x, filt_T(dS_dfx2)), 2.0)),
                                       ttnn.mul(y_d, filt_T(dS_dfxy))), Ns_loss)
            g_ssim = ttnn.mul(dmeanS, -L_LAM)
            g_l1 = ttnn.mul(ttnn.sign(ttnn.sub(x, y_d)), (1.0 - L_LAM) / N_loss)
            gimg = ttnn.add(g_l1, g_ssim)
            gflat = ttnn.transpose(ttnn.reshape(ttnn.typecast(gimg, BF), (3, H * W)), -2, -1)
            gCt = ttnn.reshape(ttnn.embedding(gidx_u, gflat), (T, 256, 3))
            ttnn.copy(gCt, gC_buf)
            return gCt

        def rend_bwd():
            conic_t = ttnn.embedding(idx_u, ttnn.typecast(conic, BF))
            mu_t = ttnn.embedding(idx_u, ttnn.typecast(mu2d, BF))
            if args.arm_pairwise:
                gthU, gcol, goc, gmcz_occ_t = render_bwd_pw(gC_buf, rc)
            else:
                gthU, gcol, goc = render_bwd(Phi, col_t, oc_t, gC_buf, rc)
                gmcz_occ_t = None
            gct, gmt = theta_bwd(gthU, conic_t, mu_t, origins_t, valid6, T, K)
            return gct, gmt, gcol, goc, gmcz_occ_t

        gct, gmt, gcol, goc, gmcz_occ_t = rend_bwd()

        def scatter_dev():
            """bin->gaussian grad scatter ON DEVICE: gather each gaussian's <=SMAX slot grads via the
            sinv_u inv-table + sum. Replaces the host index_add round-trip -> no per-iter sync. Writes the
            same geom-bwd grad buffers the host scatter did: (a,b,c,mx,my,col0,col1,col2,op) [+ gmcz_occ
            for arm-PW]. Invalid slots are excluded by sinv_u (sentinel row -> zrow), so no vf mask needed."""
            chans = [ttnn.typecast(gct, BF), ttnn.typecast(gmt, BF),
                     ttnn.typecast(gcol, BF), ttnn.typecast(goc, BF)]
            dests = [gcon_buf["a"], gcon_buf["b"], gcon_buf["c"], gcon_buf["mx"], gcon_buf["my"],
                     gco_buf[0], gco_buf[1], gco_buf[2], gocl_buf]
            if args.arm_pairwise:
                chans.append(ttnn.typecast(gmcz_occ_t, BF))       # 10th channel: PW z-force
                dests.append(gmcz_occ_buf)
            gcat = ttnn.concat(chans, dim=-1)                     # [T,K,NCH]
            gpad = ttnn.concat([ttnn.reshape(gcat, (T * K, NCH)), zrow], dim=0)   # [T*K+1,NCH]
            g = ttnn.sum(ttnn.typecast(ttnn.embedding(sinv_u, gpad), DT), dim=1, keepdim=False)  # [G,NCH]
            for c, b in zip(ttnn.split(g, 1, dim=1), dests):
                ttnn.copy(c, b)
            return g

        def geom_bwd():
            gg = device_bwd_core(cache, gcon_buf["a"], gcon_buf["b"], gcon_buf["c"], gcon_buf["mx"], gcon_buf["my"])
            sx, sy, sz = ttnn.exp(P["lx"]), ttnn.exp(P["ly"]), ttnn.exp(P["lz"])
            out = {"mx": gg["gmx"], "my": gg["gmy"], "mz": gg["gmz"], "qw": gg["gqw"], "qx": gg["gqx"],
                   "qy": gg["gqy"], "qz": gg["gqz"], "lx": M(gg["gsx"], sx), "ly": M(gg["gsy"], sy), "lz": M(gg["gsz"], sz)}
            gkeo = A(A(M(gco_buf[0], color[0]), M(gco_buf[1], color[1])), A(M(gco_buf[2], color[2]), gocl_buf))
            cmask = [ttnn.gtz(A(M(P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
            out["cr"] = M(M(M(gco_buf[0], keo), cmask[0]), C0)
            out["cg"] = M(M(M(gco_buf[1], keo), cmask[1]), C0)
            out["cb"] = M(M(M(gco_buf[2], keo), cmask[2]), C0)
            one = ttnn.add(ttnn.mul(o, 0.0), 1.0)
            o1mo = M(o, ttnn.add(one, ttnn.neg(o)))            # sigmoid'(op) = o(1-o)
            if args.depth_weight:
                rho = cache["rho"]
                out["op"] = M(M(M(gkeo, keep), o1mo), rho)     # keo=keep*o*rho -> extra rho factor
                grad_rho = M(M(gkeo, keep), o)                 # dL/drho
                grad_pre = M(grad_rho, M(rho, A(one, ttnn.neg(rho))))   # *rho(1-rho)
                ttnn.copy(M(grad_pre, A(tau_buf, ttnn.neg(cache["mcz"]))), gbeta_buf)  # grad_beta_pg = grad_pre*(tau-depth)
                ttnn.copy(M(grad_pre, beta_buf), gtau_buf)                           # grad_tau_pg  = grad_pre*beta
            elif args.arm_softmin:
                rho_sm = cache["rho"]
                # opacity grad: keo=keep*o*rho_sm -> extra rho_sm factor (mirrors arm B)
                out["op"] = M(M(M(gkeo, keep), o1mo), rho_sm)
                grad_rho = M(M(gkeo, keep), o)                 # dL/drho_sm  [G,1]
                # tau grad: d(rho_sm)/d(tau) = rho_sm*(mcz-zref)/tau^2
                ttnn.copy(M(grad_rho, M(rho_sm, M(A(cache["mcz"], ttnn.neg(zref_buf)), inv_tau2_buf))),
                          gsmtau_buf)
                # z-force (C3): d(rho_sm)/d(mcz) = -rho_sm/tau -> backprop into means
                # mcz = Rv[2][0]*mx + Rv[2][1]*my + Rv[2][2]*mz + tv[2]
                grad_mcz_occ = M(grad_rho, M(rho_sm, ttnn.neg(inv_tau_buf)))   # grad_rho*(-rho_sm/tau)
                out["mx"] = A(out["mx"], M(grad_mcz_occ, Rvb[2][0]))
                out["my"] = A(out["my"], M(grad_mcz_occ, Rvb[2][1]))
                out["mz"] = A(out["mz"], M(grad_mcz_occ, Rvb[2][2]))
            else:
                out["op"] = M(M(gkeo, keep), o1mo)
            if args.arm_pairwise:
                # z-force (C3): gmcz_occ_buf is the host-scattered per-gaussian sum of
                # render_bwd_pw's per-tile-slot gmcz_occ_t (train loop scatters it in before
                # this trace runs). Purely additive on top of whichever opacity branch ran above
                # (PW itself falls into the plain `else` branch -- no rho fold-in in geom_fwd).
                out["mx"] = A(out["mx"], M(gmcz_occ_buf, Rvb[2][0]))
                out["my"] = A(out["my"], M(gmcz_occ_buf, Rvb[2][1]))
                out["mz"] = A(out["mz"], M(gmcz_occ_buf, Rvb[2][2]))
            # MCMC regularization gradients (density control): opacity-L1 + scale-L1.
            # dReg/dop_raw = lam_o/G * o(1-o);  dReg/dlx = lam_s/(3G) * exp(lx)=sx  (mean over G / 3G)
            lo, ls = args.lam_o / G, args.lam_s / (3.0 * G)
            if lo > 0:
                out["op"] = ttnn.add(out["op"], ttnn.mul(o1mo, lo))
            if ls > 0:
                out["lx"] = ttnn.add(out["lx"], ttnn.mul(sx, ls))
                out["ly"] = ttnn.add(out["ly"], ttnn.mul(sy, ls))
                out["lz"] = ttnn.add(out["lz"], ttnn.mul(sz, ls))
            return out

        gout = geom_bwd()

        def adam_inplace(t):
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            for k in PN:
                g, mk, vk, p, t1, t2 = gacc[k], mom[k], vom[k], P[k], tmp1[k], tmp2[k]
                ttnn.mul(mk, B1, output_tensor=mk)
                ttnn.mul(g, 1.0 - B1, output_tensor=t1); ttnn.add(mk, t1, output_tensor=mk)
                ttnn.mul(vk, B2, output_tensor=vk)
                ttnn.mul(g, g, output_tensor=t1); ttnn.mul(t1, 1.0 - B2, output_tensor=t1); ttnn.add(vk, t1, output_tensor=vk)
                ttnn.sqrt(vk, output_tensor=t1); ttnn.mul(t1, 1.0 / math.sqrt(bc2), output_tensor=t1); ttnn.add(t1, EPS, output_tensor=t1)
                ttnn.mul(mk, LR[k] / bc1, output_tensor=t2); ttnn.div(t2, t1, output_tensor=t2)
                ttnn.subtract(p, t2, output_tensor=p)

        def adam_fused(t):
            """Batched Adam: concat the 14 grads -> [G,14], one update, split -> subtract into each P[k].
            ~140 tiny ops -> ~10, killing the small-model dispatch bottleneck. Reads gout directly (no gacc)."""
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            g = ttnn.concat([gout[k] for k in PN], dim=1)                              # [G,14]
            ttnn.mul(mom_m, B1, output_tensor=mom_m)
            ttnn.add(mom_m, ttnn.mul(g, 1.0 - B1), output_tensor=mom_m)                # m = B1 m + (1-B1) g
            ttnn.mul(vom_m, B2, output_tensor=vom_m)
            ttnn.add(vom_m, ttnn.mul(ttnn.mul(g, g), 1.0 - B2), output_tensor=vom_m)   # v = B2 v + (1-B2) g^2
            denom = ttnn.add(ttnn.mul(ttnn.sqrt(vom_m), 1.0 / math.sqrt(bc2)), EPS)    # sqrt(v)/sqrt(bc2)+eps
            step = ttnn.div(ttnn.mul(ttnn.mul(mom_m, LR_vec), 1.0 / bc1), denom)       # LR*m/bc1 / denom  [G,14]
            for k, c in zip(PN, ttnn.split(step, 1, dim=1)):
                ttnn.subtract(P[k], c, output_tensor=P[k])

        def adam_bt(t):
            """host-side Adam for the 2 global depth-weight scalars (grads reduced from [G] device buffers)."""
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            grads = {"beta": float(dn(gbeta_buf).sum()), "tau": float(dn(gtau_buf).sum())}
            for k in ("beta", "tau"):
                g = grads[k]
                bt_m[k] = B1 * bt_m[k] + (1 - B1) * g
                bt_v[k] = B2 * bt_v[k] + (1 - B2) * g * g
                bt[k] -= args.lr_bt * (bt_m[k] / bc1) / (math.sqrt(bt_v[k] / bc2) + EPS)
            setbuf(beta_buf, torch.full((G,), bt["beta"]), DT)
            setbuf(tau_buf, torch.full((G,), bt["tau"]), DT)

        def adam_sm(t):
            """host-side Adam for the softmin tau scalar (grad reduced from [G] device buffer sum)."""
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            g = float(dn(gsmtau_buf).sum())
            sm_m["tau"] = B1 * sm_m["tau"] + (1 - B1) * g
            sm_v["tau"] = B2 * sm_v["tau"] + (1 - B2) * g * g
            sm["tau"] -= args.lr_smtau * (sm_m["tau"] / bc1) / (math.sqrt(sm_v["tau"] / bc2) + EPS)
            sm["tau"] = max(sm["tau"], 1e-3)             # clamp tau strictly positive
            setbuf(inv_tau_buf, torch.full((G,), 1.0 / sm["tau"]), DT)
            setbuf(inv_tau2_buf, torch.full((G,), 1.0 / (sm["tau"] ** 2)), DT)

        def adam_pw(t):
            """host-side Adam for the global pw_tau scalar (grad reduced from [T,1,1] device buffer sum)."""
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            g = float(dn(gpwtau_buf).sum())          # sum over T tiles
            pw_m["tau"] = B1 * pw_m["tau"] + (1 - B1) * g
            pw_v["tau"] = B2 * pw_v["tau"] + (1 - B2) * g * g
            pw["tau"] -= args.lr_pwtau * (pw_m["tau"] / bc1) / (math.sqrt(pw_v["tau"] / bc2) + EPS)
            pw["tau"] = max(pw["tau"], 1e-3)             # clamp tau strictly positive
            # Update the full-replicate tau buffers by an IN-PLACE DEVICE scalar fill (mul-by-0 then
            # add-scalar) instead of a host torch.full + copy_host_to_device. pw_inv_tau_buf is [T,K,K]
            # (=67 MB fp32 at res512/K128) -> the host upload was ~40 ms/iter and dominated the Adam
            # bucket; the device fill costs a couple ms and moves no data over PCIe.
            inv_tau, inv_tau2 = 1.0 / pw["tau"], 1.0 / (pw["tau"] ** 2)
            for buf, val in ((pw_inv_tau_buf, inv_tau), (pw_inv_tau_k1_buf, inv_tau), (pw_inv_tau2_buf, inv_tau2)):
                ttnn.multiply(buf, 0.0, output_tensor=buf); ttnn.add(buf, val, output_tensor=buf)

        def _logit(p):
            return torch.log(p / (1.0 - p))

        GEOM_K = ("mx", "my", "mz", "qw", "qx", "qy", "qz", "lx", "ly", "lz", "cr", "cg", "cb")

        @torch.no_grad()
        def relocate_host():
            """MCMC relocation (host branchy glue): move dead (o<thr) slots onto live gaussians
            sampled ∝ opacity, contribution-preserving o->o/n split. Resets Adam moments for touched."""
            ttnn.synchronize_device(DEV)
            vals = {k: dn(P[k]).reshape(-1).clone() for k in PN}
            o = torch.sigmoid(vals["op"])
            dead = torch.where(o < args.dead_thr)[0]
            alive = torch.where(o >= args.dead_thr)[0]
            if dead.numel() == 0 or alive.numel() == 0:
                return 0
            probs = o[alive] / o[alive].sum()
            targets = alive[torch.multinomial(probs, dead.numel(), replacement=True)]
            uniq, counts = torch.unique(targets, return_counts=True)
            n_at = {int(t): int(c) + 1 for t, c in zip(uniq.tolist(), counts.tolist())}
            new_o = {int(t): float((o[int(t)] / n_at[int(t)]).clamp(1e-6, 1 - 1e-6)) for t in uniq.tolist()}
            for t in uniq.tolist():
                vals["op"][t] = _logit(torch.tensor(new_o[t]))
            for d, t in zip(dead.tolist(), targets.tolist()):
                for k in GEOM_K:
                    vals[k][d] = vals[k][t]
                vals["mx"][d] += 0.005 * float(torch.randn(())); vals["my"][d] += 0.005 * float(torch.randn(()))
                vals["mz"][d] += 0.005 * float(torch.randn(()))
                vals["op"][d] = _logit(torch.tensor(new_o[t]))
            for k in PN:
                setbuf(P[k], vals[k], DT)
            touched = torch.unique(torch.cat([dead, uniq]))
            if args.fused_adam:                                # fused Adam moments live in [G,14] merged buffers
                for Mm in (mom_m, vom_m):
                    mh = dn(Mm); mh[touched] = 0.0; setbuf(Mm, mh, DT)
            else:
                for k in PN:                                   # reset Adam moments for touched slots
                    mk = dn(mom[k]).reshape(-1).clone(); vk = dn(vom[k]).reshape(-1).clone()
                    mk[touched] = 0.0; vk[touched] = 0.0
                    setbuf(mom[k], mk, DT); setbuf(vom[k], vk, DT)
            return int(dead.numel())

        def set_sinv(idx, valid):
            """HOST build_inv -> upload sinv_u (device-scatter path when binning stays on host)."""
            setbuf(sinv_u, build_inv(idx, valid, G, SMAX).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        def do_bin():
            """Fill idx_u/valid6/vf for this iter's binning (+ sinv_u for device-scatter).
            --device-binning: bin ON DEVICE (bin_to_buffers) -> no host sync / no mu2d-keep download; with
            --device-scatter also build the inv-table on device (build_inv_device) -> no idx download at all.
            Else: host assign_bins + setbuf upload (original behaviour). Returns (idx,valid) host tensors,
            or (None,None) when the host needs neither (device-binning + device-scatter)."""
            if args.device_binning:
                bin_to_buffers(cache["mu2d_x"], cache["mu2d_y"], cache["zmask"], tmap, 1, K, DEV,
                               idx_u, valid6, vf, ctx=bin_ctx)
                if args.device_scatter:
                    build_inv_device(idx_u, ttnn.reshape(vf, (T, K)), cache["mu2d_x"], cache["mu2d_y"],
                                     tmap, K, DEV, sinv_u=sinv_u, ctx=bin_ctx)
                    return None, None
                ttnn.synchronize_device(DEV)
                idx = ttnn.to_torch(idx_u).long().reshape(T, K)
                valid = ttnn.to_torch(vf).float().reshape(T, K) > 0.5
            else:
                idx, valid = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
                setbuf(idx_u, idx.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                setbuf(valid6, valid[:, None, :].expand(T, 6, K).float()); setbuf(vf, valid[..., None].float())
            return idx, valid

        # ===== (B) warmup all traced fns (JIT) BEFORE capture, with view 0 =====
        setcam(tr_c, 0); ttnn.copy(gt_res[0], y_d)
        conic, mu2d, cache, keep, keo, o, color, color_o = geom_fwd()
        ttnn.synchronize_device(DEV)
        if args.depth_weight:                              # refine tau0 to the actual depth distribution
            zc = dn(cache["mcz"]).reshape(-1); kc = dn(keep).reshape(-1) > 0.5
            zv = zc[kc] if bool(kc.any()) else zc
            bt["tau"] = float(zv.median()); setbuf(tau_buf, torch.full((G,), bt["tau"]), DT)
            print(f"[fast] depth-weight ON: beta0={bt['beta']:.3f} tau0={bt['tau']:.3f} lr_bt={args.lr_bt} "
                  f"(z range {float(zv.min()):.2f}-{float(zv.max()):.2f})", flush=True)
        if args.arm_softmin:                               # set zref to median visible mcz
            zc = dn(cache["mcz"]).reshape(-1); kc = dn(keep).reshape(-1) > 0.5
            zv = zc[kc] if bool(kc.any()) else zc
            zref_val = float(zv.median())
            setbuf(zref_buf, torch.full((G,), zref_val), DT)
            print(f"[fast] arm-softmin ON: tau0={sm['tau']:.3f} zref={zref_val:.3f} lr_smtau={args.lr_smtau} "
                  f"(z range {float(zv.min()):.2f}-{float(zv.max()):.2f})", flush=True)
        idx, valid = do_bin()
        if args.device_scatter and not args.device_binning:
            set_sinv(idx, valid)                           # device-binning already built sinv_u on device
        C, rc, theta, col_t, oc_t = rend_fwd()
        loss_grad_dev(); gct, gmt, gcol, goc, gmcz_occ_t = rend_bwd()
        if args.device_scatter:
            scatter_dev()                                  # warmup: JIT concat/embedding/sum/split before capture
        gout = geom_bwd()
        ttnn.synchronize_device(DEV)

        # ===== (C) capture traces =====
        def cap(fn):
            tid = ttnn.begin_trace_capture(DEV, cq_id=0); r = fn(); ttnn.end_trace_capture(DEV, tid, cq_id=0); return tid, r
        gfid, (conic, mu2d, cache, keep, keo, o, color, color_o) = cap(geom_fwd)
        ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        rfid, (C, rc, theta, col_t, oc_t) = cap(rend_fwd)
        lfid, _ = cap(loss_grad_dev)
        rbid, (gct, gmt, gcol, goc, gmcz_occ_t) = cap(rend_bwd)
        sbid = None
        if args.device_scatter:
            sbid, _ = cap(scatter_dev)
        gbid, gout = cap(geom_bwd)
        ttnn.synchronize_device(DEV)

        def run_geom_bin(camlist, v):
            """set camera v, run geom-fwd + host binning (used by train + eval)."""
            setcam(camlist, v)
            ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
            idx, valid = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
            setbuf(idx_u, idx.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            setbuf(valid6, valid[:, None, :].expand(T, 6, K).float()); setbuf(vf, valid[..., None].float())
            return idx, valid

        @torch.no_grad()
        def eval_split(camlist, imglist):
            if args.random_bg:        # eval on the STANDARD white bg (matches gsplat / benchmark protocol)
                setbuf(bias, (w_b * torch.ones(3))[None, None, :].expand(T, 256, 3).contiguous(), BF)
            ps, ss = [], []
            for v in range(len(camlist)):
                run_geom_bin(camlist, v)
                ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
                img = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3))
                r = img.reshape(H, W, 3)
                ps.append(float(metrics.psnr(r, imglist[v]))); ss.append(float(metrics.ssim(r, imglist[v])))
            return sum(ps) / len(ps), sum(ss) / len(ss)

        # ===== (D) train loop =====
        idx, valid = run_geom_bin(tr_c, 0)
        t0 = time.perf_counter()
        PROF = os.environ.get("PW_PROFILE") == "1"
        _pacc = {}; _pn = [0]; _pt = [None]
        def mark(name):
            if not PROF:
                return
            ttnn.synchronize_device(DEV); now = time.perf_counter()
            if _pt[0] is not None and name is not None:
                _pacc[name] = _pacc.get(name, 0.0) + (now - _pt[0])
            _pt[0] = now
        for it in range(args.iters):
            if PROF and it == 10:
                _pacc.clear(); _pn[0] = 0
            if PROF:
                _pn[0] += 1
            mark(None)
            v = int(torch.randint(Ntr, (1,)).item())
            setcam(tr_c, v)
            if args.random_bg:                                 # composite GT over a random bg; c_B=bg
                bg = torch.rand(3)
                setbuf(y_d, tr_pm[v] + tr_tr[v] * bg[:, None, None], DT)
                setbuf(bias, (w_b * bg)[None, None, :].expand(T, 256, 3).contiguous(), BF)
            else:
                ttnn.copy(gt_res[v], y_d)
            mark("input")
            ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False)
            mark("geom_fwd")
            if it % args.bin_every == 0:
                if not args.device_binning:
                    ttnn.synchronize_device(DEV)           # host binning needs mu2d/keep back; device binning has no sync
                idx, valid = do_bin()
                if args.device_scatter and not args.device_binning:
                    set_sinv(idx, valid)                   # device-binning already built sinv_u on device in do_bin
                mark("bin")
            ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False)
            mark("rend_fwd")
            ttnn.execute_trace(DEV, lfid, cq_id=0, blocking=False)
            mark("loss")
            if args.device_scatter:
                # DEVICE scatter: rend-bwd -> device gather-reduce -> geom-bwd, all one cq, NO per-iter sync.
                ttnn.execute_trace(DEV, rbid, cq_id=0, blocking=False)
                mark("rend_bwd")
                ttnn.execute_trace(DEV, sbid, cq_id=0, blocking=False)
                mark("scatter")
            else:
                ttnn.execute_trace(DEV, rbid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
                mark("rend_bwd")
                flat = idx.reshape(-1); vfh = valid[..., None].float()
                gconic = torch.zeros(G, 3).index_add_(0, flat, dn(gct).reshape(-1, 3))
                gmu2d = torch.zeros(G, 2).index_add_(0, flat, dn(gmt).reshape(-1, 2))
                gcolor_o = torch.zeros(G, 3).index_add_(0, flat, (dn(gcol) * vfh).reshape(-1, 3))
                go_col = torch.zeros(G, 1).index_add_(0, flat, (dn(goc) * vfh).reshape(-1, 1))
                setbuf(gcon_buf["a"], gconic[:, 0], DT); setbuf(gcon_buf["b"], gconic[:, 1], DT); setbuf(gcon_buf["c"], gconic[:, 2], DT)
                setbuf(gcon_buf["mx"], gmu2d[:, 0], DT); setbuf(gcon_buf["my"], gmu2d[:, 1], DT)
                for j in range(3):
                    setbuf(gco_buf[j], gcolor_o[:, j], DT)
                setbuf(gocl_buf, go_col[:, 0], DT)
                if args.arm_pairwise:
                    gmcz_occ = torch.zeros(G, 1).index_add_(0, flat, (dn(gmcz_occ_t) * vfh).reshape(-1, 1))
                    setbuf(gmcz_occ_buf, gmcz_occ[:, 0], DT)
                mark("scatter")
            ttnn.execute_trace(DEV, gbid, cq_id=0, blocking=False)
            if not args.fused_adam:
                for k in PN:
                    ttnn.copy(gout[k], gacc[k])
            mark("geom_bwd")
            if args.fused_adam:
                adam_fused(it + 1)                         # reads gout directly (no gout->gacc copies)
            else:
                adam_inplace(it + 1)
            if args.depth_weight:
                adam_bt(it + 1)
            if args.arm_softmin:
                adam_sm(it + 1)
                # refresh zref_buf every ~500 iters (zref = median visible mcz, re-read after sync)
                if it > 0 and it % 500 == 0:
                    ttnn.synchronize_device(DEV)
                    zc = dn(cache["mcz"]).reshape(-1); kc = dn(keep).reshape(-1) > 0.5
                    zv = zc[kc] if bool(kc.any()) else zc
                    setbuf(zref_buf, torch.full((G,), float(zv.median())), DT)
            if args.arm_pairwise:
                adam_pw(it + 1)
            mark("adam")
            if args.relocate_every and it > 0 and it % args.relocate_every == 0:
                if PROF:
                    _t_reloc = time.perf_counter()
                    n_dead = relocate_host()
                    print(f"   [relocate] it={it} dead={n_dead}/{G} took {time.perf_counter()-_t_reloc:.2f}s", flush=True)
                else:
                    relocate_host()
            if it % max(1, args.iters // 10) == 0:
                ttnn.synchronize_device(DEV)
                bt_s = f" | beta {bt['beta']:.3f} tau {bt['tau']:.3f}" if args.depth_weight else ""
                sm_s = f" | sm_tau {sm['tau']:.3f}" if args.arm_softmin else ""
                pw_s = f" | pw_tau {pw['tau']:.4f}" if args.arm_pairwise else ""
                print(f"   iter {it:5d}/{args.iters}  {(it + 1) / (time.perf_counter() - t0):.2f} it/s{bt_s}{sm_s}{pw_s}", flush=True)
        ttnn.synchronize_device(DEV)
        if PROF and _pn[0] > 0:
            _n = _pn[0]; _tot = sum(_pacc.values())
            print(f"\n=== PER-ITER PROFILE (n={_n}, bin_every={args.bin_every}, G={G}, res={res}) ===", flush=True)
            for _k in sorted(_pacc, key=lambda x: -_pacc[x]):
                print(f"  {_k:10s} {_pacc[_k]/_n*1000:8.2f} ms/iter  ({100*_pacc[_k]/_tot:5.1f}%)", flush=True)
            print(f"  {'TOTAL':10s} {_tot/_n*1000:8.2f} ms/iter  (profiled {_n/_tot:.2f} it/s)", flush=True)
        train_s = time.perf_counter() - t0
        it_s = round(args.iters / train_s, 2)

        ho_p, ho_s = eval_split(te_c, te_i)
        tr_p, tr_s = eval_split(tr_c[:10], tr_i[:10])
        print(f"[fast] {args.scene_name} G={G}: {it_s} it/s | holdout {ho_p:.2f}dB/{ho_s:.4f} | "
              f"train {tr_p:.2f}dB | {train_s/60:.1f} min", flush=True)

        # Diagnostic black-bg render (shows empty-space fill); kept as-is
        rd = os.path.join(out_dir, "renders"); os.makedirs(rd, exist_ok=True)
        setbuf(bias, torch.zeros(T, 256, 3), BF)
        run_geom_bin(te_c, 0)
        ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        imgb = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3)).reshape(H, W, 3)
        _save_img(os.path.join(rd, "test_00_black.png"), imgb)

        # pull params back into the model for ply / evalcard
        with torch.no_grad():
            m.means3d.copy_(torch.stack([dn(P["mx"]).reshape(-1), dn(P["my"]).reshape(-1), dn(P["mz"]).reshape(-1)], -1))
            m.quats.copy_(torch.stack([dn(P["qw"]).reshape(-1), dn(P["qx"]).reshape(-1), dn(P["qy"]).reshape(-1), dn(P["qz"]).reshape(-1)], -1))
            m.log_scales.copy_(torch.stack([dn(P["lx"]).reshape(-1), dn(P["ly"]).reshape(-1), dn(P["lz"]).reshape(-1)], -1))
            m.color_dc.copy_(torch.stack([dn(P["cr"]).reshape(-1), dn(P["cg"]).reshape(-1), dn(P["cb"]).reshape(-1)], -1))
            m.opacity_raw.copy_(dn(P["op"]).reshape(-1))
        ply_path = ""
        if not args.no_ply:
            ply_path = paths["ply"]; plyio.save_ply(ply_path, m)
        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            f.write(f"scene {args.scene_name} G {G} K {K} res {res} iters {args.iters}\n")
            f.write(f"it_per_s {it_s}\nholdout_psnr {ho_p:.2f}\nholdout_ssim {ho_s:.4f}\ntrain_psnr {tr_p:.2f}\n")
            f.write(f"train_min {train_s/60:.1f}\nply {ply_path}\ndevice blackhole-fast-resident-deviceloss\n")
        print(f"   saved -> {out_dir}/", flush=True)

        # ===== (E) canonical outputs: eval.json, panels, rollup.json, summary.txt =====

        # Eval card (blender only — COLMAP has no per-pixel alpha for the two-bg trick)
        perc = {}
        if not args.colmap:
            if PROF:
                print("[diag] loading test rgba...", flush=True)
                _t_lb = time.perf_counter()
            _, te_rgba = data.load_blender(args.scene, "test", res=res, keep_alpha=True)
            if PROF:
                print(f"[diag] loaded {len(te_rgba)} test rgba in {time.perf_counter()-_t_lb:.2f}s", flush=True)

            _rfn_calls = [0]

            def rfn_eval(mdl, cam, b):
                """Device render with constant bg b for evalcard two-bg perceptual metrics."""
                if PROF:
                    _rfn_calls[0] += 1
                    if _rfn_calls[0] % 10 == 1:
                        print(f"[diag] rfn_eval call #{_rfn_calls[0]}", flush=True)
                setbuf(bias, torch.full((T, 256, 3), w_b * float(b)), BF)
                _setcam_direct(cam)
                ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
                if PROF and _rfn_calls[0] % 10 == 1:
                    print(f"[diag]   gfid done", flush=True)
                idx_e, valid_e = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
                if PROF and _rfn_calls[0] % 10 == 1:
                    print(f"[diag]   assign_bins done", flush=True)
                setbuf(idx_u, idx_e.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                setbuf(valid6, valid_e[:, None, :].expand(T, 6, K).float())
                setbuf(vf, valid_e[..., None].float())
                ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
                if PROF and _rfn_calls[0] % 10 == 1:
                    print(f"[diag]   rfid done", flush=True)
                return torch.zeros(H * W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3)).reshape(H, W, 3)

            try:
                card = evalcard.build(m, te_c, te_rgba, rfn_eval, method=args.method,
                                      scene=args.scene_name, G=G, res=res, iters=args.iters, K=K,
                                      train_psnr=tr_p,
                                      perf={"it_per_s": it_s, "render_fps": None,
                                            "train_s": round(train_s, 1), "peak_mem_gb": None,
                                            "device": "blackhole-fast-resident-deviceloss"})
                card["holdout"] = {"psnr": round(ho_p, 2), "ssim": round(ho_s, 4)}
                evalcard.save(card, paths["eval_json"])
                perc = card.get("perceptual", {})
                print(f"[fast]   eval card -> {paths['eval_json']}", flush=True)
            except Exception as e:
                print(f"[fast]   WARN eval card skipped: {e}", flush=True)

        # Representative-view panels (4 held-out views, full run only)
        renders_dir = None
        if not args.no_ply and not args.no_views:
            renders_dir = paths["stem"] + "_renders"
            os.makedirs(renders_dir, exist_ok=True)
            # Reset bias to white bg before panel renders
            setbuf(bias, torch.full((T, 256, 3), w_b), BF)
            for i in range(min(4, len(te_c))):
                run_geom_bin(te_c, i)
                ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
                img = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3)).reshape(H, W, 3)
                _save_panel(os.path.join(renders_dir, f"view{i:02d}_panel.png"), te_i[i], img)
            print(f"[fast]   panels -> {renders_dir}", flush=True)

        # Canonical row (all required keys, in contract order)
        row = {
            "method": args.method,
            "scene": args.scene_name, "G": G, "res": res, "K": K, "iters": args.iters,
            "train_views": Ntr,
            "holdout_psnr": round(ho_p, 2), "holdout_ssim": round(ho_s, 4),
            "train_psnr": round(tr_p, 2), "train_ssim": round(tr_s, 4),
            "hf_ratio": perc.get("hf_ratio"), "empty_space_leak": perc.get("empty_space_leak"),
            "it_per_s": it_s, "render_fps": None, "train_s": round(train_s, 1),
            "peak_mem_gb": None, "device": "blackhole-fast-resident-deviceloss",
            "ply": ply_path, "eval_json": paths["eval_json"], "renders_dir": renders_dir,
        }

        # Incremental rollup (append to any existing rows from prior invocations)
        rollup_path = os.path.join(args.out, "bh_fast_rollup.json")
        existing_rows = []
        if os.path.exists(rollup_path):
            try:
                with open(rollup_path) as f:
                    existing_rows = json.load(f)
            except Exception:
                pass
        existing_rows.append(row)
        json.dump(existing_rows, open(rollup_path, "w"), indent=2)

        # Summary (human-readable, regenerated from full rollup)
        summary_path = os.path.join(args.out, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"# {args.method} | res{res}/{args.iters}it\n")
            f.write(f"# {'scene':8} {'G':>7} {'K':>4} {'it/s':>7} {'fps':>6} "
                    f"{'ho_psnr':>8} {'ho_ssim':>8} {'hf':>8} {'leak':>8}\n")
            for r in existing_rows:
                k_str = "-" if r.get("K") is None else str(r["K"])
                hf = f"{r['hf_ratio']:.4f}" if r.get("hf_ratio") is not None else "N/A"
                leak = f"{r['empty_space_leak']:.4f}" if r.get("empty_space_leak") is not None else "N/A"
                fps_str = f"{r['render_fps']:.0f}" if r.get("render_fps") is not None else "N/A"
                f.write(f"  {r['scene']:8} {r['G']:>7} {k_str:>4} {r['it_per_s']:>7} "
                        f"{fps_str:>6} {r['holdout_psnr']:>8} {r['holdout_ssim']:>8} "
                        f"{hf:>8} {leak:>8}\n")
        print(f"[fast]   rollup -> {rollup_path}", flush=True)
        print("\n" + open(summary_path).read())

    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
