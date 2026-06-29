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
from resident_traced import render_bwd, theta_bwd, PN, B1, B2, EPS
from loss_manual import gauss_1d, band_matrix, C1 as L_C1, C2 as L_C2, LAMBDA as L_LAM

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
    # depth-weight lever. off -> reproduces arm-A exactly.
    ap.add_argument("--depth-weight", action="store_true")
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--tau0", type=float, default=4.0)          # refined from z readback at warmup
    ap.add_argument("--lr-bt", type=float, default=0.02)        # Adam LR for beta/tau scalars
    # random-background training (gsplat-style honest-opacity; WSR bakes bg into the normalized avg
    # otherwise -> grey/white empty-space gaussians). per-iter random bg, c_B=bg; eval stays white.
    ap.add_argument("--random-bg", action="store_true")
    ap.add_argument("--colmap", default=None, help="path to a COLMAP scene root (real scene); overrides blender")
    ap.add_argument("--downscale", type=int, default=4, help="COLMAP image downscale factor")
    args = ap.parse_args()
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
        # depth-weight lever: rho=sigmoid(beta*(tau-z)), sort-free, multiplies o.
        # beta/tau are global learnable scalars (broadcast to [G,1] device buffers, host-Adam updated per-iter).
        beta_buf = u(torch.full((G,), args.beta0)); tau_buf = u(torch.full((G,), args.tau0))
        gbeta_buf = u(torch.zeros(G)); gtau_buf = u(torch.zeros(G))
        bt = {"beta": args.beta0, "tau": args.tau0}
        bt_m = {"beta": 0.0, "tau": 0.0}; bt_v = {"beta": 0.0, "tau": 0.0}
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

        def rend_fwd():
            theta, col_t, oc_t = gather_theta_buf(conic, mu2d, color_o, keo, idx_u, valid6, vf, origins_t, bump_buf, T, K)
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
            gthU, gcol, goc = render_bwd(Phi, col_t, oc_t, gC_buf, rc)
            gct, gmt = theta_bwd(gthU, conic_t, mu_t, origins_t, valid6, T, K)
            return gct, gmt, gcol, goc

        gct, gmt, gcol, goc = rend_bwd()

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
            else:
                out["op"] = M(M(gkeo, keep), o1mo)
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
            for k in PN:                                       # reset Adam moments for touched slots
                mk = dn(mom[k]).reshape(-1).clone(); vk = dn(vom[k]).reshape(-1).clone()
                mk[touched] = 0.0; vk[touched] = 0.0
                setbuf(mom[k], mk, DT); setbuf(vom[k], vk, DT)
            return int(dead.numel())

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
        idx, valid = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
        setbuf(idx_u, idx.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        setbuf(valid6, valid[:, None, :].expand(T, 6, K).float()); setbuf(vf, valid[..., None].float())
        C, rc, theta, col_t, oc_t = rend_fwd()
        loss_grad_dev(); gct, gmt, gcol, goc = rend_bwd(); gout = geom_bwd()
        ttnn.synchronize_device(DEV)

        # ===== (C) capture traces =====
        def cap(fn):
            tid = ttnn.begin_trace_capture(DEV, cq_id=0); r = fn(); ttnn.end_trace_capture(DEV, tid, cq_id=0); return tid, r
        gfid, (conic, mu2d, cache, keep, keo, o, color, color_o) = cap(geom_fwd)
        ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        rfid, (C, rc, theta, col_t, oc_t) = cap(rend_fwd)
        lfid, _ = cap(loss_grad_dev)
        rbid, (gct, gmt, gcol, goc) = cap(rend_bwd)
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
        for it in range(args.iters):
            v = int(torch.randint(Ntr, (1,)).item())
            setcam(tr_c, v)
            if args.random_bg:                                 # composite GT over a random bg; c_B=bg
                bg = torch.rand(3)
                setbuf(y_d, tr_pm[v] + tr_tr[v] * bg[:, None, None], DT)
                setbuf(bias, (w_b * bg)[None, None, :].expand(T, 256, 3).contiguous(), BF)
            else:
                ttnn.copy(gt_res[v], y_d)
            ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False)
            if it % args.bin_every == 0:
                ttnn.synchronize_device(DEV)
                idx, valid = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
                setbuf(idx_u, idx.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                setbuf(valid6, valid[:, None, :].expand(T, 6, K).float()); setbuf(vf, valid[..., None].float())
            ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False)
            ttnn.execute_trace(DEV, lfid, cq_id=0, blocking=False)
            ttnn.execute_trace(DEV, rbid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
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
            ttnn.execute_trace(DEV, gbid, cq_id=0, blocking=False)
            for k in PN:
                ttnn.copy(gout[k], gacc[k])
            adam_inplace(it + 1)
            if args.depth_weight:
                adam_bt(it + 1)
            if args.relocate_every and it > 0 and it % args.relocate_every == 0:
                relocate_host()
            if it % max(1, args.iters // 10) == 0:
                ttnn.synchronize_device(DEV)
                bt_s = f" | beta {bt['beta']:.3f} tau {bt['tau']:.3f}" if args.depth_weight else ""
                print(f"   iter {it:5d}/{args.iters}  {(it + 1) / (time.perf_counter() - t0):.2f} it/s{bt_s}", flush=True)
        ttnn.synchronize_device(DEV)
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
            _, te_rgba = data.load_blender(args.scene, "test", res=res, keep_alpha=True)

            def rfn_eval(mdl, cam, b):
                """Device render with constant bg b for evalcard two-bg perceptual metrics."""
                setbuf(bias, torch.full((T, 256, 3), w_b * float(b)), BF)
                _setcam_direct(cam)
                ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
                idx_e, valid_e = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
                setbuf(idx_u, idx_e.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                setbuf(valid6, valid_e[:, None, :].expand(T, 6, K).float())
                setbuf(vf, valid_e[..., None].float())
                ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
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
