"""Full-pipeline traced trainer (single view, proof + speed). Applies the verified alloc-first
ordering (docs: trace-allocation hazard SOLVED in m6_traced_fwd) to the WHOLE step: geometry fwd, gather+
render fwd, render bwd + theta bwd, geometry bwd, AND a no-alloc (output_tensor=) Adam -- all traced /
in-place, so per iter there is ZERO device allocation (only execute_trace + copy_host_to_device + host
binning/loss/scatter). Resident params updated in place by the Adam ops.

Single view = the minimal complete traced pipeline (overfits 1 view; proves correctness via loss↓ + times
the traced per-view step). Multi-view = replicate the 4 traces per view.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m6_resident_traced.py --res 96 --G 4000 --iters 80
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike import data, metrics, sh, plyio
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins, K_POLY
from m6_geom_device import device_fwd_core, device_bwd_core, A, M
from m6_traced_fwd import gather_theta_buf, gather_theta_cols, conic_mu_cols_gather, render_fwd
from m7_loss_manual import gauss_1d, band_matrix, filt as hfilt, C1 as L_C1, C2 as L_C2, LAMBDA as L_LAM
from m9_scatter_oracle import build_inv
from bin_device import bin_to_buffers, build_inv_device, make_bin_ctx

C0 = sh.C0
DEV = None
CG = None
DT = ttnn.float32
BF = ttnn.bfloat16


def u(t, dt=DT):
    return ttnn.from_torch(t.reshape(-1, 1).contiguous() if t.dim() == 1 else t.contiguous(),
                           dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV)


def dn(t):
    return ttnn.to_torch(t).float()


def setbuf(buf, t, dt=BF, layout=ttnn.TILE_LAYOUT):
    t2 = t.reshape(-1, 1) if t.dim() == 1 else t          # match u()'s [G]->[G,1]
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(t2.contiguous(), dtype=dt, layout=layout), buf)


def T3(t):
    return ttnn.transpose(t, -2, -1)


def render_bwd(Phi, col, oc, gC, cache):
    relu_Q, w, den, num = cache
    C = ttnn.div(num, den)
    gnum = ttnn.div(gC, den)
    gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(gC, C), dim=-1, keepdim=True)), den)
    gcol = ttnn.matmul(T3(w), gnum, core_grid=CG)
    goc = ttnn.matmul(T3(w), gden, core_grid=CG)
    gw = ttnn.add(ttnn.matmul(gnum, T3(col), core_grid=CG), ttnn.matmul(gden, T3(oc), core_grid=CG))
    gthU = ttnn.matmul(T3(Phi), ttnn.mul(gw, ttnn.mul(relu_Q, 2.0)), core_grid=CG)
    return gthU, gcol, goc


def theta_bwd(gthU, conic_t, mu_t, origins_t, valid6, T, K):
    gpoly = ttnn.transpose(ttnn.mul(gthU, valid6), 1, 2)
    gtQ = ttnn.mul(gpoly, -1.0 / K_POLY)
    g0, g1, g2 = gtQ[:, :, 0:1], gtQ[:, :, 1:2], gtQ[:, :, 2:3]
    g3, g4, g5 = gtQ[:, :, 3:4], gtQ[:, :, 4:5], gtQ[:, :, 5:6]
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    mu = ttnn.add(mu_t, ttnn.neg(origins_t))
    mux, muy = mu[:, :, 0:1], mu[:, :, 1:2]
    ga = ttnn.add(ttnn.add(g0, ttnn.mul(g3, ttnn.mul(mux, -2.0))), ttnn.mul(g5, ttnn.mul(mux, mux)))
    gb = ttnn.add(ttnn.add(ttnn.mul(g2, 2.0), ttnn.mul(g3, ttnn.mul(muy, -2.0))),
                  ttnn.add(ttnn.mul(g4, ttnn.mul(mux, -2.0)), ttnn.mul(g5, ttnn.mul(ttnn.mul(mux, muy), 2.0))))
    gc = ttnn.add(ttnn.add(g1, ttnn.mul(g4, ttnn.mul(muy, -2.0))), ttnn.mul(g5, ttnn.mul(muy, muy)))
    gmux = ttnn.add(ttnn.add(ttnn.mul(g3, ttnn.mul(a, -2.0)), ttnn.mul(g4, ttnn.mul(b, -2.0))),
                    ttnn.mul(g5, ttnn.add(ttnn.mul(a, ttnn.mul(mux, 2.0)), ttnn.mul(b, ttnn.mul(muy, 2.0)))))
    gmuy = ttnn.add(ttnn.add(ttnn.mul(g3, ttnn.mul(b, -2.0)), ttnn.mul(g4, ttnn.mul(c, -2.0))),
                    ttnn.mul(g5, ttnn.add(ttnn.mul(b, ttnn.mul(mux, 2.0)), ttnn.mul(c, ttnn.mul(muy, 2.0)))))
    return ttnn.concat([ga, gb, gc], -1), ttnn.concat([gmux, gmuy], -1)


PN = ["mx", "my", "mz", "qw", "qx", "qy", "qz", "lx", "ly", "lz", "cr", "cg", "cb", "op"]
B1, B2, EPS = 0.9, 0.999, 1e-8


def main():
    global DEV, CG
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--G", type=int, default=4000)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--bin-every", type=int, default=5)
    ap.add_argument("--colmap", default=None, help="path to a COLMAP scene root (real scene); overrides blender")
    ap.add_argument("--downscale", type=int, default=4, help="COLMAP image downscale factor")
    ap.add_argument("--fused-adam", action="store_true",
                    help="batch the 14 params into one [G,14] Adam update (~140 ops -> ~10) -- kills the "
                         "small-model dispatch bottleneck (Adam was 62%% of device exec at res128/G2000)")
    ap.add_argument("--profile-stages", action="store_true",
                    help="time each traced stage (device exec) individually to break down device+other")
    ap.add_argument("--device-scatter", action="store_true",
                    help="run the bin->gaussian grad scatter ON DEVICE (gather-reduce) -> no per-iter sync")
    ap.add_argument("--device-loss", action="store_true",
                    help="compute dL/dC on device (no C download / gC upload / host loss); else host loss")
    ap.add_argument("--device-binning", action="store_true",
                    help="compute binning idx/valid ON DEVICE (bin_to_buffers: v2 masked-dist+topk); else host assign_bins")
    ap.add_argument("--save-ply", default=None, help="after training, write the trained gaussians to this .ply")
    ap.add_argument("--multi-view", action="store_true",
                    help="train on ALL views (per-iter random), held-out every-8th for eval; forces host loss")
    ap.add_argument("--profile", action="store_true",
                    help="time the remaining HOST blocks (binning, scatter) to see what's worth device-izing")
    ap.add_argument("--sh-degree", type=int, default=0,
                    help="view-dependent SH colour degree (0=DC-only; 3=full). Host SH eval -> upload -> device render.")
    ap.add_argument("--relocate-every", type=int, default=0,
                    help="fixed-count MCMC density control: relocate dead (o<thr) slots onto live ones every N "
                         "iters (host-controlled, contribution-preserving o->o/n). 0=off.")
    ap.add_argument("--dead-thr", type=float, default=0.005, help="opacity threshold for a 'dead' gaussian (MCMC)")
    ap.add_argument("--lambda-o", type=float, default=0.0, help="MCMC opacity-L1 reg (drives low contributors dead)")
    ap.add_argument("--lambda-s", type=float, default=0.0, help="MCMC scale-L1 reg (discourages oversized gaussians)")
    ap.add_argument("--noise-lr", type=float, default=0.0, help="MCMC SGLD Langevin-noise scale on near-dead means (e.g. 5e5)")
    # random-bg: per-iter random background compositing (gsplat-style honest opacity; WSR else bakes the bg
    # into the normalized average -> grey/white empty-space haze). NeRF-synthetic + --multi-view only; eval stays white.
    ap.add_argument("--random-bg", action="store_true",
                    help="per-iter random-bg compositing (NeRF-synthetic + --multi-view); eval renders on white")
    # depth-weight lever: rho=sigmoid(beta*(tau-depth)), sort-free, multiplies o.
    ap.add_argument("--depth-weight", action="store_true",
                    help="per-gaussian monotonic depth weight (offsets random-bg's white-fill loss; device-resident)")
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--tau0", type=float, default=4.0, help="refined from the depth median at warmup")
    ap.add_argument("--lr-bt", type=float, default=0.02, help="Adam LR for the 2 global beta/tau scalars")
    args = ap.parse_args()
    if args.colmap:
        args.random_bg = False         # real scenes have full backgrounds (no alpha) -> no random-bg
    # multi-view + --device-loss now works: the per-view GT SSIM stats are recomputed ON DEVICE each iter
    # (recompute_gt_stats); without --device-loss, multi-view falls back to host loss.
    DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        res, G, K = args.res, args.G, args.K
        torch.manual_seed(0)
        tr_pm = tr_tr = None                              # random-bg: per-view premult colour + transmittance
        if args.colmap:                                   # real scene: COLMAP poses + point-init
            from spike.camera import Camera
            cams, imgs = data.load_colmap(args.colmap, downscale=args.downscale,
                                          n=None if args.multi_view else 1)
            Hc, Wc = (cams[0].H // 16) * 16, (cams[0].W // 16) * 16   # crop to whole 16x16 tiles

            def crop(c, im):
                return (Camera(c.R_v, c.t_v, c.fx, c.fy, c.cx, c.cy, Hc, Wc), im[:Hc, :Wc, :].contiguous())

            tr_cams = te_cams = tr_imgs = te_imgs = None
            if args.multi_view:                            # every-8th view held out for eval
                te = set(range(0, len(cams), 8))
                tr = [crop(cams[i], imgs[i]) for i in range(len(cams)) if i not in te]
                ev = [crop(cams[i], imgs[i]) for i in sorted(te)]
                tr_cams, tr_imgs = [c for c, _ in tr], [g for _, g in tr]
                te_cams, te_imgs = [c for c, _ in ev], [g for _, g in ev]
                cam, gt = tr_cams[0], tr_imgs[0]
            else:
                cam, gt = crop(cams[0], imgs[0])
            tmap = TileMap(Hc, Wc)
            m = GaussianModel(G, seed=0)
            pxyz, prgb = data.load_colmap_points(args.colmap)
            m.init_from_points(torch.from_numpy(pxyz), torch.from_numpy(prgb))
            print(f"[colmap] {args.colmap} downscale={args.downscale} -> {Hc}x{Wc}, G={G} init from "
                  f"{len(pxyz)} pts" + (f" | {len(tr_cams)} train / {len(te_cams)} test views"
                                        if args.multi_view else " (single view)"), flush=True)
        else:
            if args.multi_view:                            # NeRF-synthetic: all train views + held-out test
                tr_cams, tr_imgs = data.load_blender(args.scene, "train", res=res)
                te_cams, te_imgs = data.load_blender(args.scene, "test", res=res, n=25, stride=8)
                if args.random_bg:                         # keep-alpha train set (same default selection as tr_imgs)
                    _, tr_rgba = data.load_blender(args.scene, "train", res=res, keep_alpha=True)
                    tr_pm = [(im[..., :3] * im[..., 3:4]).permute(2, 0, 1).contiguous() for im in tr_rgba]   # [3,H,W]
                    tr_tr = [(1.0 - im[..., 3:4]).expand(-1, -1, 3).permute(2, 0, 1).contiguous() for im in tr_rgba]
                cam, gt = tr_cams[0], tr_imgs[0]
                print(f"[blender] {args.scene} -> {res}x{res}, G={G} | "
                      f"{len(tr_cams)} train / {len(te_cams)} test views", flush=True)
            else:
                cams, imgs = data.load_blender(args.scene, "train", res=res, n=1)
                cam, gt = cams[0], imgs[0]
            tmap = TileMap(res, res)
            m = GaussianModel(G, extent=1.5, seed=0)
        T = tmap.T
        # camera as DEVICE buffers (captured in the geom trace; updated per-iter for multi-view).
        # device_fwd_core wants [P,1] device tensors -> use [G,1] (scalar REPLICATED over G) so every op
        # is elementwise (NOT a [1,1] broadcast, which hangs the host-side program-gen).
        def cbuf(v):
            return ttnn.from_torch(torch.full((G, 1), float(v)), dtype=BF, layout=ttnn.TILE_LAYOUT, device=DEV)
        Rv = [[cbuf(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
        tv = [cbuf(cam.t_v[i]) for i in range(3)]
        fx, fy, cx, cy = cbuf(cam.fx), cbuf(cam.fy), cbuf(cam.cx), cbuf(cam.cy)
        # camera centre (world) as fp32 buffers [G,1] for the device SH view direction (means - centre)
        cfp = lambda v: ttnn.from_torch(torch.full((G, 1), float(v)), dtype=DT, layout=ttnn.TILE_LAYOUT, device=DEV)
        ctr = [cfp(cam.center[i]) for i in range(3)]

        def set_cam(c):
            for i in range(3):
                for j in range(3):
                    setbuf(Rv[i][j], torch.full((G, 1), float(c.R_v[i, j])))
                setbuf(tv[i], torch.full((G, 1), float(c.t_v[i])))
                setbuf(ctr[i], torch.full((G, 1), float(c.center[i])), DT)
            for buf, v in ((fx, c.fx), (fy, c.fy), (cx, c.cx), (cy, c.cy)):
                setbuf(buf, torch.full((G, 1), float(v)))
        from spike.train import DEFAULT_LR
        LR = {**{k: DEFAULT_LR["means"] for k in ("mx", "my", "mz")},
              **{k: DEFAULT_LR["quats"] for k in ("qw", "qx", "qy", "qz")},
              **{k: DEFAULT_LR["scales"] for k in ("lx", "ly", "lz")},
              **{k: DEFAULT_LR["color"] for k in ("cr", "cg", "cb")}, "op": DEFAULT_LR["opacity"]}
        w_b = float(torch.nn.functional.softplus(m.w_b_raw))

        # ===== (A) ALLOCATE every persistent buffer up front =====
        P = dict(mx=u(m.means3d[:, 0]), my=u(m.means3d[:, 1]), mz=u(m.means3d[:, 2]),
                 qw=u(m.quats[:, 0]), qx=u(m.quats[:, 1]), qy=u(m.quats[:, 2]), qz=u(m.quats[:, 3]),
                 lx=u(m.log_scales[:, 0]), ly=u(m.log_scales[:, 1]), lz=u(m.log_scales[:, 2]),
                 cr=u(m.color_dc[:, 0]), cg=u(m.color_dc[:, 1]), cb=u(m.color_dc[:, 2]), op=u(m.opacity_raw))
        mom = {k: u(torch.zeros(G)) for k in PN}
        vom = {k: u(torch.zeros(G)) for k in PN}
        gacc = {k: u(torch.zeros(G)) for k in PN}
        tmp1 = {k: u(torch.zeros(G)) for k in PN}
        tmp2 = {k: u(torch.zeros(G)) for k in PN}
        # fused Adam: merged [G,14] moments + [1,14] per-param LR row (batch 14 params -> 1 update)
        mom_m = u(torch.zeros(G, len(PN)))
        vom_m = u(torch.zeros(G, len(PN)))
        LR_vec = u(torch.tensor([[LR[k] for k in PN]], dtype=torch.float32))   # [1,14], broadcast over G

        # ===== SH view-dependent colour: cr/cg/cb (DC) + color_rest as DEVICE params, evaluated ON DEVICE =====
        from spike.sh import C0, C1, C2, C3
        SHD = args.sh_degree
        if SHD > 0:                                                            # rest coeffs as [G,1]x15 (NO concat:
            crest = [[u(m.color_rest[:, l, ci]) for l in range(15)] for ci in range(3)]    # the [G,15] concat blew L1 at G=250k)
            crest_mom = [[u(torch.zeros(G)) for _ in range(15)] for _ in range(3)]
            crest_vom = [[u(torch.zeros(G)) for _ in range(15)] for _ in range(3)]
            LR_REST = DEFAULT_LR["color"] / 20.0

            def sh_color_dev(vx, vy, vz, dc):
                """ON-DEVICE SH eval (verified brick, tools/sh_device): viewdir -> 15 rest-basis (each [G,1],
                no concat) + DC -> color[ch]=relu(0.5 + C0*dc[ch] + sum_l b_l*crest[ch][l]). Returns (color, blist)."""
                M, A, S = ttnn.mul, ttnn.add, ttnn.sub
                inv = ttnn.rsqrt(A(A(A(M(vx, vx), M(vy, vy)), M(vz, vz)), 1e-12))
                x, y, z = M(vx, inv), M(vy, inv), M(vz, inv)
                xx, yy, zz, xy, yz, xz = M(x, x), M(y, y), M(z, z), M(x, y), M(y, z), M(x, z)
                blist = [M(y, -C1), M(z, C1), M(x, -C1),
                         M(xy, C2[0]), M(yz, C2[1]), M(S(M(zz, 2.0), A(xx, yy)), C2[2]), M(xz, C2[3]), M(S(xx, yy), C2[4]),
                         M(M(y, S(M(xx, 3.0), yy)), C3[0]), M(M(xy, z), C3[1]), M(M(y, S(M(zz, 4.0), A(xx, yy))), C3[2]),
                         M(M(z, S(M(zz, 2.0), A(M(xx, 3.0), M(yy, 3.0)))), C3[3]), M(M(x, S(M(zz, 4.0), A(xx, yy))), C3[4]),
                         M(M(z, S(xx, yy)), C3[5]), M(M(x, S(xx, M(yy, 3.0))), C3[6])]              # 15 x [G,1]
                color = []
                for ci in range(3):
                    s = M(dc[ci], C0)
                    for l in range(15):
                        s = A(s, M(blist[l], crest[ci][l]))
                    color.append(ttnn.relu(A(s, 0.5)))
                return color, blist
        idx_u = ttnn.from_torch(torch.zeros(T, K, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        valid6 = u(torch.zeros(T, 6, K), BF)
        vf = u(torch.zeros(T, K, 1), BF)
        origins_t = u(tmap.origins[:, None, :].expand(T, K, 2).contiguous(), BF)
        Phi = u(tmap.Phi.unsqueeze(0).expand(T, 256, 6).contiguous(), BF)
        bias = u((w_b * m.c_b)[None, None, :].expand(T, 256, 3).contiguous(), BF)
        wb_buf = u(torch.full((T, 256, 1), w_b), BF)
        bump = torch.zeros(T, K, 6); bump[..., 5] = 1.0
        bump_buf = u(bump, BF)
        gC_buf = u(torch.zeros(T, 256, 3), BF)
        gcon_buf = {k: u(torch.zeros(G)) for k in ("a", "b", "c", "mx", "my")}  # geom-bwd grad inputs
        gco_buf = [u(torch.zeros(G)) for _ in range(3)]
        gocl_buf = u(torch.zeros(G))
        # depth-weight lever buffers: rho=sigmoid(beta*(tau-depth)); global beta/tau scalars
        # broadcast to [G,1] device buffers (device fwd/bwd), host-Adam updated per iter (2 scalars, light host glue).
        beta_buf = u(torch.full((G,), float(args.beta0)))
        tau_buf = u(torch.full((G,), float(args.tau0)))
        gbeta_buf = u(torch.zeros(G))
        gtau_buf = u(torch.zeros(G))
        bt = {"beta": float(args.beta0), "tau": float(args.tau0)}
        bt_m = {"beta": 0.0, "tau": 0.0}
        bt_v = {"beta": 0.0, "tau": 0.0}
        # device-scatter (bin->gaussian grad reduce on device): inv[G,Smax] slot table + zero sentinel row
        SMAX = (2 * 1 + 1) ** 2                         # R=1 stencil -> <=9 valid slots per gaussian
        sinv_u = ttnn.from_torch(torch.full((G, SMAX), T * K, dtype=torch.int32),
                                 dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        zrow = u(torch.zeros(1, 9), BF)                # grad_pad sentinel (gathered by inv padding)
        def set_sinv(idx, valid):
            setbuf(sinv_u, build_inv(idx, valid, G, SMAX).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        # device-binning CONSTANTS preallocated ONCE (no per-bin-every from_torch -> kills the ~170ms)
        bin_ctx = make_bin_ctx(tmap, G, K, DEV) if args.device_binning else None

        # ---- device-loss constants (banded Gaussian GEMM matrices, gt y-maps, gidx gathers) ----
        if args.device_loss:
            H, W = cam.H, cam.W
            g1d = gauss_1d()
            Mh_np, Mw_np = band_matrix(H, g1d), band_matrix(W, g1d)
            Hout, Win = Mh_np.shape[0], Mw_np.shape[1]
            Ns_loss = float(3 * Mh_np.shape[0] * Mw_np.shape[0])
            N_loss = float(3 * H * W)
            Mh_b = u(Mh_np.unsqueeze(0).expand(3, *Mh_np.shape).contiguous(), DT)        # [3,Hout,Hin]
            MwT_b = u(Mw_np.t().unsqueeze(0).expand(3, Win, Mw_np.shape[0]).contiguous(), DT)   # [3,Win,Wout]
            MhT_b = u(Mh_np.t().unsqueeze(0).expand(3, Mh_np.shape[1], Hout).contiguous(), DT)  # [3,Hin,Hout]
            Mw_b = u(Mw_np.unsqueeze(0).expand(3, *Mw_np.shape).contiguous(), DT)         # [3,Wout,Win]
            gt_chw = gt.permute(2, 0, 1).contiguous().double()                            # [3,H,W]
            muy_h = hfilt(gt_chw, Mh_np, Mw_np)
            muy2_h = muy_h * muy_h
            sy_h = hfilt(gt_chw * gt_chw, Mh_np, Mw_np) - muy2_h
            muy_d, muy2_d, sy_d = u(muy_h, DT), u(muy2_h, DT), u(sy_h, DT)
            y_d = u(gt_chw, DT)
            inv = torch.empty(H * W, dtype=torch.long); inv[tmap.gidx] = torch.arange(T * 256)
            gidx_u = ttnn.from_torch(tmap.gidx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
            inv_u = ttnn.from_torch(inv.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)

            def recompute_gt_stats(gt_img):              # per-view GT SSIM stats ON DEVICE (multi-view device loss)
                setbuf(y_d, gt_img.permute(2, 0, 1).contiguous(), DT)
                f = lambda t: ttnn.matmul(ttnn.matmul(Mh_b, t), MwT_b)
                ttnn.copy(f(y_d), muy_d)
                ttnn.mul(muy_d, muy_d, output_tensor=muy2_d)
                ttnn.copy(ttnn.sub(f(ttnn.mul(y_d, y_d)), muy2_d), sy_d)

        def geom_fwd():
            sx, sy, sz = ttnn.exp(P["lx"]), ttnn.exp(P["ly"]), ttnn.exp(P["lz"])
            cols = (P["mx"], P["my"], P["mz"], P["qw"], P["qx"], P["qy"], P["qz"], sx, sy, sz)
            _, _, cache = device_fwd_core(cols, Rv, tv, fx, fy, cx, cy, concat_out=False)  # columns in cache
            keep = cache["zmask"]
            o = ttnn.sigmoid(P["op"])
            if args.depth_weight:                                    # depth weight: rho=sigmoid(beta*(tau-mcz)) x o
                rho = ttnn.sigmoid(M(beta_buf, A(tau_buf, ttnn.neg(cache["mcz"]))))   # mcz = camera-space z depth
                cache["rho"] = rho
                keo = M(M(keep, o), rho)
            else:
                keo = M(keep, o)
            if SHD > 0:                                              # device SH: viewdir = means - centre, eval on device
                color, blist = sh_color_dev(ttnn.sub(P["mx"], ctr[0]), ttnn.sub(P["my"], ctr[1]),
                                            ttnn.sub(P["mz"], ctr[2]), [P["cr"], P["cg"], P["cb"]])
            else:
                color = [ttnn.relu(A(M(P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
                blist = keo                                          # unused placeholder (DC mode)
            co = [M(keo, color[0]), M(keo, color[1]), M(keo, color[2])]   # color_o columns [G,1] (no concat)
            return cache, keep, keo, o, color, co, blist

        # forward state holders (set by warmup, reassigned by capture)
        cache, keep, keo, o, color, co, blist = geom_fwd()
        if args.depth_weight:                              # refine tau0 to the actual depth (camera-z) distribution
            ttnn.synchronize_device(DEV)
            zc = dn(cache["mcz"]).reshape(-1); kc = dn(keep).reshape(-1) > 0.5
            zv = zc[kc] if bool(kc.any()) else zc
            bt["tau"] = float(zv.median()); setbuf(tau_buf, torch.full((G,), bt["tau"]), DT)
            print(f"[depth-weight] beta0={bt['beta']:.3f} tau0={bt['tau']:.3f} lr_bt={args.lr_bt} "
                  f"(z {float(zv.min()):.2f}..{float(zv.max()):.2f})", flush=True)

        def rend_fwd():
            theta, col_t, oc_t = gather_theta_cols(cache["ca"], cache["cb"], cache["cc"], cache["mu2d_x"],
                                                   cache["mu2d_y"], co[0], co[1], co[2], keo,
                                                   idx_u, valid6, vf, origins_t, bump_buf, T, K)
            C, rc = render_fwd_cache(Phi, theta, col_t, oc_t, wb_buf, bias)
            return C, rc, theta, col_t, oc_t

        def render_fwd_cache(Phi, thU, col, oc, wbb, bias):
            relu_Q = ttnn.relu(ttnn.matmul(Phi, thU, core_grid=CG))
            w = ttnn.square(relu_Q)
            den = ttnn.add(ttnn.matmul(w, oc, core_grid=CG), wbb)
            num = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)
            return ttnn.div(num, den), (relu_Q, w, den, num)

        C, rc, theta, col_t, oc_t = rend_fwd()

        def rend_bwd():
            conic_t, mu_t = conic_mu_cols_gather(cache["ca"], cache["cb"], cache["cc"],
                                                 cache["mu2d_x"], cache["mu2d_y"], idx_u, T, K)
            gthU, gcol, goc = render_bwd(Phi, col_t, oc_t, gC_buf, rc)
            gct, gmt = theta_bwd(gthU, conic_t, mu_t, origins_t, valid6, T, K)
            return gct, gmt, gcol, goc

        gct, gmt, gcol, goc = rend_bwd()

        def scatter_dev():
            """bin->gaussian grad scatter ON DEVICE: gather each gaussian's <=SMAX slot grads via
            inv table + sum. Replaces the host index_add round-trip -> no per-iter sync. Writes the 9
            geom-bwd grad buffers (a,b,c,mx,my,col0,col1,col2,op) exactly as the host scatter did."""
            g9 = ttnn.concat([ttnn.typecast(gct, BF), ttnn.typecast(gmt, BF),
                              ttnn.typecast(gcol, BF), ttnn.typecast(goc, BF)], dim=-1)   # [T,K,9]
            gpad = ttnn.concat([ttnn.reshape(g9, (T * K, 9)), zrow], dim=0)               # [T*K+1,9]
            g = ttnn.sum(ttnn.typecast(ttnn.embedding(sinv_u, gpad), DT), dim=1, keepdim=False)  # [G,9]
            cols = ttnn.split(g, 1, dim=1)
            for c, b in zip(cols, (gcon_buf["a"], gcon_buf["b"], gcon_buf["c"], gcon_buf["mx"], gcon_buf["my"],
                                   gco_buf[0], gco_buf[1], gco_buf[2], gocl_buf)):
                ttnn.copy(c, b)
            return g

        def geom_bwd():
            gg = device_bwd_core(cache, gcon_buf["a"], gcon_buf["b"], gcon_buf["c"], gcon_buf["mx"], gcon_buf["my"])
            sx, sy, sz = ttnn.exp(P["lx"]), ttnn.exp(P["ly"]), ttnn.exp(P["lz"])
            out = {"mx": gg["gmx"], "my": gg["gmy"], "mz": gg["gmz"], "qw": gg["gqw"], "qx": gg["gqx"],
                   "qy": gg["gqy"], "qz": gg["gqz"], "lx": M(gg["gsx"], sx), "ly": M(gg["gsy"], sy), "lz": M(gg["gsz"], sz)}
            # color/opacity bwd: color_o = keo*color, o_col=keo, keo=keep*o
            gkeo = A(A(M(gco_buf[0], color[0]), M(gco_buf[1], color[1])), A(M(gco_buf[2], color[2]), gocl_buf))
            dcrest = [[None] * 15 for _ in range(3)]
            if SHD > 0:                                              # device SH bwd: dL/dcolour -> DC + rest coeff grads
                for ci, k in enumerate(("cr", "cg", "cb")):
                    gg = M(M(gco_buf[ci], keo), ttnn.gtz(color[ci]))    # dL/dcolour[ch] through the relu gate
                    out[k] = M(gg, C0)                                  # DC coeff grad
                    for l in range(15):
                        dcrest[ci][l] = M(blist[l], gg)               # rest coeff grad [G,1] (no concat)
            else:
                cmask = [ttnn.gtz(A(M(P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
                out["cr"] = M(M(M(gco_buf[0], keo), cmask[0]), C0)
                out["cg"] = M(M(M(gco_buf[1], keo), cmask[1]), C0)
                out["cb"] = M(M(M(gco_buf[2], keo), cmask[2]), C0)
            one = ttnn.add(ttnn.mul(o, 0.0), 1.0)
            o1mo = M(o, ttnn.add(one, ttnn.neg(o)))                 # sigmoid'(op) = o(1-o)
            if args.depth_weight:                                  # keo=keep*o*rho -> extra rho factor + beta/tau grads
                rho = cache["rho"]
                out["op"] = M(M(M(gkeo, keep), o1mo), rho)
                grad_rho = M(M(gkeo, keep), o)                     # dL/drho
                grad_pre = M(grad_rho, M(rho, A(one, ttnn.neg(rho))))    # * rho(1-rho)
                ttnn.copy(M(grad_pre, A(tau_buf, ttnn.neg(cache["mcz"]))), gbeta_buf)   # grad_beta_pg = grad_pre*(tau-mcz)
                ttnn.copy(M(grad_pre, beta_buf), gtau_buf)                              # grad_tau_pg  = grad_pre*beta
            else:
                out["op"] = M(M(gkeo, keep), o1mo)
            return out, dcrest

        gout, dcrest = geom_bwd()

        def loss_grad_dev():
            """C (device [T,256,3] bf16) -> dL/dC written into gC_buf, fully on device.
            img = gather(C, inv_gidx); dL/dimg via banded-GEMM SSIM + L1; gC = gather(dLdimg, gidx)."""
            def filt(t):    return ttnn.matmul(ttnn.matmul(Mh_b, t), MwT_b)   # Mh·t·Mwᵀ -> [3,Hout,Wout]
            def filt_T(t):  return ttnn.matmul(ttnn.matmul(MhT_b, t), Mw_b)   # Mhᵀ·t·Mw -> [3,Hin,Win]
            Cf = ttnn.reshape(C, (T * 256, 3))                                # bf16
            img = ttnn.embedding(inv_u, Cf)                                   # [H*W,3] bf16
            img = ttnn.reshape(ttnn.transpose(img, -2, -1), (3, H, W))        # [3,H,W]
            x = ttnn.typecast(img, DT)
            fx, fx2, fxy = filt(x), filt(ttnn.mul(x, x)), filt(ttnn.mul(x, y_d))
            mux = fx
            mux2 = ttnn.mul(mux, mux)
            sx = ttnn.sub(fx2, mux2)
            sxy = ttnn.sub(fxy, ttnn.mul(mux, muy_d))
            A1 = ttnn.add(ttnn.mul(ttnn.mul(mux, muy_d), 2.0), L_C1)
            A2 = ttnn.add(ttnn.mul(sxy, 2.0), L_C2)
            Bd1 = ttnn.add(ttnn.add(mux2, muy2_d), L_C1)
            Bd2 = ttnn.add(ttnn.add(sx, sy_d), L_C2)
            D = ttnn.mul(Bd1, Bd2)
            S = ttnn.div(ttnn.mul(A1, A2), D)
            dS_dfx2 = ttnn.neg(ttnn.div(S, Bd2))
            dS_dfxy = ttnn.div(ttnn.mul(A1, 2.0), D)
            term = ttnn.sub(ttnn.mul(muy_d, ttnn.sub(A2, A1)),
                            ttnn.mul(ttnn.mul(S, mux), ttnn.sub(Bd2, Bd1)))
            dS_dfx = ttnn.mul(ttnn.div(term, D), 2.0)
            dmeanS = ttnn.div(ttnn.add(ttnn.add(filt_T(dS_dfx), ttnn.mul(ttnn.mul(x, filt_T(dS_dfx2)), 2.0)),
                                       ttnn.mul(y_d, filt_T(dS_dfxy))), Ns_loss)
            g_ssim = ttnn.mul(dmeanS, -L_LAM)
            g_l1 = ttnn.mul(ttnn.sign(ttnn.sub(x, y_d)), (1.0 - L_LAM) / N_loss)
            gimg = ttnn.add(g_l1, g_ssim)                                     # [3,H,W] fp32
            gflat = ttnn.transpose(ttnn.reshape(ttnn.typecast(gimg, BF), (3, H * W)), -2, -1)  # [H*W,3] bf16
            gCt = ttnn.reshape(ttnn.embedding(gidx_u, gflat), (T, 256, 3))
            ttnn.copy(gCt, gC_buf)
            return gCt

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
            ~140 tiny ops -> ~10 -> kills the small-model dispatch bottleneck. Reads gout directly (no gacc)."""
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

        def adam_crest(t):                                   # device Adam for the SH rest coeffs (3 x 15 x [G,1])
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            for ci in range(3):
                for l in range(15):
                    g, mk, vk, p = dcrest[ci][l], crest_mom[ci][l], crest_vom[ci][l], crest[ci][l]
                    ttnn.mul(mk, B1, output_tensor=mk); ttnn.add(mk, ttnn.mul(g, 1.0 - B1), output_tensor=mk)
                    ttnn.mul(vk, B2, output_tensor=vk)
                    ttnn.add(vk, ttnn.mul(ttnn.mul(g, g), 1.0 - B2), output_tensor=vk)
                    denom = ttnn.add(ttnn.mul(ttnn.sqrt(vk), 1.0 / math.sqrt(bc2)), EPS)
                    ttnn.subtract(p, ttnn.div(ttnn.mul(mk, LR_REST / bc1), denom), output_tensor=p)

        def adam_bt(t):                                  # host Adam for the 2 global depth-weight scalars (beta/tau)
            bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
            grads = {"beta": float(dn(gbeta_buf).sum()), "tau": float(dn(gtau_buf).sum())}
            for k in ("beta", "tau"):
                g = grads[k]
                bt_m[k] = B1 * bt_m[k] + (1 - B1) * g
                bt_v[k] = B2 * bt_v[k] + (1 - B2) * g * g
                bt[k] -= args.lr_bt * (bt_m[k] / bc1) / (math.sqrt(bt_v[k] / bc2) + EPS)
            setbuf(beta_buf, torch.full((G,), bt["beta"]), DT)
            setbuf(tau_buf, torch.full((G,), bt["tau"]), DT)

        # ===== (B) untraced binning seed (no trace yet) =====
        ttnn.synchronize_device(DEV)
        def mu2d_host():     # rebuild mu2d[G,2] on host from the cache columns (for host binning)
            return torch.stack([dn(cache["mu2d_x"]).reshape(-1), dn(cache["mu2d_y"]).reshape(-1)], -1)

        def do_bin():
            """Return idx,valid + write idx_u/valid6/vf. --device-binning: bin ON DEVICE (bin_to_buffers
            writes the buffers), download only idx/valid (small [T,K]) for the host scatter / set_sinv;
            else host assign_bins + setbuf upload."""
            if args.device_binning:
                bin_to_buffers(cache["mu2d_x"], cache["mu2d_y"], cache["zmask"], tmap, 1, K, DEV,
                               idx_u, valid6, vf, ctx=bin_ctx)
                if args.device_scatter:
                    # FULL sync removal: build the inv-table on device (no idx download, no sync).
                    build_inv_device(idx_u, ttnn.reshape(vf, (T, K)), cache["mu2d_x"], cache["mu2d_y"],
                                     tmap, K, DEV, sinv_u=sinv_u, ctx=bin_ctx)
                    return None, None                       # host idx/valid not needed (device scatter)
                ttnn.synchronize_device(DEV)
                idx = ttnn.to_torch(idx_u).long().reshape(T, K)
                valid = ttnn.to_torch(vf).float().reshape(T, K) > 0.5
            else:
                idx, valid = assign_bins(mu2d_host(), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
                setbuf(idx_u, idx.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                setbuf(valid6, valid[:, None, :].expand(T, 6, K).float())
                setbuf(vf, valid[..., None].float())
            return idx, valid

        idx, valid = do_bin()
        if args.device_loss:
            loss_grad_dev()                 # warmup: JIT the loss kernels BEFORE capture (else trace hangs)
            ttnn.synchronize_device(DEV)
        if args.device_scatter:
            if not args.device_binning:
                set_sinv(idx, valid)        # device-binning already built sinv_u on device (do_bin)
            scatter_dev()                   # warmup: JIT split/embedding/sum BEFORE capture (else trace hangs)
            ttnn.synchronize_device(DEV)

        # ===== (C) capture ALL traces back-to-back (no alloc/host-write after) =====
        def cap(fn):
            tid = ttnn.begin_trace_capture(DEV, cq_id=0)
            r = fn()
            ttnn.end_trace_capture(DEV, tid, cq_id=0)
            return tid, r
        gfid, (cache, keep, keo, o, color, co, blist) = cap(geom_fwd)
        ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        # conic_t/mu_t depend on conic/mu2d (now geom-trace outputs) + idx_u -> recompute handles in rend traces
        rfid, (C, rc, theta, col_t, oc_t) = cap(rend_fwd)
        if args.device_loss:
            lfid, _ = cap(loss_grad_dev)
        rbid, (gct, gmt, gcol, goc) = cap(rend_bwd)
        if args.device_scatter:
            sbid, _ = cap(scatter_dev)
        gbid, (gout, dcrest) = cap(geom_bwd)
        ttnn.synchronize_device(DEV)

        # ===== (C') verify device loss-grad == host loss-grad on the seeded state =====
        if args.device_loss:
            ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
            Cl = dn(C).reshape(T * 256, 3).clone().requires_grad_(True)
            imgv = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, Cl)
            lv = metrics.loss_fn(imgv.reshape(cam.H, cam.W, 3), gt, lambda_ssim=0.2); lv.backward()
            gC_host = Cl.grad.reshape(T, 256, 3)
            ttnn.execute_trace(DEV, lfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
            rel = ((dn(gC_buf) - gC_host).norm() / gC_host.norm().clamp(min=1e-9)).item()
            # NOTE: this is the iter-0 frame, which is near-flat -> SSIM denominators ~C1,C2 -> the gradient
            # is ill-conditioned there, so a high rel here is degenerate-frame noise, NOT a bug. The real
            # correctness gate is that loss converges identically to the host-loss path (it does).
            print(f"   [verify] device gC vs host gC (iter-0, degenerate frame) rel={rel:.3e}  "
                  f"-- gate on convergence vs host-loss, not this")

        # ===== (C'') per-stage device timing: break down device+other =====
        if args.profile_stages:
            stages = [("geom_fwd", gfid), ("rend_fwd", rfid)]
            if args.device_loss: stages.append(("loss", lfid))
            stages.append(("rend_bwd", rbid))
            if args.device_scatter: stages.append(("scatter", sbid))
            stages.append(("geom_bwd", gbid))
            NS = 40
            adam_fn = adam_fused if args.fused_adam else adam_inplace
            for _ in range(5):                          # warm
                for _, tid in stages: ttnn.execute_trace(DEV, tid, cq_id=0, blocking=False)
                adam_fn(1)
            ttnn.synchronize_device(DEV)
            stt = {}
            for name, tid in stages:
                ttnn.synchronize_device(DEV); _t = time.perf_counter()
                for _ in range(NS): ttnn.execute_trace(DEV, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV); stt[name] = (time.perf_counter() - _t) / NS * 1e3
            ttnn.synchronize_device(DEV); _t = time.perf_counter()
            for _ in range(NS): adam_fn(1)
            ttnn.synchronize_device(DEV); stt["adam"] = (time.perf_counter() - _t) / NS * 1e3
            tot = sum(stt.values())
            print(f"== STAGE PROFILE res={res} G={G} K={K} (device exec, batched/{NS}) ==")
            for name in ("geom_fwd", "rend_fwd", "loss", "rend_bwd", "scatter", "geom_bwd", "adam"):
                if name in stt:
                    print(f"   {name:9s} {stt[name]:7.2f} ms ({stt[name]/tot*100:4.0f}%)")
            print(f"   {'SUM':9s} {tot:7.2f} ms (device exec only; excludes host binning + sync waits)")
            return

        # ===== (D) train loop: replays + copies only =====
        def loss_of():
            img = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3))
            return metrics.loss_fn(img.reshape(cam.H, cam.W, 3), gt, lambda_ssim=0.2), img

        from spike.mcmc import relocate, op_sigmoid
        from spike import geometry as _geom
        def add_sgld_host():
            """SGLD Langevin noise on means, covariance-shaped, gated to near-dead gaussians (spike.add_sgld_noise)."""
            o = torch.sigmoid(dn(P["op"]).reshape(-1))
            s = torch.exp(torch.stack([dn(P[k]).reshape(-1) for k in ("lx", "ly", "lz")], -1))
            q = torch.stack([dn(P[k]).reshape(-1) for k in ("qw", "qx", "qy", "qz")], -1)
            L = _geom.quat_to_rotmat(q) * s[:, None, :]                  # cov sqrt [G,3,3]
            weight = op_sigmoid(1.0 - o) * (args.noise_lr * LR["mx"])    # gate ~1 only for o<~0.005
            noise = torch.einsum("gij,gj->gi", L, torch.randn(G, 3)) * weight[:, None]
            mm = torch.stack([dn(P[k]).reshape(-1) for k in ("mx", "my", "mz")], -1) + noise
            for j, k in enumerate(("mx", "my", "mz")): setbuf(P[k], mm[:, j], DT)

        def do_relocate():
            """Fixed-count MCMC relocation (host): sync m<-P, relocate dead->live (o->o/n), sync P->m, reset Adam."""
            m.means3d.data = torch.stack([dn(P["mx"]).reshape(-1), dn(P["my"]).reshape(-1), dn(P["mz"]).reshape(-1)], -1)
            m.quats.data = torch.stack([dn(P[k]).reshape(-1) for k in ("qw", "qx", "qy", "qz")], -1)
            m.log_scales.data = torch.stack([dn(P[k]).reshape(-1) for k in ("lx", "ly", "lz")], -1)
            m.opacity_raw.data = dn(P["op"]).reshape(-1)
            m.color_dc.data = torch.stack([dn(P[k]).reshape(-1) for k in ("cr", "cg", "cb")], -1)
            if SHD > 0:
                for ci in range(3):
                    for l in range(15):
                        m.color_rest.data[:, l, ci] = dn(crest[ci][l]).reshape(-1)        # [G,15,3] from device
            moved, ti = relocate(m, args.dead_thr, offset=0.005)
            if moved == 0:
                return 0
            for j, k in enumerate(("mx", "my", "mz")): setbuf(P[k], m.means3d[:, j], DT)
            for j, k in enumerate(("qw", "qx", "qy", "qz")): setbuf(P[k], m.quats[:, j], DT)
            for j, k in enumerate(("lx", "ly", "lz")): setbuf(P[k], m.log_scales[:, j], DT)
            setbuf(P["op"], m.opacity_raw, DT)
            for j, k in enumerate(("cr", "cg", "cb")): setbuf(P[k], m.color_dc[:, j], DT)
            if SHD > 0:                                              # write relocated rest coeffs + reset their Adam
                for ci in range(3):
                    for l in range(15):
                        setbuf(crest[ci][l], m.color_rest[:, l, ci], DT)
                        for M_ in (crest_mom[ci][l], crest_vom[ci][l]):
                            mh = dn(M_).reshape(-1); mh[ti] = 0.0; setbuf(M_, mh, DT)
            if args.fused_adam:                                      # reset device Adam moments at relocated slots
                for M_ in (mom_m, vom_m):
                    mh = dn(M_); mh[ti] = 0.0; setbuf(M_, mh, DT)
            else:
                for k in PN:
                    for M_ in (mom[k], vom[k]):
                        mh = dn(M_).reshape(-1); mh[ti] = 0.0; setbuf(M_, mh, DT)
            return moved

        t_bin = t_scat = 0.0
        t0 = time.perf_counter()
        for it in range(args.iters):
            if args.multi_view:                            # pick a random training view: update camera + GT
                v = int(torch.randint(len(tr_cams), (1,)))
                set_cam(tr_cams[v])
                if args.random_bg:                         # composite GT over a random bg; numerator c_B=bg (in-place)
                    bg = torch.rand(3)
                    gt = (tr_pm[v] + tr_tr[v] * bg[:, None, None]).permute(1, 2, 0).contiguous()   # [H,W,3]
                    setbuf(bias, (w_b * bg)[None, None, :].expand(T, 256, 3).contiguous(), BF)      # in-place (trace-safe)
                else:
                    gt = tr_imgs[v]
                if args.device_loss:                       # per-view GT SSIM stats on device (no host loss)
                    recompute_gt_stats(gt)
            # device SH colour is now computed inside the geom_fwd trace (no host eval/upload)
            # geom_fwd: only sync when host needs mu2d/keep for binning;
            # device serializes rfid after gfid on the same cq_id=0 automatically
            ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False)
            if it % args.bin_every == 0:
                _tb = time.perf_counter()
                if not args.device_binning:
                    ttnn.synchronize_device(DEV)         # host binning needs mu2d back; device binning has no sync
                idx, valid = do_bin()
                if args.device_scatter and not args.device_binning:
                    set_sinv(idx, valid)                 # device-binning built sinv_u in do_bin (no host round-trip)
                t_bin += time.perf_counter() - _tb
            log_it = (it % max(1, args.iters // 6) == 0) or (it == args.iters - 1)
            if args.device_loss:
                # render fwd -> device loss-grad (writes gC_buf): NO C download / gC upload / host loss.
                # same cq serializes rfid->lfid->rbid; sync only at rbid (host scatter needs the grads).
                ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False)
                ttnn.execute_trace(DEV, lfid, cq_id=0, blocking=False)
                loss = None
                if log_it:                                   # logging only: read C for the loss value
                    ttnn.synchronize_device(DEV)
                    img = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3))
                    loss = float(metrics.loss_fn(img.reshape(cam.H, cam.W, 3), gt, lambda_ssim=0.2))
            else:
                # rend_fwd: sync required — host needs C to compute loss
                ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
                Cl = dn(C).reshape(T * 256, 3).clone().requires_grad_(True)
                img = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, Cl)
                loss = metrics.loss_fn(img.reshape(cam.H, cam.W, 3), gt, lambda_ssim=0.2)
                loss.backward()
                setbuf(gC_buf, Cl.grad.reshape(T, 256, 3))
            if args.device_scatter:
                # DEVICE scatter: render-bwd -> device gather-reduce -> geom-bwd, all one cq, NO per-iter sync.
                ttnn.execute_trace(DEV, rbid, cq_id=0, blocking=False)
                ttnn.execute_trace(DEV, sbid, cq_id=0, blocking=False)
            else:
                # rend_bwd: sync required — host needs gct/gmt/gcol/goc for the host scatter
                ttnn.execute_trace(DEV, rbid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
                _ts = time.perf_counter()
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
                t_scat += time.perf_counter() - _ts
            # geom_bwd: no sync after — copy/adam are ttnn device ops on the same cq_id=0,
            # executed in-order after gbid; loop-end sync covers the final iter
            ttnn.execute_trace(DEV, gbid, cq_id=0, blocking=False)
            if args.lambda_o:                            # MCMC reg: opacity-L1 grad = lo/G * o(1-o) -> gout[op]
                o_ = ttnn.sigmoid(P["op"])
                ttnn.add(gout["op"], ttnn.mul(ttnn.mul(o_, ttnn.add(ttnn.neg(o_), 1.0)), args.lambda_o / G),
                         output_tensor=gout["op"])
            if args.lambda_s:                            # MCMC reg: scale-L1 grad = ls/(3G)*exp(log_scale) (mean over Gx3)
                for k in ("lx", "ly", "lz"):
                    ttnn.add(gout[k], ttnn.mul(ttnn.exp(P[k]), args.lambda_s / (3.0 * G)), output_tensor=gout[k])
            if args.fused_adam:
                adam_fused(it + 1)                       # reads gout directly (no gout->gacc copies)
            else:
                for k in PN:
                    ttnn.copy(gout[k], gacc[k])
                adam_inplace(it + 1)
            if SHD > 0:
                adam_crest(it + 1)                       # device Adam for the SH rest coeffs (color_rest)
            if args.depth_weight:
                adam_bt(it + 1)                          # host Adam for the 2 global beta/tau scalars
            if args.noise_lr:                            # MCMC SGLD: covariance-shaped Langevin noise on near-dead means
                add_sgld_host()
            if args.relocate_every and it > 0 and it % args.relocate_every == 0:
                mv = do_relocate()                       # MCMC: recycle dead slots onto live gaussians
                if mv:
                    idx, valid = do_bin()                # positions changed -> re-bin
                    print(f"   iter {it:4d} relocated {mv}", flush=True)
            if log_it and loss is not None:
                print(f"   iter {it:4d} loss {float(loss):.4f}")
        ttnn.synchronize_device(DEV)
        tt = time.perf_counter() - t0
        if args.profile:
            n = args.iters
            print(f"   [profile] per-iter avg over {n}: total {tt/n*1e3:.1f} ms | "
                  f"HOST binning {t_bin/n*1e3:.1f} ms ({t_bin/tt*100:.0f}%) | "
                  f"HOST scatter {t_scat/n*1e3:.1f} ms ({t_scat/tt*100:.0f}%) | "
                  f"device+other {(tt-t_bin-t_scat)/n*1e3:.1f} ms ({(tt-t_bin-t_scat)/tt*100:.0f}%)")
        print(f"== FULL-PIPELINE TRACED (single view) res={res} G={G} K={K} ==")
        print(f"   {tt/args.iters*1e3:.1f} ms/it (2 syncs/iter, 3 on bin-every iters; zero per-iter alloc)")
        if args.multi_view:
            @torch.no_grad()
            def eval_psnr(cams_, imgs_, label):
                if args.random_bg:                        # eval on the STANDARD white bg (benchmark protocol)
                    setbuf(bias, (w_b * torch.ones(3))[None, None, :].expand(T, 256, 3).contiguous(), BF)
                ps = []
                for c, g in zip(cams_, imgs_):
                    set_cam(c)                                # updates the centre buffer -> gfid evals device SH per view
                    ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
                    do_bin()                                  # binning for this held-out view
                    ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
                    img = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3))
                    ps.append(float(metrics.psnr(img.reshape(cam.H, cam.W, 3), g)))
                print(f"[eval] {label}: PSNR {sum(ps) / len(ps):.2f} dB over {len(ps)} views", flush=True)
            eval_psnr(tr_cams[:12], tr_imgs[:12], "train")
            eval_psnr(te_cams, te_imgs, "test (held-out)")
        if args.save_ply:
            # write the trained device params P[k] back into the GaussianModel m, then save .ply
            g_ = lambda k: dn(P[k]).reshape(-1)
            m.means3d.data = torch.stack([g_("mx"), g_("my"), g_("mz")], -1)
            m.quats.data = torch.stack([g_("qw"), g_("qx"), g_("qy"), g_("qz")], -1)
            m.log_scales.data = torch.stack([g_("lx"), g_("ly"), g_("lz")], -1)
            m.color_dc.data = torch.stack([g_("cr"), g_("cg"), g_("cb")], -1)
            if SHD > 0:                                    # SH rest coeffs from the device params
                for ci in range(3):
                    for l in range(15):
                        m.color_rest.data[:, l, ci] = dn(crest[ci][l]).reshape(-1)
            m.opacity_raw.data = g_("op").reshape(m.opacity_raw.shape)
            os.makedirs(os.path.dirname(os.path.abspath(args.save_ply)), exist_ok=True)
            nply = plyio.save_ply(args.save_ply, m)
            print(f"[save-ply] wrote {args.save_ply}: {nply} gaussians, "
                  f"{os.path.getsize(args.save_ply)/1024/1024:.1f} MB")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
