"""FULL ASSEMBLY: the all-device resident trainer. Params + Adam moments resident on device;
per iter the forward (geometry -> activations -> gather -> binned render) and backward (render-bwd ->
theta-bwd -> scatter[host] -> geometry-bwd) and Adam all run on device, with the ONLY host work being
binning indices (every-N) + the loss gradient (small image round-trip). Built from the verified pieces:
device_fwd_core/device_bwd_core (geometry), the gather+theta build, the m4_train_step render fwd/bwd,
and the resident_core device Adam. Per-view loop (no tiling); UNTRACED first (correctness), traces later.

Oracle: same held-out PSNR as tools/m4_train_binned.py / train_manual.py.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/resident_trainer.py --res 96 --G 4000 --iters 60
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

from spike import data, metrics, sh
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins, K_POLY
from geom_device import device_fwd_core, device_bwd_core, A, M, S
from resident_core import adam_dev

C0 = sh.C0
DEV = None
CG = None
DT = ttnn.float32        # geometry/params/Adam in fp32 (geometry needs it)
BF = ttnn.bfloat16       # render in bf16


def u(t, dt=DT):
    return ttnn.from_torch(t.reshape(-1, 1).contiguous() if t.dim() == 1 else t.contiguous(),
                           dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV)


def T3(t):
    return ttnn.transpose(t, -2, -1)


def setbuf(buf, t):
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(t.reshape(-1, 1).contiguous(), dtype=DT, layout=ttnn.TILE_LAYOUT), buf)


# ---------------- gather + theta build (device, bf16 for render path) ----------------
def gather_theta(conic, mu2d, color_o, o_col, idx, valid, origins, T, K):
    ii = ttnn.from_torch(idx.to(torch.int32).reshape(T, K), dtype=ttnn.uint32,
                         layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)

    def emb(tab):  # tab device [G,D] (fp32) -> bf16 table -> [T,K,D]
        return ttnn.embedding(ii, ttnn.typecast(tab, BF) if tab.get_dtype() != BF else tab)
    conic_t, mu_t = emb(conic), emb(mu2d)
    color_t, ocol_t = emb(color_o), emb(o_col)
    org = ttnn.from_torch(origins[:, None, :].expand(T, K, 2).contiguous(), dtype=BF, layout=ttnn.TILE_LAYOUT, device=DEV)
    mu = ttnn.add(mu_t, ttnn.neg(org))
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    mux, muy = mu[:, :, 0:1], mu[:, :, 1:2]
    mux2, muy2, muxy = ttnn.mul(mux, mux), ttnn.mul(muy, muy), ttnn.mul(mux, muy)
    t2 = ttnn.mul(b, 2.0)
    t3 = ttnn.mul(ttnn.add(ttnn.mul(a, mux), ttnn.mul(b, muy)), -2.0)
    t4 = ttnn.mul(ttnn.add(ttnn.mul(b, mux), ttnn.mul(c, muy)), -2.0)
    t5 = ttnn.add(ttnn.add(ttnn.mul(a, mux2), ttnn.mul(ttnn.mul(b, muxy), 2.0)), ttnn.mul(c, muy2))
    tQ = ttnn.concat([a, c, t2, t3, t4, t5], dim=-1)
    bump = torch.zeros(T, K, 6); bump[..., 5] = 1.0
    theta = ttnn.transpose(ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), u(bump, BF)), 1, 2)
    valid6 = u(valid[:, None, :].expand(T, 6, K).float().contiguous(), BF)
    vf = u(valid[..., None].float(), BF)
    theta = ttnn.mul(theta, valid6)
    return theta, ttnn.mul(color_t, vf), ttnn.mul(ocol_t, vf), conic_t, mu_t


# ---------------- render fwd/bwd (device, bf16) ----------------
def render_fwd(Phi, thU, col, oc, wb, bias):
    relu_Q = ttnn.relu(ttnn.matmul(Phi, thU, core_grid=CG))
    w = ttnn.square(relu_Q)
    den = ttnn.add(ttnn.matmul(w, oc, core_grid=CG), wb)
    num = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)
    return ttnn.div(num, den), (relu_Q, w, den, num)


def render_bwd(Phi, thU, col, oc, gC, cache):
    relu_Q, w, den, num = cache
    C = ttnn.div(num, den)
    gnum = ttnn.div(gC, den)
    gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(gC, C), dim=-1, keepdim=True)), den)
    gcol = ttnn.matmul(T3(w), gnum, core_grid=CG)
    goc = ttnn.matmul(T3(w), gden, core_grid=CG)
    gw = ttnn.add(ttnn.matmul(gnum, T3(col), core_grid=CG), ttnn.matmul(gden, T3(oc), core_grid=CG))
    gthU = ttnn.matmul(T3(Phi), ttnn.mul(gw, ttnn.mul(relu_Q, 2.0)), core_grid=CG)
    return gthU, gcol, goc, gnum, gden


# ---------------- theta-bwd (device): gthU[T,6,K] -> gconic_t[T,K,3], gmu_t[T,K,2] ----------------
def theta_bwd(gthU, conic_t, mu_t, origins, valid, T, K):
    valid6 = u(valid[:, None, :].expand(T, 6, K).float().contiguous(), BF)
    gpoly = ttnn.transpose(ttnn.mul(gthU, valid6), 1, 2)
    gtQ = ttnn.mul(gpoly, -1.0 / K_POLY)
    g0, g1, g2 = gtQ[:, :, 0:1], gtQ[:, :, 1:2], gtQ[:, :, 2:3]
    g3, g4, g5 = gtQ[:, :, 3:4], gtQ[:, :, 4:5], gtQ[:, :, 5:6]
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    org = u(origins[:, None, :].expand(T, K, 2).contiguous(), BF)
    mu = ttnn.add(mu_t, ttnn.neg(org))
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


# ---------------- resident params + device Adam ----------------
PNAMES = ["mx", "my", "mz", "qw", "qx", "qy", "qz", "lx", "ly", "lz", "cr", "cg", "cb", "op"]


class Resident:
    def __init__(self, model, lr):
        cols = dict(mx=model.means3d[:, 0], my=model.means3d[:, 1], mz=model.means3d[:, 2],
                    qw=model.quats[:, 0], qx=model.quats[:, 1], qy=model.quats[:, 2], qz=model.quats[:, 3],
                    lx=model.log_scales[:, 0], ly=model.log_scales[:, 1], lz=model.log_scales[:, 2],
                    cr=model.color_dc[:, 0], cg=model.color_dc[:, 1], cb=model.color_dc[:, 2],
                    op=model.opacity_raw)
        self.P = {k: u(v.detach()) for k, v in cols.items()}
        G = model.G
        self.m = {k: u(torch.zeros(G)) for k in PNAMES}
        self.v = {k: u(torch.zeros(G)) for k in PNAMES}
        self.lr = {**{k: lr["means"] for k in ("mx", "my", "mz")},
                   **{k: lr["quats"] for k in ("qw", "qx", "qy", "qz")},
                   **{k: lr["scales"] for k in ("lx", "ly", "lz")},
                   **{k: lr["color"] for k in ("cr", "cg", "cb")}, "op": lr["opacity"]}
        self.wb = float(model.w_b_raw)          # tiny scalar param kept on host
        self.wb_m = self.wb_v = 0.0
        self.lr_wb = lr["wb"]

    def adam(self, grads, t, gwb):
        for k in PNAMES:
            p2, self.m[k], self.v[k] = adam_dev(self.P[k], self.m[k], self.v[k], grads[k], t, self.lr[k])
            ttnn.copy(p2, self.P[k])     # write into the FIXED param buffer (so geom traces read updates)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.wb_m = b1 * self.wb_m + (1 - b1) * gwb
        self.wb_v = b2 * self.wb_v + (1 - b2) * gwb * gwb
        mh = self.wb_m / (1 - b1 ** t); vh = self.wb_v / (1 - b2 ** t)
        self.wb -= self.lr_wb * mh / (math.sqrt(vh) + eps)


def activations(R):
    sx, sy, sz = ttnn.exp(R.P["lx"]), ttnn.exp(R.P["ly"]), ttnn.exp(R.P["lz"])
    color = [ttnn.relu(A(M(R.P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]   # clamp(0.5+C0*dc, 0)
    o = ttnn.sigmoid(R.P["op"])
    w_b = math.log1p(math.exp(R.wb)) if R.wb < 20 else R.wb                   # softplus(host scalar)
    return (sx, sy, sz), color, o, w_b


def dn(t):
    return ttnn.to_torch(t).float()


def train(args):
    global DEV, CG
    tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, stride=max(1, 100 // args.n_train))
    te_c, te_i = data.load_blender(args.scene, "test", res=args.res, n=args.n_test, stride=max(1, 200 // args.n_test))
    tmap = TileMap(args.res, args.res)
    T, K, G = tmap.T, args.K, args.G
    torch.manual_seed(args.seed)
    model = GaussianModel(G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    R = Resident(model, DEFAULT_LR)
    cams_s = [( [[float(c.R_v[i, j]) for j in range(3)] for i in range(3)],
                [float(c.t_v[i]) for i in range(3)], float(c.fx), float(c.fy), float(c.cx), float(c.cy)) for c in tr_c]
    Phi = u(tmap.Phi.unsqueeze(0).expand(T, 256, 6).contiguous(), BF)
    c_b = model.c_b
    bins = [None] * len(tr_c)

    GF = None
    if args.traced_geom:
        GF = []
        for vi in range(len(tr_c)):
            Rv, tv, fx, fy, cx, cy = cams_s[vi]
            gbuf = {k: u(torch.zeros(G)) for k in ("a", "b", "c", "mx", "my")}

            def mkfwd(Rv, tv, fx, fy, cx, cy):
                def f():
                    sx, sy, sz = ttnn.exp(R.P["lx"]), ttnn.exp(R.P["ly"]), ttnn.exp(R.P["lz"])
                    cols = (R.P["mx"], R.P["my"], R.P["mz"], R.P["qw"], R.P["qx"], R.P["qy"], R.P["qz"], sx, sy, sz)
                    return device_fwd_core(cols, Rv, tv, fx, fy, cx, cy)
                return f

            def mkbwd(cache, gbuf):
                def b():
                    return device_bwd_core(cache, gbuf["a"], gbuf["b"], gbuf["c"], gbuf["mx"], gbuf["my"])
                return b
            fwd = mkfwd(Rv, tv, fx, fy, cx, cy)
            conic, mu2d, cache = fwd(); ttnn.synchronize_device(DEV)
            fid = ttnn.begin_trace_capture(DEV, cq_id=0)
            conic, mu2d, cache = fwd()
            ttnn.end_trace_capture(DEV, fid, cq_id=0); ttnn.synchronize_device(DEV)
            bwd = mkbwd(cache, gbuf)
            gout = bwd(); ttnn.synchronize_device(DEV)
            bid = ttnn.begin_trace_capture(DEV, cq_id=0)
            gout = bwd()
            ttnn.end_trace_capture(DEV, bid, cq_id=0); ttnn.synchronize_device(DEV)
            GF.append(dict(fid=fid, bid=bid, conic=conic, mu2d=mu2d, cache=cache, gbuf=gbuf, gout=gout))

    def step(it):
        (sx, sy, sz), color, o, w_b = activations(R)
        keo_cache, cache_geo, cache_ren, gttheta_meta = [], [], [], []
        imgs_C = []
        # ---- forward all views ----
        for vi, cam in enumerate(tr_c):
            Rv, tv, fx, fy, cx, cy = cams_s[vi]
            if GF is not None:
                ttnn.execute_trace(DEV, GF[vi]["fid"], cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
                conic, mu2d, cgeo = GF[vi]["conic"], GF[vi]["mu2d"], GF[vi]["cache"]
            else:
                cols = (R.P["mx"], R.P["my"], R.P["mz"], R.P["qw"], R.P["qx"], R.P["qy"], R.P["qz"], sx, sy, sz)
                conic, mu2d, cgeo = device_fwd_core(cols, Rv, tv, fx, fy, cx, cy)
            keep = cgeo["zmask"]
            keo = M(keep, o)
            color_o = ttnn.concat([M(keo, color[0]), M(keo, color[1]), M(keo, color[2])], dim=-1)  # [G,3] fp32
            o_col = ttnn.reshape(keo, (G, 1)) if keo.shape[-1] != 1 else keo
            mu2d_h = dn(mu2d)
            keep_h = dn(keep).reshape(-1) > 0.5
            if it % args.bin_every == 0 or bins[vi] is None:
                bins[vi] = assign_bins(mu2d_h, keep_h, tmap, 1, K)
            idx, valid = bins[vi]
            theta, col_t, oc_t, conic_t, mu_t = gather_theta(conic, mu2d, color_o, o_col, idx, valid, tmap.origins, T, K)
            bias = u((w_b * c_b)[None, None, :].expand(T, 256, 3).contiguous(), BF)
            C, cren = render_fwd(Phi, theta, col_t, oc_t, float(w_b), bias)
            imgs_C.append(dn(C).reshape(T * 256, 3))
            cache_geo.append((cgeo, conic, mu2d, idx, valid, keep, keo, color_o))
            cache_ren.append((theta, col_t, oc_t, cren, conic_t, mu_t))
        # ---- loss + gC (host, tiny graph) ----
        loss = 0.0
        Cl = [c.clone().requires_grad_(True) for c in imgs_C]
        for vi, (gt, Cv) in enumerate(zip(tr_i, Cl)):
            H, W = gt.shape[0], gt.shape[1]
            img = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, Cv)
            loss = loss + metrics.loss_fn(img.reshape(H, W, 3), gt, lambda_ssim=0.2)
        loss = loss / len(tr_c)
        loss.backward()
        # ---- backward all views ----
        gacc = {k: u(torch.zeros(G)) for k in PNAMES}
        gcolor = [u(torch.zeros(G)) for _ in range(3)]
        go = u(torch.zeros(G))
        gwb_tot = 0.0
        for vi in range(len(tr_c)):
            cgeo, conic, mu2d, idx, valid, keep, keo, color_o = cache_geo[vi]
            theta, col_t, oc_t, cren, conic_t, mu_t = cache_ren[vi]
            gC = u(Cl[vi].grad.reshape(T, 256, 3), BF)
            gthU, gcol, goc, gnum, gden = render_bwd(Phi, theta, col_t, oc_t, gC, cren)
            gconic_t, gmu_t = theta_bwd(gthU, conic_t, mu_t, tmap.origins, valid, T, K)
            # scatter (host): [T,K,*] grads -> [G,*]
            flat = idx.reshape(-1)
            vf = valid[..., None].float()
            gconic = torch.zeros(G, 3).index_add_(0, flat, (dn(gconic_t)).reshape(-1, 3))
            gmu2d = torch.zeros(G, 2).index_add_(0, flat, (dn(gmu_t)).reshape(-1, 2))
            gcolor_o = torch.zeros(G, 3).index_add_(0, flat, (dn(gcol) * vf).reshape(-1, 3))
            go_col = torch.zeros(G, 1).index_add_(0, flat, (dn(goc) * vf).reshape(-1, 1))
            # geometry bwd (device): traced replay or direct
            if GF is not None:
                gb = GF[vi]["gbuf"]
                setbuf(gb["a"], gconic[:, 0]); setbuf(gb["b"], gconic[:, 1]); setbuf(gb["c"], gconic[:, 2])
                setbuf(gb["mx"], gmu2d[:, 0]); setbuf(gb["my"], gmu2d[:, 1])
                ttnn.execute_trace(DEV, GF[vi]["bid"], cq_id=0, blocking=False)
                ttnn.synchronize_device(DEV)
                gg = GF[vi]["gout"]
            else:
                gg = device_bwd_core(cgeo, u(gconic[:, 0]), u(gconic[:, 1]), u(gconic[:, 2]), u(gmu2d[:, 0]), u(gmu2d[:, 1]))
            sx, sy, sz = ttnn.exp(R.P["lx"]), ttnn.exp(R.P["ly"]), ttnn.exp(R.P["lz"])
            gmap = {"mx": gg["gmx"], "my": gg["gmy"], "mz": gg["gmz"], "qw": gg["gqw"], "qx": gg["gqx"],
                    "qy": gg["gqy"], "qz": gg["gqz"], "lx": M(gg["gsx"], sx), "ly": M(gg["gsy"], sy), "lz": M(gg["gsz"], sz)}
            for k in ("mx", "my", "mz", "qw", "qx", "qy", "qz", "lx", "ly", "lz"):
                gacc[k] = A(gacc[k], gmap[k])
            # color/opacity bwd (device): color_o = keo*color ; o_col = keo ; keo = keep*o
            gco = [u(gcolor_o[:, j]) for j in range(3)]
            gocl = u(go_col[:, 0])
            color = [ttnn.relu(A(M(R.P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
            gkeo = A(A(M(gco[0], color[0]), M(gco[1], color[1])), A(M(gco[2], color[2]), gocl))
            for j in range(3):
                gcolor[j] = A(gcolor[j], M(gco[j], keo))
            go = A(go, M(gkeo, keep))
            gwb_tot += float((dn(gnum) * c_b[None, None, :]).sum() + dn(gden).sum())
        # color_dc/opacity grads
        cmaskj = [ttnn.gtz(A(M(R.P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
        o = ttnn.sigmoid(R.P["op"])
        gacc["cr"] = M(M(gcolor[0], cmaskj[0]), C0); gacc["cg"] = M(M(gcolor[1], cmaskj[1]), C0); gacc["cb"] = M(M(gcolor[2], cmaskj[2]), C0)
        gacc["op"] = M(go, M(o, S(u(torch.ones(G)), o)))
        gwb_raw = gwb_tot * (1.0 / (1.0 + math.exp(-R.wb)))     # softplus'(wb)=sigmoid(wb)
        R.adam(gacc, it + 1, gwb_raw)
        return float(loss)

    def eval_psnr():
        # host render via the resident params read back (uses the same device forward)
        ps = []
        (sx, sy, sz), color, o, w_b = activations(R)
        for cam, gt in zip(te_c, te_i):
            Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]; tv = [float(cam.t_v[i]) for i in range(3)]
            cols = (R.P["mx"], R.P["my"], R.P["mz"], R.P["qw"], R.P["qx"], R.P["qy"], R.P["qz"], sx, sy, sz)
            conic, mu2d, cgeo = device_fwd_core(cols, Rv, tv, cam.fx, cam.fy, cam.cx, cam.cy)
            keep = cgeo["zmask"]; keo = M(keep, o)
            color_o = ttnn.concat([M(keo, color[0]), M(keo, color[1]), M(keo, color[2])], dim=-1)
            o_col = ttnn.reshape(keo, (G, 1)) if keo.shape[-1] != 1 else keo
            idx, valid = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
            theta, col_t, oc_t, _, _ = gather_theta(conic, mu2d, color_o, o_col, idx, valid, tmap.origins, T, K)
            bias = u((w_b * c_b)[None, None, :].expand(T, 256, 3).contiguous(), BF)
            C, _ = render_fwd(Phi, theta, col_t, oc_t, float(w_b), bias)
            img = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, dn(C).reshape(T * 256, 3))
            ps.append(float(metrics.psnr(img.reshape(cam.H, cam.W, 3), gt)))
        return sum(ps) / len(ps)

    print(f"== FULL-RESIDENT trainer res={args.res} G={G} K={K} {args.n_train}tr/{args.n_test}te {args.iters}it bin{args.bin_every} ==")
    step(0)  # warm
    t0 = time.perf_counter()
    for it in range(args.iters):
        L = step(it)
        if it % max(1, args.iters // 6) == 0:
            print(f"   iter {it:4d} loss {L:.4f}")
    tt = time.perf_counter() - t0
    print(f"   {tt/args.iters*1e3:.0f} ms/it | held-out {eval_psnr():.2f} dB")


def main():
    global DEV, CG
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--G", type=int, default=4000)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-every", type=int, default=5)
    ap.add_argument("--traced-geom", action="store_true", help="trace the per-view geometry fwd/bwd")
    args = ap.parse_args()
    DEV = ttnn.open_device(device_id=0)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        train(args)
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
