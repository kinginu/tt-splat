"""Full-pipeline trace — verify a TRACED forward (geom-fwd trace + gather+render-fwd
trace, persistent I/O, NO untraced per-iter allocation) produces a finite image matching the untraced
reference. This tests the fix for the trace-allocation hazard: if gather/render run INSIDE traces (not
untraced after the geom trace), nothing allocates per-iter and the trace buffers are not corrupted.

Single view. Oracle: the untraced device forward (device_fwd_core + gather + render) on the same
params. PASS = finite + matches.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/traced_fwd.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike import data, sh
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins, K_POLY
from geom_device import device_fwd_core, A, M

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


def gather_theta_buf(conic, mu2d, color_o, o_col, idx_u, valid6, vf, origins_t, bump_buf, T, K):
    """All inputs persistent device tensors (no from_torch inside -> trace-safe)."""
    def emb(tab):
        return ttnn.embedding(idx_u, ttnn.typecast(tab, BF))
    conic_t, mu_t = emb(conic), emb(mu2d)
    color_t, ocol_t = emb(color_o), emb(o_col)
    mu = ttnn.add(mu_t, ttnn.neg(origins_t))
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    mux, muy = mu[:, :, 0:1], mu[:, :, 1:2]
    mux2, muy2, muxy = ttnn.mul(mux, mux), ttnn.mul(muy, muy), ttnn.mul(mux, muy)
    t2 = ttnn.mul(b, 2.0)
    t3 = ttnn.mul(ttnn.add(ttnn.mul(a, mux), ttnn.mul(b, muy)), -2.0)
    t4 = ttnn.mul(ttnn.add(ttnn.mul(b, mux), ttnn.mul(c, muy)), -2.0)
    t5 = ttnn.add(ttnn.add(ttnn.mul(a, mux2), ttnn.mul(ttnn.mul(b, muxy), 2.0)), ttnn.mul(c, muy2))
    tQ = ttnn.concat([a, c, t2, t3, t4, t5], dim=-1)
    theta = ttnn.transpose(ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), bump_buf), 1, 2)
    theta = ttnn.mul(theta, valid6)
    return theta, ttnn.mul(color_t, vf), ttnn.mul(ocol_t, vf)


def gather_theta_cols(ca, cb, cc, mu2d_x, mu2d_y, col0, col1, col2, o_col,
                      idx_u, valid6, vf, origins_t, bump_buf, T, K):
    """Large-G gather: embed the conic/mu/color COLUMNS [G,1] separately (G-independent L1, reads from
    DRAM) instead of one [G,k] embedding of a per-gaussian concat -> avoids the [G,k] concat L1 wall.
    All concats here are at per-tile-slot [T,K,·] scale (G-independent)."""
    def emb(tab):
        return ttnn.embedding(idx_u, ttnn.typecast(tab, BF))
    a, b, c = emb(ca), emb(cb), emb(cc)                                 # [T,K,1] each
    mux = ttnn.add(emb(mu2d_x), ttnn.neg(origins_t[:, :, 0:1]))
    muy = ttnn.add(emb(mu2d_y), ttnn.neg(origins_t[:, :, 1:2]))
    mux2, muy2, muxy = ttnn.mul(mux, mux), ttnn.mul(muy, muy), ttnn.mul(mux, muy)
    t2 = ttnn.mul(b, 2.0)
    t3 = ttnn.mul(ttnn.add(ttnn.mul(a, mux), ttnn.mul(b, muy)), -2.0)
    t4 = ttnn.mul(ttnn.add(ttnn.mul(b, mux), ttnn.mul(c, muy)), -2.0)
    t5 = ttnn.add(ttnn.add(ttnn.mul(a, mux2), ttnn.mul(ttnn.mul(b, muxy), 2.0)), ttnn.mul(c, muy2))
    tQ = ttnn.concat([a, c, t2, t3, t4, t5], dim=-1)                    # [T,K,6] G-independent
    theta = ttnn.transpose(ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), bump_buf), 1, 2)
    theta = ttnn.mul(theta, valid6)
    color_t = ttnn.concat([emb(col0), emb(col1), emb(col2)], dim=-1)    # [T,K,3] G-independent
    return theta, ttnn.mul(color_t, vf), ttnn.mul(emb(o_col), vf)


def conic_mu_cols_gather(ca, cb, cc, mu2d_x, mu2d_y, idx_u, T, K):
    """Backward helper: gather the conic/mu columns -> conic_t[T,K,3], mu_t[T,K,2] (for theta_bwd)."""
    def emb(tab):
        return ttnn.embedding(idx_u, ttnn.typecast(tab, BF))
    conic_t = ttnn.concat([emb(ca), emb(cb), emb(cc)], dim=-1)
    mu_t = ttnn.concat([emb(mu2d_x), emb(mu2d_y)], dim=-1)
    return conic_t, mu_t


def render_fwd(Phi, thU, col, oc, wb_buf, bias):
    relu_Q = ttnn.relu(ttnn.matmul(Phi, thU, core_grid=CG))
    w = ttnn.square(relu_Q)
    den = ttnn.add(ttnn.matmul(w, oc, core_grid=CG), wb_buf)
    num = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)
    return ttnn.div(num, den)


def main():
    global DEV, CG
    DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        res, G, K = 96, 4000, 128
        cams, _ = data.load_blender("data/nerf_synthetic/ficus", "train", res=res, n=1)
        cam = cams[0]
        tmap = TileMap(res, res)
        T = tmap.T
        torch.manual_seed(0)
        m = GaussianModel(G, extent=1.5, seed=0)
        Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
        tv = [float(cam.t_v[i]) for i in range(3)]
        fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
        w_b = float(torch.nn.functional.softplus(m.w_b_raw))

        # resident params (geometry + color/opacity) as [G,1] buffers
        P = dict(mx=u(m.means3d[:, 0]), my=u(m.means3d[:, 1]), mz=u(m.means3d[:, 2]),
                 qw=u(m.quats[:, 0]), qx=u(m.quats[:, 1]), qy=u(m.quats[:, 2]), qz=u(m.quats[:, 3]),
                 lx=u(m.log_scales[:, 0]), ly=u(m.log_scales[:, 1]), lz=u(m.log_scales[:, 2]),
                 cr=u(m.color_dc[:, 0]), cg=u(m.color_dc[:, 1]), cb=u(m.color_dc[:, 2]), op=u(m.opacity_raw))

        def geom_fwd():
            sx, sy, sz = ttnn.exp(P["lx"]), ttnn.exp(P["ly"]), ttnn.exp(P["lz"])
            cols = (P["mx"], P["my"], P["mz"], P["qw"], P["qx"], P["qy"], P["qz"], sx, sy, sz)
            conic, mu2d, cache = device_fwd_core(cols, Rv, tv, fx, fy, cx, cy)
            keep = cache["zmask"]
            o = ttnn.sigmoid(P["op"])
            keo = M(keep, o)
            color = [ttnn.relu(A(M(P[k], C0), 0.5)) for k in ("cr", "cg", "cb")]
            color_o = ttnn.concat([M(keo, color[0]), M(keo, color[1]), M(keo, color[2])], dim=-1)
            return conic, mu2d, keep, color_o, keo

        def setbuf(buf, t, dt=BF, layout=ttnn.TILE_LAYOUT):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t.contiguous(), dtype=dt, layout=layout), buf)

        # ---- (A) ALLOCATE every persistent buffer BEFORE any trace capture ----
        idx_u = ttnn.from_torch(torch.zeros(T, K, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        valid6 = u(torch.zeros(T, 6, K), BF)
        vf = u(torch.zeros(T, K, 1), BF)
        origins_t = u(tmap.origins[:, None, :].expand(T, K, 2).contiguous(), BF)
        Phi = u(tmap.Phi.unsqueeze(0).expand(T, 256, 6).contiguous(), BF)
        bias = u((w_b * m.c_b)[None, None, :].expand(T, 256, 3).contiguous(), BF)
        wb_buf = u(torch.full((T, 256, 1), w_b), BF)
        bump = torch.zeros(T, K, 6); bump[..., 5] = 1.0
        bump_buf = u(bump, BF)

        # ---- (B) untraced reference FIRST (allocations fine; no trace yet) + real binning ----
        conic, mu2d, keep, color_o, o_col = geom_fwd()
        mu2d_h = dn(mu2d); keep_h = dn(keep).reshape(-1) > 0.5
        idx, valid = assign_bins(mu2d_h, keep_h, tmap, 1, K)
        setbuf(idx_u, idx.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        setbuf(valid6, valid[:, None, :].expand(T, 6, K).float())
        setbuf(vf, valid[..., None].float())

        def rend_fwd():
            theta, col_t, oc_t = gather_theta_buf(conic, mu2d, color_o, o_col, idx_u, valid6, vf, origins_t, bump_buf, T, K)
            return render_fwd(Phi, theta, col_t, oc_t, wb_buf, bias)
        C_ref = dn(rend_fwd()).reshape(T * 256, 3)         # untraced ref (warmup too)
        img_ref = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, C_ref)
        ttnn.synchronize_device(DEV)

        # ---- (C) capture both traces back-to-back; NO allocation/host-write after this ----
        gfid = ttnn.begin_trace_capture(DEV, cq_id=0)
        conic, mu2d, keep, color_o, o_col = geom_fwd()
        ttnn.end_trace_capture(DEV, gfid, cq_id=0)
        rfid = ttnn.begin_trace_capture(DEV, cq_id=0)
        C = rend_fwd()
        ttnn.end_trace_capture(DEV, rfid, cq_id=0); ttnn.synchronize_device(DEV)

        # ---- (D) traced forward: replay geom -> (rebin via copies) -> replay rend ----
        ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        idx2, valid2 = assign_bins(dn(mu2d), dn(keep).reshape(-1) > 0.5, tmap, 1, K)
        setbuf(idx_u, idx2.to(torch.int32).reshape(T, K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        setbuf(valid6, valid2[:, None, :].expand(T, 6, K).float())
        setbuf(vf, valid2[..., None].float())
        ttnn.execute_trace(DEV, rfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        C_traced = dn(C).reshape(T * 256, 3)
        img_traced = torch.zeros(cam.H * cam.W, 3).index_copy(0, tmap.gidx, C_traced)

        finite = bool(torch.isfinite(img_traced).all())
        rel = ((img_traced - img_ref).norm() / img_ref.norm().clamp(min=1e-9)).item()
        print("== traced FORWARD (geom-fwd + gather+render-fwd traces) ==")
        print(f"   image finite: {finite}   rel vs untraced: {rel:.4e}")
        print(f"   -> {'PASS (no hazard)' if finite and rel < 0.02 else 'FAIL/corrupted'}")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
