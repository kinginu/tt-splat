"""ttnn-trace the resident device forward. Profiling showed the device path is dispatch-bound (geometry
21 ms + gather/render 15 ms = ~36 ms of ~80+ small host-dispatched ops, with negligible compute). Trace
captures the op sequence once and replays it with one host call -> dispatch collapses. Host binning
(10.8 ms python) stays outside the trace (it is host). For a fixed view the binning indices are
constant, so geometry+gather+render capture as ONE trace over persistent inputs.

Oracle: tools/m4_train_binned.render_binned_device (image rel). Metric: traced vs untraced device
ms/view -> the dispatch-collapse factor.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_trace_forward.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import ttnn

from spike import data, forward, geometry
from spike.model import GaussianModel
import m5_resident_forward as rf
import m4_train_binned as mtb
from m4_train_binned import TileMap, assign_bins, render_binned_device, K_POLY

NEAR = rf.NEAR


def gather_render_persistent(conic, mu2d, color_o_d, o_col_d, idx_d, org_d, valid6_d, vf_d,
                             bump_d, Phi_t, bias, w_b, T, K):
    """gather + theta_u build + render -- only persistent device inputs (trace-safe)."""
    conic_t, mu_t = ttnn.embedding(idx_d, conic), ttnn.embedding(idx_d, mu2d)
    color_t, ocol_t = ttnn.embedding(idx_d, color_o_d), ttnn.embedding(idx_d, o_col_d)
    mu = ttnn.add(mu_t, ttnn.neg(org_d))
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    mux, muy = mu[:, :, 0:1], mu[:, :, 1:2]
    mux2, muy2, muxy = ttnn.mul(mux, mux), ttnn.mul(muy, muy), ttnn.mul(mux, muy)
    t2 = ttnn.mul(b, 2.0)
    t3 = ttnn.mul(ttnn.add(ttnn.mul(a, mux), ttnn.mul(b, muy)), -2.0)
    t4 = ttnn.mul(ttnn.add(ttnn.mul(b, mux), ttnn.mul(c, muy)), -2.0)
    t5 = ttnn.add(ttnn.add(ttnn.mul(a, mux2), ttnn.mul(ttnn.mul(b, muxy), 2.0)), ttnn.mul(c, muy2))
    tQ = ttnn.concat([a, c, t2, t3, t4, t5], dim=-1)
    theta = ttnn.transpose(ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), bump_d), 1, 2)
    theta = ttnn.mul(theta, valid6_d)
    color_t, ocol_t = ttnn.mul(color_t, vf_d), ttnn.mul(ocol_t, vf_d)
    return rf.render_fwd(Phi_t, theta, color_t, ocol_t, w_b, bias)


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=128 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        rf.CG = CG
        mtb._DEV, mtb.CG, mtb.CKC = dev, CG, None
        res, G, K = 128, 8000, 256
        cams, _ = data.load_blender("data/nerf_synthetic/ficus", "train", res=res, n=1)
        cam = cams[0]
        tmap = TileMap(res, res)
        torch.manual_seed(0)
        m = GaussianModel(G, extent=1.5, seed=0)

        cols = [rf.u(c.reshape(-1, 1), dev) for c in
                (m.means3d[:, 0], m.means3d[:, 1], m.means3d[:, 2],
                 m.quats[:, 0], m.quats[:, 1], m.quats[:, 2], m.quats[:, 3])]
        scales = torch.exp(m.log_scales).detach()
        cols += [rf.u(scales[:, i].reshape(-1, 1), dev) for i in range(3)]
        Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
        tv = [float(cam.t_v[i]) for i in range(3)]

        # one untraced pass -> binning indices + per-gaussian color/opacity (host, fixed view)
        conic0, mu2d0, mcz0 = rf.geom_resident(dev, cols, Rv, tv, cam.fx, cam.fy, cam.cx, cam.cy)
        mu2d_h = ttnn.to_torch(mu2d0).float()
        keep = ttnn.to_torch(mcz0).float().reshape(-1) > NEAR
        o = torch.sigmoid(m.opacity_raw).detach()
        color = forward.color_from_dc(m.color_dc).detach()
        keo = (keep.float() * o)[:, None]
        color_o, o_col = (keo * color), keo
        idx, valid = assign_bins(mu2d_h, keep, tmap, rf.R_STENCIL, K)
        T = tmap.T

        # persistent device inputs for the traced region
        idx_d = ttnn.from_torch(idx.to(torch.int32).reshape(T, K), dtype=ttnn.uint32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        color_o_d, o_col_d = rf.u(color_o, dev), rf.u(o_col, dev)
        org_d = rf.u(tmap.origins[:, None, :].expand(T, K, 2).contiguous(), dev)
        bump = torch.zeros(T, K, 6); bump[..., 5] = 1.0
        bump_d = rf.u(bump, dev)
        valid6_d = rf.u(valid[:, None, :].expand(T, 6, K).float().contiguous(), dev)
        vf_d = rf.u(valid[..., None].float(), dev)
        Phi_t = rf.u(tmap.Phi.unsqueeze(0).expand(T, 256, 6).contiguous(), dev)
        w_b = F.softplus(m.w_b_raw).detach()
        bias = rf.u((w_b * m.c_b)[None, None, :].expand(T, 256, 3).contiguous(), dev)

        def device_fwd():
            conic, mu2d, _ = rf.geom_resident(dev, cols, Rv, tv, cam.fx, cam.fy, cam.cx, cam.cy)
            return gather_render_persistent(conic, mu2d, color_o_d, o_col_d, idx_d, org_d,
                                            valid6_d, vf_d, bump_d, Phi_t, w_b=w_b, bias=bias, T=T, K=K)

        print("== ttnn-trace of the resident device forward ==")
        ref = render_binned_device(m, cam, tmap, R=rf.R_STENCIL, K=K)
        C = device_fwd()
        img = torch.zeros(cam.H * cam.W, 3)
        img[tmap.gidx] = ttnn.to_torch(C).float().reshape(T * 256, 3)
        rel = ((img.reshape(cam.H, cam.W, 3) - ref).norm() / ref.norm()).item()
        print(f"   image rel {rel:.4f}  ({'PASS' if rel < 0.05 else 'see note'})")

        # untraced device timing
        N = 12
        for _ in range(3):
            device_fwd()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(N):
            device_fwd()
        ttnn.synchronize_device(dev)
        t_un = (time.perf_counter() - t0) / N

        # capture + replay trace
        device_fwd()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        Ctr = device_fwd()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t_tr = (time.perf_counter() - t0) / N
        ttnn.release_trace(dev, tid)

        print(f"\n   device forward UNTRACED : {t_un*1e3:7.2f} ms/view")
        print(f"   device forward TRACED   : {t_tr*1e3:7.2f} ms/view   ({t_un/t_tr:.1f}x dispatch collapse)")
        print(f"   (host binning ~10.8 ms stays outside the trace -> vectorize separately)")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
