"""Where does partial-BH's ~45 ms/view actually go? Per-stage timing of the render_binned_device path so
the next perf lever is data-driven (trace gave 1.07x, binning-vec 1.1x -> suspect the cost is spread).

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_partial_breakdown.py
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

from spike import data, forward, geometry, metrics
from spike.model import GaussianModel
import m4_train_binned as mtb
from m4_train_binned import TileMap, assign_bins, theta_u_gathered, _DevRenderBinned, K_POLY


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        mtb._DEV, mtb.CG, mtb.CKC = dev, ttnn.CoreGrid(x=11, y=10), None
        res, G, K, R = 128, 8000, 256, 1
        cams, imgs = data.load_blender("data/nerf_synthetic/ficus", "train", res=res, n=1)
        cam, gt = cams[0], imgs[0]
        tmap = TileMap(res, res)
        torch.manual_seed(0)
        m = GaussianModel(G, extent=1.5, seed=0)
        for p in m.parameters():
            p.requires_grad_(True)

        def stages(timed):
            t = {}
            t0 = time.perf_counter()
            Rm = geometry.quat_to_rotmat(m.quats)
            cov = geometry.cov3d(torch.exp(m.log_scales), Rm)
            mu2d, conic, depth, keep = geometry.project_ewa(
                m.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, 0.3, 0.2)
            color = forward.color_from_dc(m.color_dc)
            o = torch.sigmoid(m.opacity_raw); w_b = F.softplus(m.w_b_raw)
            keo = (keep.float() * o)[:, None]
            color_o, o_col = keo * color, keo
            t["geom"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            idx, valid = assign_bins(mu2d.detach(), keep.detach(), tmap, R, K)
            t["bin"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            vf = valid[..., None].float()
            conic_t, mu_t = conic[idx], mu2d[idx]
            theta_u = theta_u_gathered(conic_t, mu_t, tmap.origins, K_POLY) * valid[:, None, :].float()
            color_o_t, o_col_t = color_o[idx] * vf, o_col[idx] * vf
            t["gather"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            img = _DevRenderBinned.apply(theta_u, color_o_t, o_col_t, w_b, tmap.Phi, tmap.gidx, m.c_b, cam.H, cam.W)
            ttnn.synchronize_device(dev)
            t["render_fwd"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            loss = metrics.loss_fn(img, gt, lambda_ssim=0.2)
            loss.backward()
            ttnn.synchronize_device(dev)
            t["bwd(host autograd + device render bwd)"] = time.perf_counter() - t0
            return t

        for _ in range(3):
            for p in m.parameters():
                p.grad = None
            stages(False)
        N = 10
        acc = {}
        for _ in range(N):
            for p in m.parameters():
                p.grad = None
            t = stages(True)
            for k, v in t.items():
                acc[k] = acc.get(k, 0.0) + v
        print(f"== partial-BH per-view stage breakdown (res={res} G={G} K={K}) ==")
        tot = 0.0
        for k in ("geom", "bin", "gather", "render_fwd", "bwd(host autograd + device render bwd)"):
            ms = acc[k] / N * 1e3
            tot += ms
            print(f"   {k:42s} {ms:7.2f} ms")
        print(f"   {'TOTAL (fwd+bwd, 1 view)':42s} {tot:7.2f} ms")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
