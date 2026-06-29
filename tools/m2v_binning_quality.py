"""Binning de-risk (no hardware): does spatial-hash binning preserve quality vs the dense render?

The binning model: 16x16 tiles; each gaussian -> its center tile; a tile gathers gaussians from
its (2R+1)^2 tile stencil; a fixed per-tile budget K drops overflow (kept = K nearest tile-center).
Each tile renders its 16x16 pixel block from only its <=K gaussians. The question this answers:
for what (R, K) does binned == dense (so the "N-body neighbor search" approximation is lossless),
and how big must K be (does it stay SRAM-friendly)? Perf is NOT measured here (that needs silicon).

Run on the GPU box (native venv or the cuda container):
    .venv/bin/python tools/m2v_binning_quality.py --res 128 --G 2000 --iters 1500
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from spike import arms, data, forward, geometry, metrics, train
from spike.device import default_device
from spike.model import GaussianModel

K_INF = 10 ** 9


def _geom(model, cam, blur_eps=0.3, near=0.2):
    R = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), R)
    mu2d, conic, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, blur_eps, near)
    return mu2d, conic, keep


@torch.no_grad()
def render_binned(model, cam, R, K, k=4.0):
    """Tiled render: each 16x16 tile uses only gaussians in its (2R+1)^2 stencil, capped at K."""
    dev, dt = model.means3d.device, model.means3d.dtype
    mu2d, conic, keep = _geom(model, cam)
    color = forward.color_from_dc(model.color_dc)
    w_b, c_b = F.softplus(model.w_b_raw), model.c_b
    H, W = cam.H, cam.W
    nty, ntx = H // 16, W // 16
    gx = (mu2d[:, 0] / 16).floor().long().clamp(0, ntx - 1)
    gy = (mu2d[:, 1] / 16).floor().long().clamp(0, nty - 1)

    img = torch.empty(H, W, 3, device=dev, dtype=dt)
    max_stencil = 0
    for ty in range(nty):
        for tx in range(ntx):
            in_stencil = keep & (gx >= tx - R) & (gx <= tx + R) & (gy >= ty - R) & (gy <= ty + R)
            idx = in_stencil.nonzero(as_tuple=True)[0]
            max_stencil = max(max_stencil, int(idx.numel()))
            tcx, tcy = (tx + 0.5) * 16.0, (ty + 0.5) * 16.0
            if idx.numel() > K:                              # overflow -> keep K nearest tile center
                d = (mu2d[idx, 0] - tcx) ** 2 + (mu2d[idx, 1] - tcy) ** 2
                idx = idx[torch.topk(d, K, largest=False).indices]
            rows = torch.arange(ty * 16, ty * 16 + 16, device=dev, dtype=dt)
            cols = torch.arange(tx * 16, tx * 16 + 16, device=dev, dtype=dt)
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")
            pxb, pyb = (cc + 0.5).reshape(-1), (rr + 0.5).reshape(-1)         # [256]
            if idx.numel() == 0:
                block = c_b[None, :].expand(256, 3)                          # background only
            else:
                Q = forward.quad_form(pxb, pyb, mu2d[idx], conic[idx])        # [256,n]
                w_geo = forward.poly_splat_wgeo(Q, k, keep[idx])
                block = arms.blend_A(w_geo, model.opacity_raw[idx], color[idx], w_b, c_b)
            img[ty * 16:ty * 16 + 16, tx * 16:tx * 16 + 16, :] = block.reshape(16, 16, 3)
    return img, max_stencil


def _spread(n_total, n):
    return max(1, n_total // max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    assert args.res % 16 == 0, "res must be a multiple of 16"

    dev = default_device(args.device)
    torch.manual_seed(0)
    tr_cams, tr_imgs = data.load_blender(args.scene, "train", res=args.res, device=dev, n=8, stride=_spread(100, 8))
    ho_cams, ho_imgs = data.load_blender(args.scene, "val", res=args.res, device=dev, n=2, stride=_spread(100, 2))
    dname = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"[binning de-risk] device={dev} [{dname}] | res={args.res} G={args.G} | training arm A {args.iters} it...")

    model = GaussianModel(args.G, extent=1.5, seed=0, device=dev)
    train.fit(model, tr_cams, tr_imgs, "A", iters=args.iters)
    dense = sum(float(metrics.psnr(train.render(model, c, "A"), gt)) for c, gt in zip(ho_cams, ho_imgs)) / len(ho_cams)
    print(f"  dense held-out PSNR = {dense:.2f} dB  ({args.res//16}x{args.res//16} tiles, ~{args.G/((args.res//16)**2):.0f} gaussians/tile avg)\n")

    print(f"  {'R':>2} {'K':>6} {'binned_PSNR':>12} {'vs_dense':>9} {'maxStencil':>11}")
    for R in (1, 2):
        for K in (32, 64, 128, K_INF):
            ps, vsd, mx = [], [], 0
            for c, gt in zip(ho_cams, ho_imgs):
                b, m = render_binned(model, c, R, K)
                mx = max(mx, m)
                ps.append(float(metrics.psnr(b, gt)))
                vsd.append(float(metrics.psnr(b, train.render(model, c, "A"))))
            klab = "all" if K == K_INF else str(K)
            print(f"  {R:>2} {klab:>6} {sum(ps)/len(ps):>12.2f} {sum(vsd)/len(vsd):>9.2f} {mx:>11}")
    print("\n  vs_dense >> dense => binned matches the dense render; pick the smallest (R,K) that saturates.")


if __name__ == "__main__":
    main()
