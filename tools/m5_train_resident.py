"""The FULL-BH train loop. Wires the verified bricks into one trainer where the P*G hot
path AND the per-tile operand build run on the Blackhole:
  host  : geometry (G-setup, torch autograd -- cheap, "anywhere") + binning (branchy) + Adam
  device: gather + theta_u build [DevGather] -> render fwd + bwd (m4_train_binned._DevRenderBinned,
          silicon-verified) ; DevGather backward = theta_u-build bwd + scatter-add.
The autograd boundary is the geometry outputs (conic, mu2d, color_o, o_col); host autograd carries grads
through the (cheap) geometry to the params. Oracle: same held-out PSNR as the partial-BH trainer
(tools/m4_train_binned.py, render-only-on-device). Use --compare to time/score both at one config.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_train_resident.py --compare --res 128 --G 8000 --K 256
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import ttnn

from spike import data, forward, geometry, metrics, plyio
from spike.model import GaussianModel
import m4_train_binned as mtb
from m4_train_binned import (TileMap, assign_bins, _DevRenderBinned, K_POLY, up, dn)
from m5_theta_bwd import theta_build_bwd_dev

DEV = None
DT = ttnn.bfloat16


class DevGather(torch.autograd.Function):
    """conic,mu2d,color_o,o_col + (idx,valid,origins) -> theta_u[T,6,K], color_o_t[T,K,3], o_col_t[T,K,1].
    Forward = device gather (embedding) + theta_u build. Backward = theta_u-build bwd +
    host scatter-add of the per-tile grads back to the per-gaussian tensors."""
    @staticmethod
    def forward(ctx, conic, mu2d, color_o, o_col, idx, valid, origins, K):
        T = idx.shape[0]

        def emb(tab):
            w = up(tab.detach())
            ii = ttnn.from_torch(idx.to(torch.int32).reshape(T, K), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
            return ttnn.embedding(ii, w)
        conic_t, mu_t = emb(conic), emb(mu2d)
        color_t, ocol_t = emb(color_o), emb(o_col)
        org = up(origins[:, None, :].expand(T, K, 2).contiguous())
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
        theta = ttnn.transpose(ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), up(bump)), 1, 2)
        theta = ttnn.mul(theta, up(valid[:, None, :].expand(T, 6, K).float().contiguous()))
        vf = up(valid[..., None].float())
        color_o_t, o_col_t = ttnn.mul(color_t, vf), ttnn.mul(ocol_t, vf)
        ctx.conic, ctx.mu2d, ctx.idx, ctx.valid, ctx.origins, ctx.K = (
            conic.detach(), mu2d.detach(), idx, valid, origins, K)
        ctx.G = conic.shape[0]
        return dn(theta), dn(color_o_t), dn(o_col_t)

    @staticmethod
    def backward(ctx, gtheta_u, gcolor_o_t, go_col_t):
        idx, valid, K, G = ctx.idx, ctx.valid, ctx.K, ctx.G
        conic_t = ctx.conic[idx]                                       # [T,K,3] host re-gather (cheap)
        mu_t = ctx.mu2d[idx]                                           # [T,K,2]
        gconic_t, gmu_t = theta_build_bwd_dev(DEV, DT, conic_t, mu_t, ctx.origins, valid, gtheta_u)
        vf = valid[..., None].float()
        flat = idx.reshape(-1)
        gconic = torch.zeros(G, 3).index_add_(0, flat, gconic_t.reshape(-1, 3))
        gmu2d = torch.zeros(G, 2).index_add_(0, flat, gmu_t.reshape(-1, 2))
        gcolor_o = torch.zeros(G, 3).index_add_(0, flat, (gcolor_o_t * vf).reshape(-1, 3))
        go_col = torch.zeros(G, 1).index_add_(0, flat, (go_col_t * vf).reshape(-1, 1))
        return gconic, gmu2d, gcolor_o, go_col, None, None, None, None


def host_geom(model, cam):
    Rm = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), Rm)
    mu2d, conic, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, 0.3, 0.2)
    color = forward.color_from_dc(model.color_dc)
    o = torch.sigmoid(model.opacity_raw)
    w_b = F.softplus(model.w_b_raw)
    keo = (keep.float() * o)[:, None]
    return mu2d, conic, keo * color, keo, keep, w_b


def render_fullbh(model, cam, tmap, R=1, K=128):
    mu2d, conic, color_o, o_col, keep, w_b = host_geom(model, cam)
    idx, valid = assign_bins(mu2d.detach(), keep.detach(), tmap, R, K)
    theta_u, color_o_t, o_col_t = DevGather.apply(conic, mu2d, color_o, o_col, idx, valid, tmap.origins, K)
    img = _DevRenderBinned.apply(theta_u, color_o_t, o_col_t, w_b, tmap.Phi, tmap.gidx, model.c_b,
                                 cam.H, cam.W)
    return img.reshape(cam.H, cam.W, 3)


def fit_and_eval(render_fn, label, args):
    tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, stride=max(1, 100 // args.n_train))
    te_c, te_i = data.load_blender(args.scene, "test", res=args.res, n=args.n_test, stride=max(1, 200 // args.n_test))
    tmap = TileMap(args.res, args.res)
    torch.manual_seed(args.seed)
    m = GaussianModel(args.G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    opt = torch.optim.Adam(m.param_groups(DEFAULT_LR))
    t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for cam, gt in zip(tr_c, tr_i):
            tot = tot + metrics.loss_fn(render_fn(m, cam, tmap, 1, args.K), gt, lambda_ssim=0.2)
        (tot / len(tr_c)).backward()
        opt.step()
    tt = time.perf_counter() - t0
    with torch.no_grad():
        tr = sum(float(metrics.psnr(render_fn(m, c, tmap, 1, args.K), g)) for c, g in zip(tr_c, tr_i)) / len(tr_c)
        te = sum(float(metrics.psnr(render_fn(m, c, tmap, 1, args.K), g)) for c, g in zip(te_c, te_i)) / len(te_c)
    print(f"   [{label:9s}] {tt/args.iters*1e3:7.0f} ms/it | train {tr:.2f} dB | test {te:.2f} dB")
    return tt / args.iters * 1e3, tr, te


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--G", type=int, default=8000)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="run partial-BH and full-BH at this config")
    args = ap.parse_args()
    DEV = ttnn.open_device(device_id=0)
    try:
        mtb._DEV, mtb.CG, mtb.CKC = DEV, ttnn.CoreGrid(x=11, y=10), None
        print(f"== full-BH train loop (res={args.res} G={args.G} K={args.K} "
              f"{args.n_train}tr/{args.n_test}te {args.iters}it) ==")
        full = fit_and_eval(render_fullbh, "full-BH", args)
        if args.compare:
            from m4_train_binned import render_binned_device
            part = fit_and_eval(render_binned_device, "partial-BH", args)
            print(f"\n   speed: full-BH {full[0]:.0f} vs partial-BH {part[0]:.0f} ms/it "
                  f"({part[0]/full[0]:.2f}x) | held-out {full[2]:.2f} vs {part[2]:.2f} dB")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
