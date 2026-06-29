"""NO-AUTOGRAD trainer. Removes the torch-autograd cost (the host cap)
entirely: operands fwd + manual bwd (operands_manual, verified) on host, render fwd+bwd via FastRender
(device, traced, batched, faststep). The ONLY autograd is a tiny graph on the per-iter image leaf to
get dL/dC (loss+SSIM grad). All param grads are set manually; opt.step() applies them.

Oracle: same held-out PSNR as tools/m4_train_binned.py. Metric: ms/it vs partial-BH.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/train_manual.py --compare --res 128 --G 8000 --K 256
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

from spike import data, metrics
from spike.model import GaussianModel
import m4_train_binned as mtb
from m4_train_binned import TileMap, render_binned_device
import faststep as fs
from operands_manual import fwd_manual, bwd_manual


def train_step_manual(model, cams, gts, tmap, K, opt, bins=None):
    T = tmap.T
    with torch.no_grad():
        theta_all, col_all, oc_all, cache = fwd_manual(model, cams, tmap, 1, K, bins=bins)
        w_b = F.softplus(model.w_b_raw)
        B = theta_all.shape[0]
        bias_all = (w_b * model.c_b)[None, None, :].expand(B, 256, 3).contiguous()
        wb_all = w_b.reshape(1, 1, 1).expand(B, 256, 1).contiguous()
    key = (B, K)
    if key not in fs._FR:
        fs._FR[key] = fs.FastRender(B, K, tmap.Phi)
    fr = fs._FR[key]
    C_all = fr.forward(theta_all, col_all, oc_all, bias_all, wb_all)        # [B,256,3] host

    C_leaf = C_all.clone().requires_grad_(True)                            # tiny autograd: loss only
    loss = 0.0
    for v, gt in enumerate(gts):
        H, W = gt.shape[0], gt.shape[1]
        img = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, C_leaf[v * T:(v + 1) * T].reshape(T * 256, 3))
        loss = loss + metrics.loss_fn(img.reshape(H, W, 3), gt, lambda_ssim=0.2)
    loss = loss / len(gts)
    loss.backward()
    gC_all = C_leaf.grad

    gthU, gcol, goc, gnum, gden = fr.backward(gC_all)
    with torch.no_grad():
        grads = bwd_manual(cache, gthU, gcol, goc)
        gw_b = (gnum * model.c_b[None, None, :]).sum() + gden.sum()
        grads["w_b_raw"] = gw_b * torch.sigmoid(model.w_b_raw)              # softplus bwd
        for n, p in model.named_parameters():
            p.grad = grads.get(n, None)
    opt.step()
    bins_used = [(p["idx"], p["valid"]) for p in cache["pv"]]
    return float(loss), bins_used


def fit_eval(use_manual, label, args):
    tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, stride=max(1, 100 // args.n_train))
    te_c, te_i = data.load_blender(args.scene, "test", res=args.res, n=args.n_test, stride=max(1, 200 // args.n_test))
    tmap = TileMap(args.res, args.res)
    torch.manual_seed(args.seed)
    m = GaussianModel(args.G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    if use_manual:
        for p in m.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(m.param_groups(DEFAULT_LR))
    bins = None
    if use_manual:
        _, bins = train_step_manual(m, tr_c, tr_i, tmap, args.K, opt, None)  # warm (JIT+capture) off-clock
    else:
        render_binned_device(m, tr_c[0], tmap, 1, args.K)
    t0 = time.perf_counter()
    for it in range(args.iters):
        if use_manual:
            use = None if (it % args.bin_every == 0) else bins             # recompute bins every N iters
            _, bins = train_step_manual(m, tr_c, tr_i, tmap, args.K, opt, use)
        else:
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for cam, gt in zip(tr_c, tr_i):
                tot = tot + metrics.loss_fn(render_binned_device(m, cam, tmap, 1, args.K), gt, lambda_ssim=0.2)
            (tot / len(tr_c)).backward(); opt.step()
    tt = time.perf_counter() - t0
    with torch.no_grad():
        te = sum(float(metrics.psnr(render_binned_device(m, c, tmap, 1, args.K), g))
                 for c, g in zip(te_c, te_i)) / len(te_c)
    print(f"   [{label:12s}] {tt/args.iters*1e3:7.0f} ms/it | held-out {te:.2f} dB")
    return tt / args.iters * 1e3, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--G", type=int, default=8000)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-every", type=int, default=1, help="recompute binning every N iters (stale bins)")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    fs.DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        fs.CG = ttnn.CoreGrid(x=11, y=10)
        mtb._DEV, mtb.CG, mtb.CKC = fs.DEV, fs.CG, None
        print(f"== NO-AUTOGRAD trainer (manual ops + FastRender) res={args.res} G={args.G} K={args.K} "
              f"{args.n_train}tr/{args.n_test}te {args.iters}it bin-every={args.bin_every} ==")
        man = fit_eval(True, "manual", args)
        if args.compare:
            pa = fit_eval(False, "partial-BH", args)
            print(f"\n   speed: manual {man[0]:.0f} vs partial-BH {pa[0]:.0f} ms/it "
                  f"({pa[0]/man[0]:.2f}x faster) | held-out {man[1]:.2f} vs {pa[1]:.2f} dB")
    finally:
        ttnn.close_device(fs.DEV)


if __name__ == "__main__":
    main()
