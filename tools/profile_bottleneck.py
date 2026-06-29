"""Bottleneck profiler: per-stage timing of the HYBRID train step (train_step_manual) at the
high-res config (res=800). Confirms WHERE the ~host-bound per-iter time goes before running the
full sweep, so the fix targets the real limiter (host compute vs host<->device transfers).

Stages timed (= the 6 calls in train_manual.train_step_manual):
  1 geom+gather  fwd_manual         host (binning runs only on bins=None iters -> split reported)
  2 render fwd   FastRender.forward device  (further split: copy-in / exec+sync / copy-out)
  3 loss+bwd     host autograd on the image leaf
  4 render bwd   FastRender.backward device (further split: copy-in / exec+sync / copy-out)
  5 geom bwd     bwd_manual         host scatter + geometry backward
  6 adam         opt.step           host

Single fixed view (stochastic 1/iter in the sweep; the per-iter view does not change the breakdown).
bins recomputed every --bin-every iters (the shipped default 5); bin-iters vs reuse-iters reported apart.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/profile_bottleneck.py --res 800 --G 1000
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
from spike.train import DEFAULT_LR
import m4_train_binned as mtb
from m4_train_binned import TileMap, dn
import faststep as fs
from operands_manual import fwd_manual, bwd_manual


def _t():
    return time.perf_counter()


def profiled_step(model, cams, gts, tmap, K, opt, fr, bins, acc):
    """Replicates train_step_manual with per-stage timers. acc = dict of accumulators (ms)."""
    T = tmap.T
    DEV = fs.DEV

    # ---- 1. geom + gather (host) ; binning inside only when bins is None ----
    t = _t()
    with torch.no_grad():
        theta_all, col_all, oc_all, cache = fwd_manual(model, cams, tmap, 1, K, bins=bins)
        w_b = F.softplus(model.w_b_raw)
        B = theta_all.shape[0]
        bias_all = (w_b * model.c_b)[None, None, :].expand(B, 256, 3).contiguous()
        wb_all = w_b.reshape(1, 1, 1).expand(B, 256, 1).contiguous()
    acc["1_geom_gather"] += _t() - t

    # ---- 2. render fwd (device) split copy-in / exec / copy-out ----
    t = _t()
    fr._set(fr.thU, theta_all); fr._set(fr.col, col_all); fr._set(fr.oc, oc_all)
    fr._set(fr.bias, bias_all); fr._set(fr.wb, wb_all)
    acc["2a_fwd_copyin"] += _t() - t
    t = _t()
    ttnn.execute_trace(DEV, fr.fid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
    acc["2b_fwd_exec"] += _t() - t
    t = _t()
    C_all = dn(fr.C)
    acc["2c_fwd_copyout"] += _t() - t

    # ---- 3. loss + backward on the host image leaf ----
    t = _t()
    C_leaf = C_all.clone().requires_grad_(True)
    loss = 0.0
    for v, gt in enumerate(gts):
        H, W = gt.shape[0], gt.shape[1]
        img = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, C_leaf[v * T:(v + 1) * T].reshape(T * 256, 3))
        loss = loss + metrics.loss_fn(img.reshape(H, W, 3), gt, lambda_ssim=0.2)
    loss = loss / len(gts)
    loss.backward()
    gC_all = C_leaf.grad
    acc["3_loss_bwd"] += _t() - t

    # ---- 4. render bwd (device) split ----
    t = _t()
    fr._set(fr.gC, gC_all)
    acc["4a_bwd_copyin"] += _t() - t
    t = _t()
    ttnn.execute_trace(DEV, fr.bid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
    acc["4b_bwd_exec"] += _t() - t
    t = _t()
    gthU, gcol, goc, gnum, gden = [dn(x) for x in fr.bouts]
    acc["4c_bwd_copyout"] += _t() - t

    # ---- 5. geom bwd (host scatter + geometry backward) ----
    t = _t()
    with torch.no_grad():
        grads = bwd_manual(cache, gthU, gcol, goc)
        gw_b = (gnum * model.c_b[None, None, :]).sum() + gden.sum()
        grads["w_b_raw"] = gw_b * torch.sigmoid(model.w_b_raw)
        for n, p in model.named_parameters():
            p.grad = grads.get(n, None)
    acc["5_geom_bwd"] += _t() - t

    # ---- 6. adam ----
    t = _t()
    opt.step()
    acc["6_adam"] += _t() - t

    bins_used = [(p["idx"], p["valid"]) for p in cache["pv"]]
    return float(loss), bins_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--G", type=int, default=1000)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--bin-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fs.DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    fs.CG = ttnn.CoreGrid(x=11, y=10)
    mtb._DEV, mtb.CG, mtb.CKC = fs.DEV, fs.CG, None
    try:
        tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=2, stride=50)
        cams, gts = [tr_c[0]], [tr_i[0]]
        tmap = TileMap(args.res, args.res)
        T = tmap.T
        torch.manual_seed(args.seed)
        model = GaussianModel(args.G, extent=1.5, seed=args.seed)
        for p in model.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(model.param_groups(DEFAULT_LR))

        print(f"== bottleneck profile  res={args.res} G={args.G} K={args.K} T={T} tiles "
              f"bin_every={args.bin_every} ==", flush=True)

        # warm up (JIT + FastRender capture)
        tw = _t()
        from train_manual import train_step_manual
        _, bins = train_step_manual(model, cams, gts, tmap, args.K, opt, None)
        fr = fs._FR[(len(cams) * T, args.K)]
        print(f"   warm-up (JIT+capture): {_t()-tw:.0f}s", flush=True)

        bin_acc = {k: 0.0 for k in ["1_geom_gather", "2a_fwd_copyin", "2b_fwd_exec", "2c_fwd_copyout",
                                    "3_loss_bwd", "4a_bwd_copyin", "4b_bwd_exec", "4c_bwd_copyout",
                                    "5_geom_bwd", "6_adam"]}
        reuse_acc = {k: 0.0 for k in bin_acc}
        n_bin = n_reuse = 0
        t0 = _t()
        for it in range(args.iters):
            is_bin = (it % args.bin_every == 0)
            use = None if is_bin else bins
            acc = bin_acc if is_bin else reuse_acc
            _, bins = profiled_step(model, cams, gts, tmap, args.K, opt, fr, use, acc)
            if is_bin:
                n_bin += 1
            else:
                n_reuse += 1
        wall = _t() - t0

        def report(name, acc, n):
            if n == 0:
                return
            tot = sum(acc.values()) / n
            print(f"\n   --- {name} (avg over {n} iters): {tot*1e3:.1f} ms/it, {1/tot:.2f} it/s ---")
            for k in sorted(acc):
                ms = acc[k] / n * 1e3
                print(f"      {k:18s} {ms:7.2f} ms  ({ms/(tot*1e3)*100:4.1f}%)")

        report("BIN iters (binning recomputed)", bin_acc, n_bin)
        report("REUSE iters (bins cached)", reuse_acc, n_reuse)
        amort = wall / args.iters
        print(f"\n   === AMORTIZED over {args.iters} iters (bin_every={args.bin_every}): "
              f"{amort*1e3:.1f} ms/it = {1/amort:.2f} it/s ===")
    finally:
        ttnn.close_device(fs.DEV)


if __name__ == "__main__":
    main()
