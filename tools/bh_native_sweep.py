"""BH native sweep: route-B@Blackhole bf16+binned stochastic training — m6/FastRender path.

Config: ficus+lego × G{30k,100k,300k} / res800 / 30k iters / 100 train views stochastic(1/iter).
Uses m6 train_step_manual (FastRender traced render, manual operands/backward) — eliminates the
per-iter from_torch overhead that made m4 slow at res800 (Phi resident on device, execute_trace
instead of re-allocating device tensors each iter).

K auto-scales with G: {G≤30k→128, G≤100k→256, G>100k→512}  (keeps per-tile dropout near-zero).
Per-view bins cache (BIN_REFRESH=500) amortises assign_bins across iters.
Results JSON keys match gsplat_native_sweep.py for direct comparison.
.ply saved per config; NOT pushed (outputs/bh_sweep/ is gitignored).

Run inside the hw container:
    # smoke (1 config, 50 iters, no .ply)
    podman-compose --profile hw run --rm hw python3 tools/bh_native_sweep.py \
        --scenes ficus --G 30000 --iters 50 --no-ply
    # full sweep
    podman-compose --profile hw run --rm hw python3 tools/bh_native_sweep.py
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import ttnn

from spike import data, metrics, plyio, evalcard
from spike.model import GaussianModel
from spike.render import render as cpu_render
from spike.train import DEFAULT_LR
import m4_train_binned as mtb
from m4_train_binned import TileMap, render_binned_device
import faststep as fs
from train_manual import train_step_manual


def _auto_k(G):
    if G <= 30_000:
        return 128
    if G <= 100_000:
        return 256
    return 512


def _save_img(path, img):
    arr = (img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_panel(path, gt, render):
    """Save GT (left) | render (right) side-by-side panel as uint8 PNG."""
    gt_arr = (gt.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
    rend_arr = (render.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
    panel = np.concatenate([gt_arr, rend_arr], axis=1)
    Image.fromarray(panel).save(path)


@torch.no_grad()
def _render_fps(model, cam, tmap, K, n=30):
    render_binned_device(model, cam, tmap, R=1, K=K)
    ttnn.synchronize_device(mtb._DEV)
    t0 = time.perf_counter()
    for _ in range(n):
        render_binned_device(model, cam, tmap, R=1, K=K)
    ttnn.synchronize_device(mtb._DEV)
    return n / (time.perf_counter() - t0)


@torch.no_grad()
def _eval_split(model, cams, imgs, tmap, K):
    ps, ss = [], []
    for cam, gt in zip(cams, imgs):
        r = render_binned_device(model, cam, tmap, R=1, K=K)
        ps.append(float(metrics.psnr(r, gt)))
        ss.append(float(metrics.ssim(r, gt)))
    return sum(ps) / len(ps), sum(ss) / len(ss)


def run_one(scene_root, scene_name, G, K, res, iters, paths, bin_refresh, save_ply_file, save_views, method):
    out_dir = paths["dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[bh] {scene_name} G={G} K={K} res={res} iters={iters}")

    tr_c, tr_i = data.load_blender(scene_root, "train", res=res)
    te_c, te_i = data.load_blender(scene_root, "test",  res=res)
    te_ca, te_rgba = data.load_blender(scene_root, "test", res=res, keep_alpha=True)
    N_train = len(tr_c)
    print(f"[bh]   {N_train} train views, {len(te_c)} test views", flush=True)

    tmap = TileMap(res, res)
    torch.manual_seed(0)
    model = GaussianModel(G, extent=1.5, seed=0)
    # manual-grad path: requires_grad=False; Adam still updates via p.grad set in train_step_manual
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.param_groups(DEFAULT_LR))

    # Warm up: JIT compile + FastRender trace capture (off-clock, single view)
    print("[bh]   warming up JIT + trace capture ...", flush=True)
    t_warm = time.perf_counter()
    train_step_manual(model, [tr_c[0]], [tr_i[0]], tmap, K, opt, bins=None)
    print(f"[bh]   warm-up done in {time.perf_counter()-t_warm:.0f}s", flush=True)

    # Per-view bins cache  {vi: (idx, valid)},  bins_age {vi: iter of last recompute}
    bins_cache: dict = {}
    bins_age:   dict = {}

    t0 = time.perf_counter()
    for it in range(iters):
        vi = int(torch.randint(N_train, (1,)).item())

        age = it - bins_age.get(vi, -bin_refresh)
        use_bins = [bins_cache[vi]] if vi in bins_cache and age < bin_refresh else None

        loss_val, bins_used = train_step_manual(
            model, [tr_c[vi]], [tr_i[vi]], tmap, K, opt, use_bins)

        bins_cache[vi] = bins_used[0]
        if use_bins is None:
            bins_age[vi] = it

        if it % max(1, iters // 20) == 0 or it == iters - 1:
            elapsed = time.perf_counter() - t0
            print(f"    iter {it:5d}/{iters}  loss {loss_val:.4f}  "
                  f"{(it + 1) / elapsed:.2f} it/s", flush=True)

    train_s = time.perf_counter() - t0
    it_per_s = round(iters / train_s, 2)

    # Eval: full test split + first 10 train views
    with torch.no_grad():
        ho_p, ho_s = _eval_split(model, te_c, te_i, tmap, K)
        tr_p, tr_s = _eval_split(model, tr_c[:10], tr_i[:10], tmap, K)
        fps = _render_fps(model, te_c[0], tmap, K)

    ply_path = ""
    if save_ply_file:
        ply_path = paths["ply"]
        plyio.save_ply(ply_path, model)

    # unified eval card (host cpu_render oracle with c_b swapped for the two-bg coverage trick); guarded
    perc = {}
    try:
        def rfn(mdl, cam, b):
            saved = mdl.c_b.detach().clone()
            mdl.c_b.fill_(b)
            try:
                return cpu_render(mdl, cam, "A")
            finally:
                mdl.c_b.copy_(saved)
        card = evalcard.build(model, te_ca, te_rgba, rfn, method=method,
                              scene=scene_name, G=G, res=res, iters=iters, K=K, train_psnr=tr_p,
                              perf={"it_per_s": it_per_s, "render_fps": round(fps, 1),
                                    "train_s": round(train_s, 1), "peak_mem_gb": None,
                                    "device": "blackhole-bf16-binned-m6"})
        card["holdout"] = {"psnr": round(ho_p, 2), "ssim": round(ho_s, 4)}
        evalcard.save(card, paths["eval_json"])
        perc = card.get("perceptual", {})
        print(f"[bh]   eval card -> {paths['eval_json']}", flush=True)
    except Exception as e:
        print(f"[bh]   WARN eval card skipped: {e}", flush=True)

    # Representative-view panels (4 held-out views, full run only)
    renders_dir = None
    if save_views:
        renders_dir = paths["stem"] + "_renders"
        os.makedirs(renders_dir, exist_ok=True)
        with torch.no_grad():
            for i, (c, g) in enumerate(zip(te_c[:4], te_i[:4])):
                r = render_binned_device(model, c, tmap, R=1, K=K)
                _save_panel(os.path.join(renders_dir, f"view{i:02d}_panel.png"), g, r)
        print(f"[bh]   panels -> {renders_dir}", flush=True)

    row = {
        "method": method,
        "scene": scene_name, "G": G, "res": res, "K": K, "iters": iters,
        "train_views": N_train,
        "holdout_psnr": round(ho_p, 2), "holdout_ssim": round(ho_s, 4),
        "train_psnr": round(tr_p, 2), "train_ssim": round(tr_s, 4),
        "hf_ratio": perc.get("hf_ratio"), "empty_space_leak": perc.get("empty_space_leak"),
        "it_per_s": it_per_s, "render_fps": round(fps, 1), "train_s": round(train_s, 1),
        "peak_mem_gb": None, "device": "blackhole-bf16-binned-m6",
        "ply": ply_path, "eval_json": paths["eval_json"], "renders_dir": renders_dir,
    }
    print(f"[bh] {scene_name} G={G}: {it_per_s} it/s | render {fps:.0f} fps | "
          f"holdout {ho_p:.2f} dB/{ho_s:.4f} | train {tr_p:.2f} dB | "
          f"{train_s / 60:.1f} min", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="ficus,lego")
    ap.add_argument("--G", default="1000,5000,10000")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--out", default="outputs", help="root; layout = <out>/<method>/<scene>/G<G>_res<res>")
    ap.add_argument("--method", default="routeB_bh", help="rough algo label for the unified output path")
    ap.add_argument("--bin-refresh", type=int, default=500)
    ap.add_argument("--no-ply", action="store_true")
    ap.add_argument("--no-views", action="store_true",
                    help="skip GT|render panel saves even on full runs")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    Gs = [int(g.strip()) for g in args.G.split(",") if g.strip()]
    os.makedirs(args.out, exist_ok=True)

    # Device: trace_region_size needed for FastRender trace capture
    fs.DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    fs.CG = ttnn.CoreGrid(x=11, y=10)
    mtb._DEV = fs.DEV
    mtb.CG = fs.CG
    mtb.CKC = None   # m6 path does not use HiFi4 config

    try:
        rows = []
        for scene in scenes:
            root = os.path.join("data/nerf_synthetic", scene)
            for G in Gs:
                K = _auto_k(G)
                paths = evalcard.run_paths(args.method, scene, G, args.res, iters=args.iters,
                                           K=K, root=args.out)
                save_views = not args.no_ply and not args.no_views
                row = run_one(root, scene, G, K, args.res, args.iters, paths,
                              args.bin_refresh, not args.no_ply, save_views, args.method)
                rows.append(row)
                json.dump(rows, open(os.path.join(args.out, "bh_sweep_rollup.json"), "w"), indent=2)

        with open(os.path.join(args.out, "summary.txt"), "w") as f:
            f.write(f"# {args.method} | res{args.res}/{args.iters}it\n")
            f.write(f"# {'scene':8} {'G':>7} {'K':>4} {'it/s':>7} {'fps':>6} "
                    f"{'ho_psnr':>8} {'ho_ssim':>8} {'hf':>8} {'leak':>8}\n")
            for r in rows:
                k_str = "-" if r.get("K") is None else str(r["K"])
                hf = f"{r['hf_ratio']:.4f}" if r.get("hf_ratio") is not None else "N/A"
                leak = f"{r['empty_space_leak']:.4f}" if r.get("empty_space_leak") is not None else "N/A"
                f.write(f"  {r['scene']:8} {r['G']:>7} {k_str:>4} {r['it_per_s']:>7} "
                        f"{r['render_fps']:>6.0f} {r['holdout_psnr']:>8} {r['holdout_ssim']:>8} "
                        f"{hf:>8} {leak:>8}\n")

        print("\n" + open(os.path.join(args.out, "summary.txt")).read())
        print(f"saved {len(rows)} configs -> {args.out}/")
    finally:
        ttnn.close_device(fs.DEV)


if __name__ == "__main__":
    main()
