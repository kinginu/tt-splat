"""gsplat perf scaling at res800 across large G — the GPU side of the BH-vs-GPU crossover search.

Measures train it/s (stochastic 1-view/iter, the standard 3DGS step), render fps, and peak VRAM as a
function of G at a fixed res — PERF ONLY (no quality, no .ply). Pairs with the BH `routeB_bh` eval cards
(flat O(P·K) it/s) to locate the crossover G, if any: GPU it/s degrades with G (more gaussians per tile
+ eventually VRAM pressure), BH stays ~flat. Where they cross (if within feasible G) is where BH wins.

    docker compose run --rm baseline python tools/gsplat_perf_crossover.py --res 800 --G 30000,100000,300000,1000000
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spike import data, metrics  # noqa: E402
from spike.model import GaussianModel  # noqa: E402
from spike.train import DEFAULT_LR  # noqa: E402
import tools.baseline_gsplat as bg  # noqa: E402


def measure(model, cams, imgs, warmup, steps, seed):
    opt = torch.optim.Adam(model.param_groups(DEFAULT_LR))
    rng = random.Random(seed)
    n = len(cams)

    def step():
        i = rng.randrange(n)
        opt.zero_grad(set_to_none=True)
        loss = metrics.loss_fn(bg.render_gsplat(model, cams[i]), imgs[i], 0.2)
        loss.backward()
        opt.step()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    it_s = steps / (time.time() - t)
    # render fps
    with torch.no_grad():
        bg.render_gsplat(model, cams[0])
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            bg.render_gsplat(model, cams[0])
        torch.cuda.synchronize()
        fps = 50 / (time.time() - t)
    return it_s, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--G", default="30000,100000,300000,1000000")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/gsplat_crossover.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cams, imgs = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, device=dev)
    scene = os.path.basename(args.scene.rstrip("/"))
    Gs = [int(x) for x in args.G.split(",") if x]
    rows = []
    for G in Gs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            m = GaussianModel(G, extent=1.5, seed=args.seed, device=dev)
            it_s, fps = measure(m, cams, imgs, args.warmup, args.steps, args.seed)
            vram = torch.cuda.max_memory_allocated() / 1e9
            row = {"scene": scene, "res": args.res, "G": G, "it_per_s": round(it_s, 2),
                   "render_fps": round(fps, 1), "peak_vram_gb": round(vram, 2)}
            print(f"[gsplat] G={G:>8} res{args.res}: {it_s:6.2f} it/s | {fps:7.1f} fps | "
                  f"VRAM {vram:5.2f} GB", flush=True)
        except RuntimeError as e:
            row = {"scene": scene, "res": args.res, "G": G, "error": "OOM" if "memory" in str(e).lower() else str(e)[:80]}
            print(f"[gsplat] G={G:>8} res{args.res}: {row['error']}", flush=True)
            torch.cuda.empty_cache()
        rows.append(row)
        del m
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"device": "RTX 3090", "scene": scene, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
