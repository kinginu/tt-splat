"""gsplat (route-A) perf on a REAL COLMAP scene — the first at-scale GPU perf point.

Loads a COLMAP scene (poses + points3D init), then sweeps G measuring train it/s (stochastic 1-view),
render fps, peak VRAM at the scene's native resolution. Gaussians are initialized FROM the SfM points
(so they project into views — realistic footprint), scale = median nearest-neighbor distance.

    docker compose run --rm baseline python tools/gsplat_real_perf.py --root data/db/playroom --G 100000,300000,1000000,2000000
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spike import data, metrics  # noqa: E402
from spike.model import GaussianModel  # noqa: E402
from spike.train import DEFAULT_LR  # noqa: E402
import tools.baseline_gsplat as bg  # noqa: E402


def init_from_points(G, xyz, dev):
    n = len(xyz)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, G) if G > n else rng.choice(n, G, replace=False)
    means = torch.tensor(xyz[idx], dtype=torch.float32, device=dev)
    s = torch.tensor(xyz[rng.choice(n, min(n, 2000), replace=False)], dtype=torch.float32, device=dev)
    d = torch.cdist(s, s)
    d.fill_diagonal_(1e9)
    nn = float(d.min(1).values.median())
    ext = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    m = GaussianModel(G, extent=ext, seed=0, device=dev)
    with torch.no_grad():
        m.means3d.copy_(means + (torch.rand_like(means) - 0.5) * nn * 0.1)
        m.log_scales.fill_(float(np.log(max(nn, 1e-4))))
    return m, nn


def measure(model, cams, imgs, warmup, steps, seed):
    opt = torch.optim.Adam(model.param_groups(DEFAULT_LR))
    rng = random.Random(seed)
    n = len(cams)

    def step():
        i = rng.randrange(n)
        opt.zero_grad(set_to_none=True)
        metrics.loss_fn(bg.render_gsplat(model, cams[i]), imgs[i], 0.2).backward()
        opt.step()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    it_s = steps / (time.time() - t)
    with torch.no_grad():
        bg.render_gsplat(model, cams[0])
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(30):
            bg.render_gsplat(model, cams[0])
        torch.cuda.synchronize()
        fps = 30 / (time.time() - t)
    return it_s, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/db/playroom")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--G", default="100000,300000,1000000,2000000")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="outputs/gsplat_real_perf.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cams, imgs = data.load_colmap(args.root, downscale=args.downscale, n=args.n_train, device=dev)
    xyz, _ = data.load_colmap_points(args.root)
    scene = os.path.basename(args.root.rstrip("/"))
    H, W = cams[0].H, cams[0].W
    print(f"[{scene}] {len(cams)} views @ {W}x{H} | {len(xyz)} SfM points", flush=True)
    rows = []
    for G in [int(x) for x in args.G.split(",") if x]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            m, nn = init_from_points(G, xyz, dev)
            it_s, fps = measure(m, cams, imgs, args.warmup, args.steps, 0)
            vram = torch.cuda.max_memory_allocated() / 1e9
            row = {"scene": scene, "W": W, "H": H, "G": G, "it_per_s": round(it_s, 2),
                   "render_fps": round(fps, 1), "peak_vram_gb": round(vram, 2)}
            print(f"[gsplat] G={G:>8}: {it_s:6.2f} it/s | {fps:7.1f} fps | VRAM {vram:5.2f} GB", flush=True)
            del m
        except RuntimeError as e:
            row = {"scene": scene, "G": G, "error": "OOM" if "memory" in str(e).lower() else str(e)[:80]}
            print(f"[gsplat] G={G:>8}: {row['error']}", flush=True)
            torch.cuda.empty_cache()
        rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"device": "RTX 3090", "scene": scene, "res": f"{W}x{H}", "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
