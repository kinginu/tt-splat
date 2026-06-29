"""Export representative comparison views for the axis-1 quality benchmark.

Trains BOTH renderers at the matched bench config (same data split / init / LRs) and dumps, per view,
a labeled 3-way panel  [ GT | gsplat (route-A) | route-B (arm A) ]  so the held-out quality gap is
visible (notably lego's solid-occluder failure). Runs INSIDE the `baseline` container (has gsplat):

    docker compose run --rm baseline python tools/export_bench_views.py \
        --scenes ficus,lego --res 128 --G 2000 --iters 3000 --seed 0

Full set -> outputs/artifacts_bench/<scene>/ (gitignored). Curated subset (held-out + 2 train) ->
docs/benchmark-ficus-lego/ (committed, mirrors the BH docs/<run>/ convention — NOT everything).
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for baseline_gsplat

from spike import data, metrics  # noqa: E402
from spike.render import render  # noqa: E402
from spike.model import GaussianModel  # noqa: E402
from spike.train import fit  # noqa: E402
from spike.device import default_device  # noqa: E402
import baseline_gsplat as bg  # noqa: E402

LABELS = ["GT", "gsplat (route-A)", "route-B (arm A)"]


def to_np(img):
    return (img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)


def panel(gt, gs, rb, gap=4, strip=16):
    """Labeled horizontal panel [GT | gsplat | route-B] as a PIL image."""
    tiles = [to_np(gt), to_np(gs), to_np(rb)]
    H, W = tiles[0].shape[:2]
    sep = np.full((H, gap, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, sep, t], axis=1)
    full = np.full((H + strip, row.shape[1], 3), 255, np.uint8)
    full[strip:] = row
    im = Image.fromarray(full)
    d = ImageDraw.Draw(im)
    for i, lab in enumerate(LABELS):
        d.text((i * (W + gap) + 2, 3), lab, fill=(0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="ficus,lego")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-holdout", type=int, default=2)
    ap.add_argument("--extent", type=float, default=1.5)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--train-keep", type=int, default=2, help="# of train views to curate into docs/")
    ap.add_argument("--out", default="outputs/artifacts_bench")
    ap.add_argument("--docs-out", default="docs/benchmark-ficus-lego")
    args = ap.parse_args()

    dev = default_device()
    os.makedirs(args.docs_out, exist_ok=True)
    summary = []

    for scene in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        root = os.path.join("data/nerf_synthetic", scene)
        tr_cams, tr_imgs = data.load_blender(root, "train", res=args.res, n=args.n_train, device=dev)
        ho_cams, ho_imgs = data.load_blender(root, "val", res=args.res, n=args.n_holdout, device=dev)
        print(f"[{scene}] {len(tr_cams)} train + {len(ho_cams)} holdout @ {args.res}px | dev={dev}", flush=True)

        torch.manual_seed(args.seed)
        mb = GaussianModel(args.G, extent=args.extent, seed=args.seed, device=dev)
        fit(mb, tr_cams, tr_imgs, "A", iters=args.iters)

        mg = GaussianModel(args.G, extent=args.extent, seed=args.seed, device=dev)
        bg.fit_gsplat(mg, tr_cams, tr_imgs, iters=args.iters)

        @torch.no_grad()
        def psnrs(cams, imgs):
            rb = [float(metrics.psnr(render(mb, c, "A", k=args.k), g)) for c, g in zip(cams, imgs)]
            gs = [float(metrics.psnr(bg.render_gsplat(mg, c), g)) for c, g in zip(cams, imgs)]
            return sum(gs) / len(gs), sum(rb) / len(rb)

        gs_ho, rb_ho = psnrs(ho_cams, ho_imgs)
        gs_tr, rb_tr = psnrs(tr_cams, tr_imgs)
        print(f"[{scene}] holdout  gsplat {gs_ho:.2f}  route-B {rb_ho:.2f}  (Δ {rb_ho-gs_ho:+.2f})", flush=True)

        # dump full set (gitignored) + curate subset into docs/
        for split, cams, imgs, keep in [("test", ho_cams, ho_imgs, len(ho_cams)),
                                        ("train", tr_cams, tr_imgs, args.train_keep)]:
            d = os.path.join(args.out, scene, split)
            os.makedirs(d, exist_ok=True)
            for i, (cam, gt) in enumerate(zip(cams, imgs)):
                with torch.no_grad():
                    im = panel(gt, bg.render_gsplat(mg, cam), render(mb, cam, "A", k=args.k))
                im.save(os.path.join(d, f"view{i:02d}_cmp.png"))
                if i < keep:
                    im.save(os.path.join(args.docs_out, f"{scene}_{split}_view{i:02d}_cmp.png"))

        summary.append(f"{scene:6} | holdout gsplat {gs_ho:6.2f}  route-B {rb_ho:6.2f}  Δ {rb_ho-gs_ho:+5.2f}"
                       f" | train gsplat {gs_tr:6.2f}  route-B {rb_tr:6.2f}")

    with open(os.path.join(args.docs_out, "metrics.txt"), "w") as f:
        f.write(f"# axis-1 bench representative views | res{args.res}/G{args.G}/{args.iters}it/seed{args.seed}\n")
        f.write("# panel order: GT | gsplat (route-A) | route-B (arm A)\n\n")
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"saved curated views -> {args.docs_out}/")


if __name__ == "__main__":
    main()
