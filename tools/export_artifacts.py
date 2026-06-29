"""Train the (B) model on a NeRF-synthetic scene and export deliverable artifacts:
  - the trained model as a standard 3DGS .ply
  - for the TRAIN views and held-out TEST views: the GT image, our (B) forward render, and a
    side-by-side (GT | render) for easy comparison.

Run inside a container (CPU here -- this box has no NVIDIA GPU; keep the config modest):
    podman-compose --profile hw run --rm hw python3 tools/export_artifacts.py \
        --res 96 --G 1500 --iters 600 --n-train 6 --n-test 2

Output layout (gitignored under outputs/):
    outputs/artifacts/<scene>/
      model.ply
      train/   viewNN_gt.png viewNN_render.png viewNN_sbs.png    <- "sample" set (images used in training)
      test/    viewNN_gt.png viewNN_render.png viewNN_sbs.png    <- held-out test set
      metrics.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from spike import data, metrics, plyio, train, evalcard
from spike.device import default_device
from spike.model import GaussianModel
from spike.render import render


def save_img(path, img):
    Image.fromarray((img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def side_by_side(gt, pred, gap=4):
    H = gt.shape[0]
    sep = torch.ones(H, gap, 3, dtype=gt.dtype, device=gt.device)
    return torch.cat([gt.clamp(0, 1), sep, pred.clamp(0, 1)], dim=1)


def _spread(total, n):
    return max(1, total // max(1, n))


@torch.no_grad()
def dump_split(model, cams, imgs, arm, out, k):
    os.makedirs(out, exist_ok=True)
    rows = []
    for i, (cam, gt) in enumerate(zip(cams, imgs)):
        r = render(model, cam, arm, k=k)
        ps = float(metrics.psnr(r, gt))
        ss = float(metrics.ssim(r, gt))
        save_img(os.path.join(out, f"view{i:02d}_gt.png"), gt)
        save_img(os.path.join(out, f"view{i:02d}_render.png"), r)
        save_img(os.path.join(out, f"view{i:02d}_sbs.png"), side_by_side(gt, r))
        rows.append((i, ps, ss))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--G", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--arm", default="A")
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--method", default="routeB", help="rough algo label for the unified output path")
    ap.add_argument("--out", default="outputs", help="root; layout = <out>/<method>/<scene>/G<G>_res<res>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = default_device(None)
    scene = os.path.basename(args.scene.rstrip("/"))
    paths = evalcard.run_paths(args.method, scene, args.G, args.res, iters=args.iters,
                               seed=args.seed, root=args.out)
    out = paths["dir"]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    tr_cams, tr_imgs = data.load_blender(args.scene, "train", res=args.res, device=dev,
                                         n=args.n_train, stride=_spread(100, args.n_train))
    te_cams, te_imgs = data.load_blender(args.scene, "test", res=args.res, device=dev,
                                         n=args.n_test, stride=_spread(200, args.n_test))
    te_cams_a, te_rgba = data.load_blender(args.scene, "test", res=args.res, device=dev,
                                           n=args.n_test, stride=_spread(200, args.n_test), keep_alpha=True)
    print(f"[export] {scene} device={dev} res={args.res} G={args.G} arm={args.arm} "
          f"| {len(tr_cams)} train + {len(te_cams)} test views | training {args.iters} it...")

    model = GaussianModel(args.G, extent=1.5, seed=args.seed, device=dev)
    t0 = time.perf_counter()
    hist = train.fit(model, tr_cams, tr_imgs, args.arm, iters=args.iters,
                     k=args.k, log_every=max(1, args.iters // 10))
    tt = time.perf_counter() - t0
    print(f"[export] trained in {tt:.1f}s ({tt/max(args.iters,1)*1e3:.0f} ms/it), final loss {hist[-1]:.4f}")

    n = plyio.save_ply(paths["ply"], model)
    print(f"[export] wrote {paths['ply']}: {n} gaussians, {os.path.getsize(paths['ply'])/1024:.0f} KB")

    tr_rows = dump_split(model, tr_cams, tr_imgs, args.arm, os.path.join(out, "train"), args.k)
    te_rows = dump_split(model, te_cams, te_imgs, args.arm, os.path.join(out, "test"), args.k)
    tr_mean = sum(p for _, p, _ in tr_rows) / len(tr_rows)
    te_mean = sum(p for _, p, _ in te_rows) / len(te_rows)

    # unified eval card (route-B render_fn = spike render with c_b swapped for the two-bg coverage trick)
    def rfn(mdl, cam, b):
        saved = mdl.c_b.detach().clone()
        mdl.c_b.fill_(b)
        o = render(mdl, cam, args.arm, k=args.k)
        mdl.c_b.copy_(saved)
        return o
    card = evalcard.build(model, te_cams_a, te_rgba, rfn, method=args.method, scene=scene,
                          G=args.G, res=args.res, iters=args.iters, seed=args.seed, arm=args.arm,
                          train_psnr=tr_mean,
                          perf={"train_s": round(tt, 1), "ms_per_it": round(tt / max(args.iters, 1) * 1e3, 2)})
    card["perceptual"]["holdout_psnr_dump"] = round(te_mean, 2)
    evalcard.save(card, paths["eval_json"])
    print(f"[export] TRAIN {tr_mean:.2f} dB | TEST {te_mean:.2f} dB -> {paths['eval_json']}")
    print(f"[export] -> {out}/  ({os.path.basename(paths['ply'])}, {os.path.basename(paths['eval_json'])}, train/, test/)")


if __name__ == "__main__":
    main()
