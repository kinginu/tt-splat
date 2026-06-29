"""Render a BH matrix-native 3DGS blender .ply (lego/ficus) at held-out TEST views (stride-8, the eval split),
with OUR renderer (poly-splat + WSR, binned O(P*K)). DC-only (the .ply has no SH rest). Panel = GT | BH render.

Run (CPU, sim container):
  python3 tools/bh_views_blender.py --ply outputs/artifacts/lego_bh/lego_g100k_full.ply --scene data/nerf_synthetic/lego --res 800 --tag lego_g100k
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from spike import data
from m4_train_binned import TileMap, _operands

DISP_H = 256


def read_ply(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        raw = np.frombuffer(f.read(n * 17 * 4), dtype="<f4").reshape(n, 17)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    m = types.SimpleNamespace(
        means3d=t(raw[:, 0:3]), color_dc=t(raw[:, 6:9]), opacity_raw=t(raw[:, 9]),
        log_scales=t(raw[:, 10:13]), quats=t(raw[:, 13:17]))
    m.w_b_raw = torch.tensor(-3.0)
    m.c_b = torch.ones(3)
    return m, n


def to_np(img):
    return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="docs/bh-views-matched")
    args = ap.parse_args()

    m, n = read_ply(args.ply)
    print(f"loaded {n} gaussians from {args.ply}", flush=True)
    cams, imgs = data.load_blender(args.scene, "test", res=args.res, n=25, stride=8)   # the eval held-out split
    pick = [3, 12, 20]                                                                  # 3 spread held-out views
    tmap = TileMap(args.res, args.res)
    os.makedirs(args.out, exist_ok=True)
    for vi, i in enumerate(pick):
        cam, gt = cams[i], imgs[i]
        with torch.no_grad():
            thU, col, oc, w_b, _ = _operands(m, cam, tmap, 1, 128)
            w = torch.relu(torch.einsum("pf,tfg->tpg", tmap.Phi, thU)) ** 2
            C = (w @ col + w_b * m.c_b) / (w @ oc + w_b)
            img = torch.zeros(cam.H * cam.W, 3)
            img[tmap.gidx] = C.reshape(-1, 3)
            img = img.reshape(cam.H, cam.W, 3).clamp(0, 1)
        panel = torch.cat([gt[:, :, :3], img], dim=1)            # GT | BH render
        pim = Image.fromarray(to_np(panel))
        pim = pim.resize((int(pim.width * DISP_H / pim.height), DISP_H))
        path = os.path.join(args.out, f"{args.tag}_holdout_{vi}.png")
        pim.save(path)
        print(f"view {vi} (test idx {i}) -> {path}", flush=True)


if __name__ == "__main__":
    main()
