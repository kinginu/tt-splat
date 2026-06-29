"""Render the BH matched playroom .ply (outputs/bench/playroom_matched.ply, the res304/downscale-4
G=100k full-feature run = 20.11 dB held-out) at a few held-out views, with OUR renderer (poly-splat + WSR).
Renders at downscale=4 (the res it was TRAINED at — full-res cameras would mismatch the intrinsics).
DC-only (the .ply has no SH rest; the view-dependent SH3 refinement is not stored). Panel = GT | BH render.

Run (CPU, sim container): python3 tools/bh_views_matched.py
"""
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
from spike.camera import Camera
from m4_train_binned import TileMap, _operands

PLY = "outputs/bench/playroom_matched.ply"
OUT = "docs/bh-views-matched"
DOWNSCALE = 4                       # the res the matched run was trained at (1264x832 / 4 -> ~316x208)
DISP_H = 240


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
    m, n = read_ply(PLY)
    print(f"loaded {n} gaussians from {PLY}", flush=True)
    cams, imgs = data.load_colmap("data/db/playroom", downscale=DOWNSCALE)
    # held-out = every-8th view (the matched-run split); pick 3 spread across it
    held = list(range(0, len(cams), 8))
    pick = [held[len(held) // 4], held[len(held) // 2], held[3 * len(held) // 4]]
    os.makedirs(OUT, exist_ok=True)
    for vi, i in enumerate(pick):
        c, gt = cams[i], imgs[i]
        Hc, Wc = (c.H // 16) * 16, (c.W // 16) * 16
        cam = Camera(c.R_v, c.t_v, c.fx, c.fy, c.cx, c.cy, Hc, Wc)
        gtc = gt[:Hc, :Wc, :]
        tmap = TileMap(Hc, Wc)
        with torch.no_grad():
            thU, col, oc, w_b, _ = _operands(m, cam, tmap, 1, 128)
            w = torch.relu(torch.einsum("pf,tfg->tpg", tmap.Phi, thU)) ** 2
            C = (w @ col + w_b * m.c_b) / (w @ oc + w_b)
            img = torch.zeros(cam.H * cam.W, 3)
            img[tmap.gidx] = C.reshape(-1, 3)
            img = img.reshape(cam.H, cam.W, 3).clamp(0, 1)
        panel = torch.cat([gtc, img], dim=1)            # GT | BH render
        pim = Image.fromarray(to_np(panel))
        pim = pim.resize((int(pim.width * DISP_H / pim.height), DISP_H))
        path = os.path.join(OUT, f"playroom_matched_holdout_{vi}.png")
        pim.save(path)
        print(f"view {vi} (colmap idx {i}) -> {path}  ({Wc}x{Hc} GT|render)", flush=True)


if __name__ == "__main__":
    main()
