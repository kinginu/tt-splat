"""Render the BH-trained matrix-native 3DGS playroom .ply at the 3 held-out views the GPU gsplat panels
use (colmap 8 / 16 / 24 -- determined by image-matching the GPU panel GT columns, NOT te[0:3]), with OUR
renderer (poly-splat + WSR) so the panel is faithful (a standard 3DGS viewer would NOT reproduce it).
CPU binned render (O(P*K)); the dense [P,G] would be ~6.5G elems at G=100k. Panel = GT (held-out) | BH render.

Run (CPU, no device): python3 tools/bh_views.py
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

PLY = "outputs/bench/playroom_G100k_mv.ply"
OUT = "docs/bh-views"
VIEWS = [8, 16, 24]          # the colmap indices the GPU gsplat panels use (image-matched), kept aligned
DISP_H = 240                 # resize each panel to this height for the doc


def read_ply(path):
    """Our save_ply layout -> a model object the binned render accepts."""
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
    m.w_b_raw = torch.tensor(-3.0)                  # trainer default (w_b = softplus(-3) ~ 0.049), untrained
    m.c_b = torch.ones(3)                           # white background
    return m, n


def to_np(img):
    return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def main():
    m, n = read_ply(PLY)
    print(f"loaded {n} gaussians from {PLY}", flush=True)
    cams, imgs = data.load_colmap("data/db/playroom", downscale=1)
    os.makedirs(OUT, exist_ok=True)
    for vi, i in enumerate(VIEWS):
        c, gt = cams[i], imgs[i]
        Hc, Wc = (c.H // 16) * 16, (c.W // 16) * 16
        cam = Camera(c.R_v, c.t_v, c.fx, c.fy, c.cx, c.cy, Hc, Wc)
        gtc = gt[:Hc, :Wc, :]
        tmap = TileMap(Hc, Wc)
        with torch.no_grad():                           # binned matrix-native render (poly-splat + WSR)
            thU, col, oc, w_b, _ = _operands(m, cam, tmap, 1, 128)
            w = torch.relu(torch.einsum("pf,tfg->tpg", tmap.Phi, thU)) ** 2     # [T,256,K]
            C = (w @ col + w_b * m.c_b) / (w @ oc + w_b)                        # [T,256,3]
            img = torch.zeros(cam.H * cam.W, 3)
            img[tmap.gidx] = C.reshape(-1, 3)
            img = img.reshape(cam.H, cam.W, 3).clamp(0, 1)
        panel = torch.cat([gtc, img], dim=1)            # GT | BH render
        pim = Image.fromarray(to_np(panel))
        pim = pim.resize((int(pim.width * DISP_H / pim.height), DISP_H))
        path = os.path.join(OUT, f"playroom_holdout_{vi}.png")
        pim.save(path)
        print(f"view {vi} (colmap idx {i}) -> {path}  ({Wc}x{Hc} GT|render)", flush=True)


if __name__ == "__main__":
    main()
