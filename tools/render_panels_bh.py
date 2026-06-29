"""Emit BH route-B held-out panels in the SHARED render_panels format (GT|route-B render) at the canonical
HOLDOUT_VIEWS, so they line up view-for-view with the NV route-A gsplat panels (docs/matched-views/).
Renders with OUR matrix-native renderer (poly-splat + WSR, binned O(P*K)); DC-only (the .ply has no SH rest).

Run (CPU, sim container): python3 tools/render_panels_bh.py [--scene lego|ficus|playroom]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from spike import data
from spike.camera import Camera
from m4_train_binned import TileMap, _operands
import render_panels

SCENES = {
    "playroom": dict(ply="outputs/bench/playroom_matched.ply", kind="colmap",
                     path="data/db/playroom", downscale=4),
    "lego": dict(ply="outputs/artifacts/lego_bh/lego_g100k_full.ply", kind="blender",
                 path="data/nerf_synthetic/lego", res=800),
    "ficus": dict(ply="outputs/artifacts/ficus_bh/ficus_g100k_full.ply", kind="blender",
                  path="data/nerf_synthetic/ficus", res=800),
}


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


def render_view(m, cam, tmap):
    with torch.no_grad():
        thU, col, oc, w_b, _ = _operands(m, cam, tmap, 1, 128)
        w = torch.relu(torch.einsum("pf,tfg->tpg", tmap.Phi, thU)) ** 2
        C = (w @ col + w_b * m.c_b) / (w @ oc + w_b)
        img = torch.zeros(cam.H * cam.W, 3)
        img[tmap.gidx] = C.reshape(-1, 3)
        return img.reshape(cam.H, cam.W, 3).clamp(0, 1)


def emit(scene, cfg):
    if not os.path.exists(cfg["ply"]):
        print(f"skip {scene}: no .ply at {cfg['ply']}")
        return
    m, n = read_ply(cfg["ply"])
    print(f"[{scene}] loaded {n} gaussians from {cfg['ply']}", flush=True)
    if cfg["kind"] == "colmap":
        cams, imgs = data.load_colmap(cfg["path"], downscale=cfg["downscale"])
        held = list(range(0, len(cams), 8))                      # the every-8th held-out (te) list
        def get(pos):
            c, gt = cams[held[pos]], imgs[held[pos]]
            Hc, Wc = (c.H // 16) * 16, (c.W // 16) * 16
            return Camera(c.R_v, c.t_v, c.fx, c.fy, c.cx, c.cy, Hc, Wc), gt[:Hc, :Wc, :3]
        n_he = len(held)
    else:
        cams, imgs = data.load_blender(cfg["path"], "test", res=cfg["res"], n=25, stride=8)
        def get(pos):
            return cams[pos], imgs[pos][:, :, :3]
        n_he = len(cams)
    gts, rds = [], []
    for pos in render_panels.holdout_view_ids(scene, n_he):
        cam, gt = get(pos)
        rds.append(render_view(m, cam, TileMap(cam.H, cam.W)))
        gts.append(gt)
    paths = render_panels.save_panels(scene, "routeb", gts, rds)
    print(f"[{scene}] -> {paths}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="one of lego/ficus/playroom; default = all available")
    args = ap.parse_args()
    targets = [args.scene] if args.scene else list(SCENES)
    for s in targets:
        emit(s, SCENES[s])


if __name__ == "__main__":
    main()
