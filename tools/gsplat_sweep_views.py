"""Render representative held-out views from the saved gsplat native-res sweep plys, as a per-scene
panel  GT | gsplat@G30k | gsplat@G100k | gsplat@G300k  (visualizes the few-view overfit as G grows).
The plys themselves are heavy and NOT pushed; only these small downscaled panels + the metrics go to
docs/gsplat-native-sweep/. Runs in the `baseline` container (needs gsplat + PIL).

    docker compose run --rm baseline python tools/gsplat_sweep_views.py
"""
import os
import sys
import types

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spike import data  # noqa: E402
import tools.baseline_gsplat as bg  # noqa: E402

PLY_DIR = "outputs/gsplat_sweep"
OUT = "docs/gsplat-native-sweep"
# ficus: G30k / G100k only (G300k gives no new insight at fixed-count)
# lego: full 3-point sweep for reference
SCENE_GS = {
    "ficus": [1000, 5000, 10000],
    "lego": [1000, 5000, 10000],
}
TILE = 256  # downscale each panel tile to this height (keep pngs small)


def read_ply(path, device):
    """Read a standard 3DGS .ply (the layout plyio.save_ply writes) back into a render-ready object."""
    with open(path, "rb") as f:
        # header
        line = f.readline()
        n = 0
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        raw = np.frombuffer(f.read(n * 17 * 4), dtype="<f4").reshape(n, 17)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return types.SimpleNamespace(
        means3d=t(raw[:, 0:3]),          # x,y,z   (cols 3:6 = normals, skip)
        color_dc=t(raw[:, 6:9]),         # f_dc_0..2
        opacity_raw=t(raw[:, 9]),        # opacity (logit)
        log_scales=t(raw[:, 10:13]),     # scale_0..2 (log)
        quats=t(raw[:, 13:17]),          # rot_0..3
    )


def to_np(img):
    return (img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)


def resize(arr, h):
    im = Image.fromarray(arr)
    w = round(im.width * h / im.height)
    return np.asarray(im.resize((w, h), Image.BILINEAR))


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for scene, gs in SCENE_GS.items():
        labels = ["GT"] + [f"gsplat G{g//1000}k" for g in gs]
        ho_c, ho_i = data.load_blender(f"data/nerf_synthetic/{scene}", "val", res=800, n=1, device=dev)
        cam, gt = ho_c[0], ho_i[0]
        tiles = [resize(to_np(gt), TILE)]
        for g in gs:
            m = read_ply(os.path.join(PLY_DIR, f"{scene}_G{g}.ply"), dev)
            with torch.no_grad():
                tiles.append(resize(to_np(bg.render_gsplat(m, cam)), TILE))
        gap = np.full((TILE, 4, 3), 255, np.uint8)
        row = tiles[0]
        for tl in tiles[1:]:
            row = np.concatenate([row, gap, tl], axis=1)
        strip = 16
        full = np.full((TILE + strip, row.shape[1], 3), 255, np.uint8)
        full[strip:] = row
        im = Image.fromarray(full)
        d = ImageDraw.Draw(im)
        x = 0
        for i, lab in enumerate(labels):
            d.text((x + 2, 3), lab, fill=(0, 0, 0))
            x += tiles[i].shape[1] + 4
        out_path = os.path.join(OUT, f"{scene}_holdout_sweep.png")
        im.save(out_path)
        print(f"saved {scene}_holdout_sweep.png ({im.size})", flush=True)


if __name__ == "__main__":
    main()
