"""Merge the shared routeb (GT|route-B) and gsplat (GT|route-A) held-out panels into a single
GT | route-B (matrix-native) | route-A (gsplat) comparison strip per held-out view.
Both inputs are render_panels.save_panels output (h=240, GT left / render right, equal-width halves).

Run (CPU, sim container): python3 tools/merge_panels.py
"""
import os

import numpy as np
from PIL import Image

MV = "docs/matched-views"
H = 240
GAP = 6


def halves(path):
    im = Image.open(path).convert("RGB")
    w = im.width
    return im.crop((0, 0, w // 2, im.height)), im.crop((w // 2, 0, w, im.height))   # GT, render


def main():
    seam = Image.fromarray(np.full((H, GAP, 3), 255, np.uint8))
    for scene in ("playroom", "lego", "ficus"):
        for i in range(3):
            rb_p = f"{MV}/{scene}_routeb_holdout_{i}.png"
            gs_p = f"{MV}/{scene}_gsplat_holdout_{i}.png"
            if not (os.path.exists(rb_p) and os.path.exists(gs_p)):
                continue
            gt, rb = halves(rb_p)            # GT + route-B from the BH panel
            _, ra = halves(gs_p)             # route-A from the gsplat panel
            cols = [gt, seam, rb, seam, ra]
            W = sum(c.width for c in cols)
            out = Image.new("RGB", (W, H), (255, 255, 255))
            x = 0
            for c in cols:
                out.paste(c, (x, 0)); x += c.width
            p = f"{MV}/{scene}_compare_holdout_{i}.png"
            out.save(p)
            print(f"{scene} view {i} -> {p}  (GT | route-B | route-A, {W}x{H})")


if __name__ == "__main__":
    main()
