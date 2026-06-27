"""Build GT | BH matrix-native | NV gsplat comparison panels for playroom held-out views.
- GT + gsplat@G100k columns are cropped from the GPU panels (docs/gsplat-real-views, on origin/main,
  fetched to /tmp/gpu_panels/p{0,1,2}.png) -- gsplat can't be re-rendered here (no gsplat .ply / no GPU).
- BH column = the right half of our own panels (docs/bh-views/playroom_holdout_*.png).
The GPU panels p0/p1/p2 are colmap views 8/16/24 (image-matched); bh_views.py renders the SAME indices,
so all three columns are the same viewpoint. Output: docs/compare-views/playroom_{0,1,2}.png.
Run (CPU): python3 tools/playroom_gt_bh_nv.py
"""
import os
from PIL import Image, ImageDraw

GPU = "/tmp/gpu_panels"            # p0/p1/p2.png = GT | gsplat@G100k | gsplat@G1M (1103x256, ~4px gaps)
BH = "docs/bh-views"               # playroom_holdout_N.png = GT | BH render (same colmap view as pN)
OUT = "docs/compare-views"
COLW, GAP, LH = 365, 4, 22         # GPU panel column width / gap / label-strip height
CW, CHh = 360, 234                 # output cell (w,h) per column
LABELS = ["GT (held-out)", "BH matrix-native (17.4 dB)", "gsplat (26.5 dB)"]


def main():
    os.makedirs(OUT, exist_ok=True)
    for i in range(3):
        gpu = Image.open(os.path.join(GPU, f"p{i}.png")).convert("RGB")     # 1103x256
        gt = gpu.crop((0, LH, COLW, 256))                                   # GT image (label dropped)
        nv = gpu.crop((COLW + GAP, LH, 2 * COLW + GAP, 256))                # gsplat@G100k image
        bhp = Image.open(os.path.join(BH, f"playroom_holdout_{i}.png")).convert("RGB")
        bh = bhp.crop((bhp.width // 2, 0, bhp.width, bhp.height))           # right half = BH render
        cells = [c.resize((CW, CHh)) for c in (gt, bh, nv)]

        hdr = 20
        W = 3 * CW + 2 * GAP
        canvas = Image.new("RGB", (W, hdr + CHh), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for k, (cell, lab) in enumerate(zip(cells, LABELS)):
            x = k * (CW + GAP)
            d.text((x + 4, 4), lab, fill=(0, 0, 0))
            canvas.paste(cell, (x, hdr))
        path = os.path.join(OUT, f"playroom_{i}.png")
        canvas.save(path)
        print(f"wrote {path} ({W}x{hdr + CHh})", flush=True)


if __name__ == "__main__":
    main()
