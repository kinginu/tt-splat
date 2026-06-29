"""SHARED held-out representative-view panel spec — NV (route-A gsplat) and BH (route-B matrix-native)
MUST emit panels in this identical format so they line up view-for-view in the matched-comparison doc.

Format (single source of truth):
  - panel        = GT (left) | rendered (right), horizontal concat, GAP px white seam
  - resized to   = height DISP_H (aspect preserved)
  - filename     = {scene}_{side}_holdout_{i}.png   side: "gsplat" (route-A) | "routeb" (route-B)
  - out dir      = docs/matched-views/  (committed; .ply stays gitignored)
  - holdout_{i}  = the SAME camera on both sides (HOLDOUT_VIEWS indexes the loaded held-out list)

A later merge step (tools/playroom_gt_bh_nv.py style) can stack the two sides into GT|route-B|route-A.
Both NV and BH import THIS module so the format can never drift; do not fork it."""
import os

import numpy as np
from PIL import Image, ImageDraw

DISP_H = 240
GAP = 4
OUT_DIR = "docs/matched-views"

# Canonical held-out view positions = index into the loaded held-out (te) list, so holdout_i is the
# SAME physical camera on the NV and BH side. playroom [1,2,3] -> colmap every-8th views 8/16/24
# (kept aligned with the legacy docs/bh-views panels); blender scenes -> first 3 test (stride-8) views.
HOLDOUT_VIEWS = {
    "playroom": [1, 2, 3],
    "lego": [0, 1, 2],
    "ficus": [0, 1, 2],
}


def holdout_view_ids(scene, n_heldout):
    """The held-out-list positions to render for `scene`, clamped to what's loaded."""
    return [v for v in HOLDOUT_VIEWS.get(scene, [0, 1, 2]) if v < n_heldout]


def _to_np(img):
    return (img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)


def save_panels(scene, side, gt_list, render_list, out_dir=OUT_DIR, disp_h=DISP_H, label=None):
    """gt_list/render_list: aligned [H,W,3] tensors in [0,1] for the chosen held-out views.
    side: 'gsplat' (route-A) or 'routeb' (route-B). Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (gt, rd) in enumerate(zip(gt_list, render_list)):
        g, r = _to_np(gt), _to_np(rd)
        H = g.shape[0]
        seam = np.full((H, GAP, 3), 255, np.uint8)
        panel = np.concatenate([g, seam, r], axis=1)
        im = Image.fromarray(panel)
        im = im.resize((max(1, int(im.width * disp_h / im.height)), disp_h), Image.BILINEAR)
        if label:
            d = ImageDraw.Draw(im)
            d.text((3, 2), label, fill=(0, 0, 0))
        p = os.path.join(out_dir, f"{scene}_{side}_holdout_{i}.png")
        im.save(p)
        paths.append(p)
    return paths
