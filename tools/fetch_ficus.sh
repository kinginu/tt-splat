#!/usr/bin/env bash
# Fetch the NeRF-synthetic "ficus" scene into data/nerf_synthetic/ficus/.
# Run INSIDE the sim container:  docker compose run --rm sim bash tools/fetch_ficus.sh
#
# Source: HuggingFace dataset pablovela5620/nerf-synthetic-mirror, where ficus is a
# standalone root folder (verified 2026-06-17): transforms_{train,test,val}.json + the
# RGBA PNG frames, standard NeRF "blender" layout. file_path entries omit ".png"
# (standard) — spike/data.py appends it. ~tens of MB; no unzip needed.
set -euo pipefail
DEST="${1:-/workspace/data/nerf_synthetic}"
REPO="pablovela5620/nerf-synthetic-mirror"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main/ficus"
mkdir -p "$DEST"

echo ">> [primary] huggingface_hub snapshot_download (ficus/*)"
python3 -m pip install -q -U huggingface_hub 2>/dev/null || true
REPO="$REPO" DEST="$DEST" python3 - <<'PY' || true
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["REPO"], repo_type="dataset",
                  allow_patterns="ficus/*", local_dir=os.environ["DEST"])
print("primary OK")
PY

if [ ! -f "$DEST/ficus/transforms_train.json" ]; then
  echo ">> [fallback] curl loop driven by the transforms JSONs"
  mkdir -p "$DEST/ficus"
  for s in train test val; do
    curl -fSL -o "$DEST/ficus/transforms_${s}.json" "$BASE/transforms_${s}.json" || true
  done
  BASE="$BASE" DEST="$DEST" python3 - <<'PY'
import json, os, urllib.request
base = os.environ["BASE"]
dest = os.path.join(os.environ["DEST"], "ficus")
for split in ("train", "test", "val"):
    fn = os.path.join(dest, f"transforms_{split}.json")
    if not os.path.exists(fn):
        continue
    for fr in json.load(open(fn))["frames"]:
        rel = fr["file_path"].lstrip("./") + ".png"
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.exists(out):
            urllib.request.urlretrieve(f"{base}/{rel}", out)
print("fallback OK")
PY
fi

echo ">> [sanity]"
DEST="$DEST" python3 - <<'PY'
import glob, json, os
d = os.path.join(os.environ["DEST"], "ficus")
m = json.load(open(os.path.join(d, "transforms_train.json")))
print("camera_angle_x =", m["camera_angle_x"], "| train frames =", len(m["frames"]),
      "| first file_path =", m["frames"][0]["file_path"])
for split in ("train", "test", "val"):
    print(f"  {split} PNGs =", len(glob.glob(os.path.join(d, split, "*.png"))))
PY
echo ">> done -> $DEST/ficus"
