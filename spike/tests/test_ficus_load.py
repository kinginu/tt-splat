"""Real-data smoke test (SKIPS if ficus isn't downloaded). Loads actual ficus frames and
pushes them through the full render pipeline, verifying white-bg compositing, that the
cameras genuinely look at the object (the OpenGL->OpenCV conversion on real poses), and
that every arm renders finite. Run `bash tools/fetch_ficus.sh` (in the container) first."""
import os

import _bootstrap  # noqa: F401
import torch

from spike import data
from spike.model import GaussianModel
from spike.render import render, ARMS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FICUS = os.path.join(ROOT, "data", "nerf_synthetic", "ficus")


def _require_data():
    if not os.path.exists(os.path.join(FICUS, "transforms_train.json")):
        try:
            import pytest
            pytest.skip("ficus not downloaded (run tools/fetch_ficus.sh)")
        except ImportError:
            raise SystemExit(0)


def test_ficus_loads_and_renders():
    _require_data()
    cams, imgs = data.load_blender(FICUS, split="train", res=128, n=4)
    assert len(cams) == len(imgs) == 4
    for img in imgs:
        assert img.shape == (128, 128, 3)
        assert 0.0 <= img.min().item() and img.max().item() <= 1.0
    # white background dominates NeRF-synthetic frames
    assert imgs[0].mean().item() > 0.7

    # cameras point at the object near the origin (catches a pose-conversion sign error)
    for cam in cams:
        c = cam.center
        look_dot = ((-c / c.norm()) @ cam.R_v[2]).item()
        assert look_dot > 0.99, look_dot

    # full pipeline renders finite for every arm at the loaded resolution
    model = GaussianModel(200, extent=1.5, seed=0)
    for arm in ARMS:
        out = render(model, cams[0], arm)
        assert out.shape == (128, 128, 3) and torch.isfinite(out).all(), arm


def test_holdout_split_distinct():
    _require_data()
    train, _ = data.load_blender(FICUS, split="train", res=64, indices=[0, 1])
    val, _ = data.load_blender(FICUS, split="val", res=64, indices=[0, 1])
    # train vs val poses must differ (genuine held-out views)
    assert not torch.allclose(train[0].center, val[0].center, atol=1e-3)


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
