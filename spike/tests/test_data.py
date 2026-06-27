"""Loader test (data-free): build a tiny fake NeRF-synthetic dataset in a temp dir and load
it, checking the OpenGL->OpenCV pose, white compositing, intrinsics, and frame selection.
Validates data.py without needing the real ficus download."""
import json
import math
import os
import tempfile

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image

from spike import camera, data

DT = torch.float64


def _write_fake_scene(root, angle_x=0.6911112070083618):
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    # 8x8 RGBA with known pixels: (0,0)=red opaque, (0,1)=green alpha0, (1,0)=blue alpha~0.5
    px = np.zeros((8, 8, 4), dtype=np.uint8)
    px[:, :] = (10, 20, 30, 255)
    px[0, 0] = (255, 0, 0, 255)
    px[0, 1] = (0, 255, 0, 0)
    px[1, 0] = (0, 0, 255, 128)
    for name in ("r_0", "r_1"):
        Image.fromarray(px, mode="RGBA").save(os.path.join(root, "train", f"{name}.png"))

    c2w0 = torch.eye(4).tolist()
    c2w1 = torch.eye(4)
    c2w1[2, 3] = 4.0
    meta = {
        "camera_angle_x": angle_x,
        "frames": [
            {"file_path": "./train/r_0", "transform_matrix": c2w0},
            {"file_path": "./train/r_1", "transform_matrix": c2w1.tolist()},
        ],
    }
    with open(os.path.join(root, "transforms_train.json"), "w") as f:
        json.dump(meta, f)
    return angle_x


def test_composite_over_white():
    rgba = torch.tensor([[1.0, 0.0, 0.0, 1.0],   # opaque red -> red
                         [0.0, 1.0, 0.0, 0.0],   # transparent -> white
                         [0.0, 0.0, 1.0, 0.5]],  # half blue -> (0.5,0.5,1)
                        dtype=DT)
    out = data.composite_over_white(rgba)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0], dtype=DT))
    assert torch.allclose(out[1], torch.tensor([1.0, 1.0, 1.0], dtype=DT))
    assert torch.allclose(out[2], torch.tensor([0.5, 0.5, 1.0], dtype=DT))


def test_intrinsics_fov():
    fx, fy, cx, cy = data.intrinsics_from_fov(0.6911112070083618, 800, 800)
    assert abs(fx - 0.5 * 800 / math.tan(0.5 * 0.6911112070083618)) < 1e-6
    assert fx == fy and cx == 400.0 and cy == 400.0


def test_select_indices():
    assert data.select_indices(10) == list(range(10))
    assert data.select_indices(10, stride=3) == [0, 3, 6, 9]
    assert data.select_indices(10, stride=3, n=2) == [0, 3]
    assert data.select_indices(10, indices=[2, 5]) == [2, 5]


def test_load_fake_scene():
    with tempfile.TemporaryDirectory() as root:
        angle_x = _write_fake_scene(root)
        cams, imgs = data.load_blender(root, split="train", res=8, dtype=DT)

        assert len(cams) == 2 and len(imgs) == 2
        assert imgs[0].shape == (8, 8, 3)

        # white compositing of the known pixels
        assert torch.allclose(imgs[0][0, 0], torch.tensor([1.0, 0.0, 0.0], dtype=DT), atol=2e-2)
        assert torch.allclose(imgs[0][0, 1], torch.tensor([1.0, 1.0, 1.0], dtype=DT), atol=2e-2)
        b = imgs[0][1, 0]
        assert b[2] > 0.95 and abs(b[0] - 128 / 255) < 3e-2

        # pose matches the pinned camera conversion (identity c2w -> diag(1,-1,-1))
        R_v_expected, _ = camera.opengl_c2w_to_opencv_w2c(torch.eye(4, dtype=DT))
        assert torch.allclose(cams[0].R_v, R_v_expected, atol=1e-9)

        # intrinsics at res=8
        fx_expected = 0.5 * 8 / math.tan(0.5 * angle_x)
        assert abs(cams[0].fx - fx_expected) < 1e-6
        assert cams[0].H == 8 and cams[0].W == 8

        # second camera is translated (+z in OpenGL -> different t_v), pose still orthonormal
        assert torch.allclose(cams[1].R_v @ cams[1].R_v.T, torch.eye(3, dtype=DT), atol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
