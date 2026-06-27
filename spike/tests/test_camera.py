"""Test 2 (NON-skippable): the OpenGL->OpenCV camera conversion. A point in
front of the camera must have depth z>0 and land at the predicted pixel. Catches the
y/z axis-flip silent error independent of gsplat."""
import _bootstrap  # noqa: F401
import torch

from spike import camera

DT = torch.float64


def _f(x):
    return torch.tensor(x, dtype=DT)


def test_identity_camera_looks_down_neg_z():
    # OpenGL camera at origin, R_c = I -> looks down -z. A point at (0,0,-5) is in front.
    c2w = torch.eye(4, dtype=DT)
    R_v, t_v = camera.opengl_c2w_to_opencv_w2c(c2w)
    fx = fy = 100.0
    cx = cy = 64.0

    pix, z = camera.project_point(_f([0.0, 0.0, -5.0]), R_v, t_v, fx, fy, cx, cy)
    assert z.item() > 0, z.item()                          # in front in OpenCV (+z forward)
    assert torch.allclose(pix, _f([cx, cy]), atol=1e-9), pix

    # +x in world -> +x in OpenCV cam (flip leaves x): pixel shifts right.
    pix2, z2 = camera.project_point(_f([1.0, 0.0, -5.0]), R_v, t_v, fx, fy, cx, cy)
    assert abs(pix2[0].item() - (fx * 1.0 / 5.0 + cx)) < 1e-9, pix2
    assert z2.item() > 0

    # a point behind the OpenGL camera (+z) must be behind in OpenCV (z<0).
    _, z_back = camera.project_point(_f([0.0, 0.0, 5.0]), R_v, t_v, fx, fy, cx, cy)
    assert z_back.item() < 0, z_back.item()


def test_translated_camera_depth():
    # camera translated to (0,0,4) (still looking down -z); world origin is 4 in front.
    c2w = torch.eye(4, dtype=DT)
    c2w[2, 3] = 4.0
    R_v, t_v = camera.opengl_c2w_to_opencv_w2c(c2w)
    fx = fy = 100.0
    cx = cy = 64.0
    pix, z = camera.project_point(_f([0.0, 0.0, 0.0]), R_v, t_v, fx, fy, cx, cy)
    assert abs(z.item() - 4.0) < 1e-9, z.item()
    assert torch.allclose(pix, _f([cx, cy]), atol=1e-9), pix


def test_rotation_is_orthonormal():
    # a non-trivial pose: rotate 90deg about world y, translate. R_v must stay orthonormal.
    th = torch.tensor(0.5, dtype=DT)
    Ry = torch.tensor([[torch.cos(th), 0, torch.sin(th)],
                       [0, 1, 0],
                       [-torch.sin(th), 0, torch.cos(th)]], dtype=DT)
    c2w = torch.eye(4, dtype=DT)
    c2w[:3, :3] = Ry
    c2w[:3, 3] = _f([1.0, 2.0, 3.0])
    R_v, _ = camera.opengl_c2w_to_opencv_w2c(c2w)
    assert torch.allclose(R_v @ R_v.T, torch.eye(3, dtype=DT), atol=1e-9)
    assert abs(torch.det(R_v).item() - 1.0) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
