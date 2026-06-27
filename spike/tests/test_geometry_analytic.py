"""Test 3 (NON-skippable): a gsplat-free analytic check of the EWA projection.
An isotropic 3D gaussian (world radius s) at camera-space (0,0,z) projects to a 2D conic
with analytically-predicted diagonal 1/((s*fx/z)^2 + blur) and mu2d at the principal point."""
import _bootstrap  # noqa: F401
import torch

from spike import geometry

DT = torch.float64


def test_isotropic_projection():
    s = 0.1                       # world-space std
    z = 4.0                       # depth
    fx = fy = 178.0
    cx = cy = 64.0
    blur = 0.3

    means3d = torch.tensor([[0.0, 0.0, 0.0]], dtype=DT)
    R_v = torch.eye(3, dtype=DT)
    t_v = torch.tensor([0.0, 0.0, z], dtype=DT)             # mu_cam = (0,0,z)
    cov = (s * s) * torch.eye(3, dtype=DT).unsqueeze(0)     # Sigma3 = s^2 I (isotropic)

    mu2d, conic_abc, depth, keep = geometry.project_ewa(
        means3d, cov, R_v, t_v, fx, fy, cx, cy, blur_eps=blur, near=0.2)

    assert torch.allclose(mu2d[0], torch.tensor([cx, cy], dtype=DT), atol=1e-9), mu2d[0]
    assert abs(depth.item() - z) < 1e-9 and bool(keep[0])

    sigma2d_diag = (s * fx / z) ** 2 + blur                # analytic 2D variance
    conic_pred = 1.0 / sigma2d_diag
    a, b, c = conic_abc[0].tolist()
    assert abs(a - conic_pred) < 1e-6, (a, conic_pred)
    assert abs(c - conic_pred) < 1e-6, (c, conic_pred)
    assert abs(b) < 1e-9, b


def test_depth_scaling():
    # doubling depth halves the projected std -> quadruples the conic diagonal (minus blur effects).
    s, fx, fy, cx, cy, blur = 0.05, 178.0, 178.0, 64.0, 64.0, 0.0
    means3d = torch.tensor([[0.0, 0.0, 0.0]], dtype=DT)
    R_v = torch.eye(3, dtype=DT)

    def conic_at(z):
        t_v = torch.tensor([0.0, 0.0, z], dtype=DT)
        cov = (s * s) * torch.eye(3, dtype=DT).unsqueeze(0)
        _, conic_abc, _, _ = geometry.project_ewa(means3d, cov, R_v, t_v, fx, fy, cx, cy,
                                                  blur_eps=blur, near=0.2)
        return conic_abc[0, 0].item()

    assert abs(conic_at(8.0) / conic_at(4.0) - 4.0) < 1e-6


def test_near_cull():
    means3d = torch.tensor([[0.0, 0.0, 0.0]], dtype=DT)
    R_v = torch.eye(3, dtype=DT)
    t_v = torch.tensor([0.0, 0.0, 0.05], dtype=DT)          # in front of near plane (0.2)
    cov = 0.01 * torch.eye(3, dtype=DT).unsqueeze(0)
    _, _, _, keep = geometry.project_ewa(means3d, cov, R_v, t_v, 100.0, 100.0, 64.0, 64.0,
                                         blur_eps=0.3, near=0.2)
    assert not bool(keep[0])


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
