"""Test 5: single-gaussian closed-form render. At the center Q=0 -> w=o and the
WSR output equals the gaussian color; at the cutoff Q=k -> w=0 and the pixel is background."""
import math

import _bootstrap  # noqa: F401
import torch

from spike import arms, forward

DT = torch.float64


def _f(x):
    return torch.tensor(x, dtype=DT)


def _logit(p):
    return math.log(p / (1 - p))


def test_single_gaussian():
    k = 4.0
    cx = cy = 8.0
    mu2d = _f([[cx, cy]])
    conic = _f([[0.25, 0.0, 0.25]])          # Q = 0.25*(dx^2+dy^2); cutoff Q=k at radius 4 px
    color = _f([[0.7, 0.2, 0.1]])
    opacity_raw = _f([_logit(0.8)])
    w_b = _f(1e-6)
    c_b = _f([1.0, 1.0, 1.0])

    # center: Q=0, w_geo=1, w=o, render == color
    px, py = _f([cx]), _f([cy])
    Q = forward.quad_form(px, py, mu2d, conic)
    assert abs(Q.item()) < 1e-12
    w_geo = forward.poly_splat_wgeo(Q, k)
    assert abs(w_geo.item() - 1.0) < 1e-12
    C = arms.blend_A(w_geo, opacity_raw, color, w_b, c_b)
    assert torch.allclose(C[0], color[0], atol=1e-3), C[0]

    # at the cutoff radius (dx=4): Q = 0.25*16 = 4 = k -> u=0 -> w_geo=0 -> background
    px2, py2 = _f([cx + 4.0]), _f([cy])
    Q2 = forward.quad_form(px2, py2, mu2d, conic)
    assert abs(Q2.item() - 4.0) < 1e-9
    w2 = forward.poly_splat_wgeo(Q2, k)
    assert w2.item() == 0.0
    C2 = arms.blend_A(w2, opacity_raw, color, w_b, c_b)
    assert torch.allclose(C2[0], c_b, atol=1e-3), C2[0]


def test_beyond_cutoff_is_zero():
    k = 4.0
    mu2d = _f([[0.0, 0.0]])
    conic = _f([[0.25, 0.0, 0.25]])
    px, py = _f([10.0]), _f([0.0])           # Q = 25 > k
    Q = forward.quad_form(px, py, mu2d, conic)
    assert Q.item() > k
    assert forward.poly_splat_wgeo(Q, k).item() == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
