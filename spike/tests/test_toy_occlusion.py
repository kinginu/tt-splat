"""Test 6 (PASS/FAIL): the load-bearing occlusion probe. Two gaussians at the
same screen position with equal footprint, near=red / far=blue. Depth-free WSR (arm A)
*averages* them (purple) — it cannot occlude; sorted alpha-exp (arm D) shows the front
(red). This is exactly the failure the spike exists to measure."""
import math

import _bootstrap  # noqa: F401
import torch

from spike import arms, forward

DT = torch.float64


def _f(x):
    return torch.tensor(x, dtype=DT)


def _logit(p):
    return math.log(p / (1 - p))


def _setup():
    k = 4.0
    mu2d = _f([[0.0, 0.0], [0.0, 0.0]])      # identical screen position
    conic = _f([[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]])   # equal footprint
    red = _f([1.0, 0.0, 0.0])
    blue = _f([0.0, 0.0, 1.0])
    color = torch.stack([red, blue])         # gaussian 0 = red (near), 1 = blue (far)
    depth = _f([2.0, 5.0])
    opacity_raw = _f([_logit(0.9), _logit(0.9)])
    w_b = _f(1e-6)
    c_b = _f([1.0, 1.0, 1.0])
    px, py = _f([0.0]), _f([0.0])            # center pixel: both Q=0
    Q = forward.quad_form(px, py, mu2d, conic)
    w_geo = forward.poly_splat_wgeo(Q, k)
    assert torch.allclose(w_geo, torch.ones_like(w_geo))
    return dict(Q=Q, w_geo=w_geo, color=color, depth=depth, opacity_raw=opacity_raw,
                w_b=w_b, c_b=c_b, red=red, blue=blue)


def test_armA_averages():
    s = _setup()
    C = arms.blend_A(s["w_geo"], s["opacity_raw"], s["color"], s["w_b"], s["c_b"])[0]
    # depth-free WSR -> color average -> purple, NOT red. (occlusion not modeled)
    assert torch.allclose(C, 0.5 * (s["red"] + s["blue"]), atol=1e-3), C


def test_armD_occludes():
    s = _setup()
    C = arms.render_D(s["Q"], s["opacity_raw"], s["color"], s["depth"], s["c_b"])[0]
    # front (red) dominates; closer to red than blue.
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    # quantitative: ~0.9*red + 0.09*blue + 0.01*white
    assert C[0].item() > 0.85 and C[2].item() < 0.12, C


def test_depth_order_matters():
    # swap which gaussian is in front; arm D must follow the depth order.
    s = _setup()
    depth_swapped = _f([5.0, 2.0])           # now blue (idx 1) is near
    C = arms.render_D(s["Q"], s["opacity_raw"], s["color"], depth_swapped, s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C


def test_softz_occludes():
    # soft-Z buys the occlusion arm A cannot: per-pixel zstar attenuates the far gaussian,
    # so the front (red) dominates like arm D -- NOT the purple average of arm A.
    s = _setup()
    beta = _f(4.0)
    C = arms.blend_SZ(s["w_geo"], s["opacity_raw"], s["depth"], beta,
                      s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.85 and C[2].item() < 0.15, C    # front-dominated, like a sorted blend
    # sharper gate -> harder occlusion (monotone in beta)
    C2 = arms.blend_SZ(s["w_geo"], s["opacity_raw"], s["depth"], _f(12.0),
                       s["color"], s["w_b"], s["c_b"])[0]
    assert C2[0].item() > C[0].item(), (C, C2)


def test_softz_depth_order():
    # soft-Z must follow depth order: swap front/back -> blue wins.
    s = _setup()
    depth_swapped = _f([5.0, 2.0])           # blue (idx 1) now in front
    C = arms.blend_SZ(s["w_geo"], s["opacity_raw"], depth_swapped, _f(4.0),
                      s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C


def test_softmin_occludes():
    s = _setup(); tau = _f(0.8)
    C = arms.blend_SM(s["w_geo"], s["opacity_raw"], s["depth"], tau, s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.85 and C[2].item() < 0.15, C
    # smaller tau -> harder occlusion (monotone)
    C2 = arms.blend_SM(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.3), s["color"], s["w_b"], s["c_b"])[0]
    assert C2[0].item() >= C[0].item()


def test_softmin_depth_order():
    s = _setup(); zsw = _f([5.0, 2.0])
    C = arms.blend_SM(s["w_geo"], s["opacity_raw"], zsw, _f(0.8), s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C


def test_softmin_zgrad():
    s = _setup(); z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_SM(s["w_geo"], s["opacity_raw"], z, _f(0.8), s["color"], s["w_b"], s["c_b"])
    (g,) = torch.autograd.grad(C.sum(), z)
    assert g.abs().sum() > 0
    # the FAR gaussian must feel a push (its rho<1, so dL/dz_far != 0)
    assert g[1].abs().item() > 0


def test_bprime_occludes():
    s = _setup()
    C = arms.blend_BP(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.5), s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.85 and C[2].item() < 0.15, C
    C2 = arms.blend_BP(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.2), s["color"], s["w_b"], s["c_b"])[0]
    assert C2[0].item() >= C[0].item()   # smaller tau -> harder


def test_bprime_depth_order():
    s = _setup()
    C = arms.blend_BP(s["w_geo"], s["opacity_raw"], _f([5.0, 2.0]), _f(0.5), s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C


def test_bprime_zgrad():
    s = _setup()
    z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_BP(s["w_geo"], s["opacity_raw"], z, _f(0.5), s["color"], s["w_b"], s["c_b"])
    (g,) = torch.autograd.grad(C.sum(), z)
    assert g.abs().sum() > 0


def test_gumbel_occludes():
    s = _setup()
    g = torch.Generator().manual_seed(0)
    C = arms.blend_E(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.5), s["color"], s["w_b"], s["c_b"], S=256, gen=g)[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.75, C                               # high-variance -> loose bound


def test_gumbel_depth_order():
    s = _setup()
    g = torch.Generator().manual_seed(0)
    C = arms.blend_E(s["w_geo"], s["opacity_raw"], _f([5.0, 2.0]), _f(0.5), s["color"], s["w_b"], s["c_b"], S=256, gen=g)[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C


def test_gumbel_zgrad():
    s = _setup()
    z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_E(s["w_geo"], s["opacity_raw"], z, _f(0.8), s["color"], s["w_b"], s["c_b"], S=8)
    (g,) = torch.autograd.grad(C.sum(), z)
    assert g.abs().sum() > 0


def test_moment_occludes():
    s = _setup()
    C = arms.blend_MO(s["w_geo"], s["opacity_raw"], s["depth"], s["color"], s["w_b"], s["c_b"], m=4)[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.80 and C[2].item() < 0.20, C    # moment recon approximate; looser bound


def test_moment_depth_order():
    s = _setup()
    C = arms.blend_MO(s["w_geo"], s["opacity_raw"], _f([5.0, 2.0]), s["color"], s["w_b"], s["c_b"], m=4)[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C


def test_moment_zgrad():
    s = _setup()
    z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_MO(s["w_geo"], s["opacity_raw"], z, s["color"], s["w_b"], s["c_b"], m=4)
    (g,) = torch.autograd.grad(C.sum(), z)
    assert g.abs().sum() > 0


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
