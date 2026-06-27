"""Verify the hand-derived analytic backward (poly-splat + WSR) against (1) finite-diff
gradcheck and (2) autograd, and confirm forward parity with arms.blend_A. fp64."""
import math

import _bootstrap  # noqa: F401
import torch

from spike import arms
from spike import backward as bw

DT = torch.float64


def _inputs(P=8, G=5, seed=0, cull=True):
    g = torch.Generator().manual_seed(seed)
    theta_u = torch.randn(G, 6, generator=g, dtype=DT, requires_grad=True)
    o = torch.rand(G, generator=g, dtype=DT, requires_grad=True)
    color = torch.rand(G, 3, generator=g, dtype=DT, requires_grad=True)
    w_b = torch.tensor(0.05, dtype=DT, requires_grad=True)
    px = torch.rand(P, generator=g, dtype=DT) * 4
    py = torch.rand(P, generator=g, dtype=DT) * 4
    Phi = torch.stack([px * px, py * py, px * py, px, py, torch.ones(P, dtype=DT)], dim=-1)
    c_b = torch.ones(3, dtype=DT)
    keep = torch.ones(G, dtype=DT)
    if cull:
        keep[0] = 0.0                         # one culled gaussian exercises the keep mask
    return theta_u, o, color, w_b, Phi, c_b, keep


def test_gradcheck_finite_difference():
    inp = _inputs()
    # finite-difference oracle: analytic grads must match numerical to fp64 tolerance
    assert torch.autograd.gradcheck(bw.WSRPolySplatHot.apply, inp, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_manual_matches_autograd():
    for seed in range(5):
        t, o, c, wb, Phi, cb, keep = _inputs(seed=seed)
        t2, o2, c2, wb2 = (x.detach().clone().requires_grad_(True) for x in (t, o, c, wb))
        Cref = bw.wsr_polysplat_ref(t2, o2, c2, wb2, Phi, cb, keep)   # autograd reference
        Cman = bw.wsr_polysplat(t, o, c, wb, Phi, cb, keep)           # manual backward
        assert torch.allclose(Cref, Cman, atol=1e-12), "forward mismatch"
        gout = torch.randn(Cref.shape, dtype=DT)
        Cref.backward(gout)
        Cman.backward(gout)
        for a, b, name in [(t.grad, t2.grad, "theta_u"), (o.grad, o2.grad, "o"),
                           (c.grad, c2.grad, "color"), (wb.grad, wb2.grad, "w_b")]:
            assert torch.allclose(a, b, atol=1e-9), (name, (a - b).abs().max().item())


def test_quotient_term_is_needed():
    # sanity that the gden (=-C/D) term is non-trivial: dropping it changes the gradient.
    t, o, c, wb, Phi, cb, keep = _inputs(cull=False)
    C = bw.wsr_polysplat(t, o, c, wb, Phi, cb, keep)
    C.sum().backward()
    assert t.grad.abs().sum() > 0 and c.grad.abs().sum() > 0


def test_forward_parity_blendA():
    t, o, c, wb, Phi, cb, _ = _inputs(cull=False)
    with torch.no_grad():
        u = Phi @ t.t()
        w_geo = u.clamp(min=0.0) ** 2
        opacity_raw = torch.log(o / (1 - o))          # so arm A's sigmoid recovers o
        Ca = arms.blend_A(w_geo, opacity_raw, c, wb, cb)
        Cm = bw.wsr_polysplat(t, o, c, wb, Phi, cb, None)
    assert torch.allclose(Ca, Cm, atol=1e-10)


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
