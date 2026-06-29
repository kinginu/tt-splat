"""Oracle tests for the view-dependent SH colour (spike/sh.eval_sh_color + render sh_degree).

Gates: (1) zero-rest SH == DC-only colour at any degree (so default behaviour is unchanged);
(2) the render is identical at sh_degree=0 vs >0 when color_rest=0 (end-to-end no-op);
(3) finite-difference gradcheck of eval_sh_color w.r.t. coefficients (backward correct);
(4) non-zero rest actually makes colour view-dependent.
"""
import torch

from spike import sh, forward
from spike.render import render
from spike.model import GaussianModel
from spike.camera import look_at_opencv, Camera


def _cam(res=16):
    R_v, t_v = look_at_opencv((0.0, 0.0, 3.0), (0.0, 0.0, 0.0))
    f = 0.5 * res / 0.5
    return Camera(R_v, t_v, f, f, res / 2, res / 2, res, res)


def test_zero_rest_equals_dc():
    torch.manual_seed(0)
    dc = torch.randn(50, 3, dtype=torch.float64)
    coeffs = torch.cat([dc[:, None, :], torch.zeros(50, 15, 3, dtype=torch.float64)], dim=1)
    dirs = torch.randn(50, 3, dtype=torch.float64)
    for deg in (0, 1, 2, 3):
        c = sh.eval_sh_color(deg, coeffs, dirs)
        assert torch.allclose(c, forward.color_from_dc(dc), atol=1e-12), f"deg {deg} != DC"


def test_render_identical_when_rest_zero():
    m = GaussianModel(200, seed=1)            # fresh model: color_rest is zeros
    cam = _cam()
    a = render(m, cam, "A", sh_degree=0)
    b = render(m, cam, "A", sh_degree=3)
    assert torch.allclose(a, b, atol=1e-6), "sh_degree>0 with zero rest must equal DC render"


def test_eval_sh_gradcheck():
    torch.manual_seed(2)
    coeffs = torch.randn(8, 16, 3, dtype=torch.float64, requires_grad=True)
    dirs = torch.randn(8, 3, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda c: sh.eval_sh_color(3, c, dirs), (coeffs,), eps=1e-6, atol=1e-6)


def test_view_dependent_color_changes_with_direction():
    torch.manual_seed(3)
    coeffs = torch.zeros(4, 16, 3, dtype=torch.float64)
    coeffs[:, 0, :] = 0.5                       # DC
    coeffs[:, 1:, :] = torch.randn(4, 15, 3, dtype=torch.float64)   # non-trivial rest
    c1 = sh.eval_sh_color(3, coeffs, torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float64))
    c2 = sh.eval_sh_color(3, coeffs, torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float64))
    assert (c1 - c2).abs().max() > 1e-3, "colour must depend on viewing direction with non-zero rest"
