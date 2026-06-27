"""MCMC tests: the WSR contribution-preserving relocation (o->o/n split, EXACT), SGLD noise
gating, and that the full MCMC loop trains."""
import _bootstrap  # noqa: F401
import torch

from spike import mcmc, synthetic
from spike.model import GaussianModel
from spike.render import render

DT = torch.float64


def _logit(p):
    return torch.log(torch.tensor(p, dtype=DT) / (1 - torch.tensor(p, dtype=DT)))


def _make_all_identical(m):
    with torch.no_grad():
        for name in ("means3d", "log_scales", "quats", "color_dc", "opacity_sh"):
            t = getattr(m, name)
            t[:] = t[0:1]


def test_split_preserves_render_exactly():
    """The claim: n coincident copies @ o/n render identically to 1 gaussian @ o (WSR)."""
    cam = synthetic.make_orbit_cameras(n=1, res=24, dtype=DT)[0]
    o0, n = 0.6, 4

    base = GaussianModel(n, seed=3, dtype=DT)
    _make_all_identical(base)
    with torch.no_grad():
        base.opacity_raw[:] = _logit(o0 / n)          # n copies, each o0/n
    C_multi = render(base, cam, "A")

    single = GaussianModel(1, seed=3, dtype=DT)
    with torch.no_grad():                              # copy base's gaussian 0 exactly (no cross-G RNG assumption)
        single.means3d[:] = base.means3d[0:1]
        single.log_scales[:] = base.log_scales[0:1]
        single.quats[:] = base.quats[0:1]
        single.color_dc[:] = base.color_dc[0:1]
        single.opacity_sh[:] = base.opacity_sh[0:1]
        single.w_b_raw.copy_(base.w_b_raw)
        single.opacity_raw[:] = _logit(o0)
    C_single = render(single, cam, "A")

    assert torch.allclose(C_single, C_multi, atol=1e-10), (C_single - C_multi).abs().max().item()

    # negative control: a WRONG split (copies kept at o0, not o0/n) MUST change the render,
    # else the test would trivially pass regardless of the split formula.
    with torch.no_grad():
        base.opacity_raw[:] = _logit(o0)
    C_wrong = render(base, cam, "A")
    assert (C_wrong - C_single).abs().max().item() > 1e-3, "negative control: test does not discriminate"


def test_relocate_keeps_render():
    """A full relocate (offset=0) on a scene with near-dead gaussians leaves the render ~unchanged."""
    cam = synthetic.make_orbit_cameras(n=1, res=24, dtype=DT)[0]
    m = GaussianModel(40, extent=0.6, seed=5, dtype=DT, gt=True)
    with torch.no_grad():
        m.opacity_raw[:12] = _logit(1e-4)              # inject 12 dead gaussians
    before = render(m, cam, "A")
    moved, touched = mcmc.relocate(m, dead_thr=0.005, offset=0.0)
    after = render(m, cam, "A")
    assert moved == 12, moved
    assert touched.numel() > 0
    assert (before - after).abs().max().item() < 2e-2, (before - after).abs().max().item()


def test_sgld_noise_is_opacity_gated():
    m = GaussianModel(2, seed=1, dtype=DT)
    with torch.no_grad():
        m.opacity_raw[0] = _logit(0.99)                # settled -> ~no noise
        m.opacity_raw[1] = _logit(1e-3)                # near-dead -> noise
    before = m.means3d.detach().clone()
    mcmc.add_sgld_noise(m, lr=5e-3, noise_lr=100.0)
    disp = (m.means3d.detach() - before).norm(dim=1)
    assert disp[0].item() < 1e-6, disp[0].item()
    assert disp[1].item() > disp[0].item()


def test_mcmc_loop_trains():
    """Full loop (Adam + SGLD + relocation + reg) reduces loss on a synthetic scene."""
    gt, cams, imgs = synthetic.make_scene(arm="A", seed=1, G_gt=15, n_views=3, res=24)
    model = GaussianModel(60, extent=0.6, seed=7)      # fp32 default
    hist, relocations = mcmc.train_mcmc(model, cams, imgs, "A", iters=80, relocate_every=30,
                                        noise_lr=50.0, offset=0.005)
    assert hist[-1] < hist[0], (hist[0], hist[-1])
    assert min(hist) < 0.6 * hist[0], (hist[0], min(hist))
    assert relocations >= 0


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
