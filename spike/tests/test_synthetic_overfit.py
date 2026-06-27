"""End-to-end self-consistency oracle (data-free): render a known gaussian scene under an
arm, then overfit a fresh random model to those images and confirm recovery. Validates
render + autograd + Adam + arms working together, with zero external data."""
import _bootstrap  # noqa: F401
import torch

from spike import metrics, synthetic, train
from spike.model import GaussianModel
from spike.render import render, ARMS


def test_render_arms_finite():
    gt = synthetic.make_gt_model(seed=1, G=30)
    cam = synthetic.make_orbit_cameras(n=1, res=32)[0]
    for arm in ARMS:
        img = render(gt, cam, arm)
        assert img.shape == (32, 32, 3), (arm, img.shape)
        assert torch.isfinite(img).all(), arm
        assert img.min().item() >= -1e-4 and img.max().item() < 5.0, (arm, img.min(), img.max())


def test_loss_decreases():
    # cheap sanity: a few Adam steps strictly reduce the loss.
    gt, cams, imgs = synthetic.make_scene(arm="A", seed=1, G_gt=20, n_views=3, res=32)
    model = GaussianModel(60, extent=0.6, seed=7, gt=False)
    hist = train.fit(model, cams, imgs, "A", iters=40)
    assert hist[-1] < hist[0], (hist[0], hist[-1])
    assert min(hist) < 0.5 * hist[0], (hist[0], min(hist))


def test_overfit_recovers():
    torch.manual_seed(0)
    arm = "A"
    gt, cams, imgs = synthetic.make_scene(arm=arm, seed=1, G_gt=30, n_views=4, res=40)
    model = GaussianModel(150, extent=0.7, seed=2, gt=False)

    psnr0 = train.eval_psnr(model, cams, imgs, arm)
    hist = train.fit(model, cams, imgs, arm, iters=400, log_every=0)
    psnr1 = train.eval_psnr(model, cams, imgs, arm)
    print(f"    [overfit] loss {hist[0]:.4f} -> {hist[-1]:.4f}   PSNR {psnr0:.2f} -> {psnr1:.2f} dB")

    assert hist[-1] < 0.3 * hist[0], (hist[0], hist[-1])     # loss collapsed
    assert psnr1 > psnr0 + 8.0, (psnr0, psnr1)               # clear improvement
    assert psnr1 > 24.0, psnr1                               # absolute recovery


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
