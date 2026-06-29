"""The overfit loop: fit a GaussianModel to a set of (camera, gt_image) pairs under
one arm, with Adam + the photometric loss. Used by both the synthetic self-consistency
oracle and (later) the real ficus overfit."""
import torch

from . import metrics
from .render import render

DEFAULT_LR = {
    "means": 5e-3, "scales": 5e-3, "quats": 1e-3, "opacity": 5e-2,
    "opacity_sh": 5e-2, "color": 1e-2, "wb": 1e-2, "depth": 1e-2,
}


def fit(model, cameras, gt_images, arm, iters=800, lambda_ssim=0.2, lr=None,
        k=4.0, blur_eps=0.3, near=0.2, log_every=0, sh_degree=0):
    """Returns the per-iteration mean loss history (list of floats)."""
    lr = {**DEFAULT_LR, **(lr or {})}
    opt = torch.optim.Adam(model.param_groups(lr))
    history = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for cam, gt in zip(cameras, gt_images):
            img = render(model, cam, arm, k=k, blur_eps=blur_eps, near=near, sh_degree=sh_degree)
            total = total + metrics.loss_fn(img, gt, lambda_ssim)
        total = total / len(cameras)
        total.backward()
        opt.step()
        history.append(float(total.detach()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    iter {it:4d}  loss {history[-1]:.5f}")
    return history


@torch.no_grad()
def eval_psnr(model, cameras, gt_images, arm, **kw):
    """Mean PSNR over the given views."""
    vals = [float(metrics.psnr(render(model, cam, arm, **kw), gt))
            for cam, gt in zip(cameras, gt_images)]
    return sum(vals) / len(vals)
