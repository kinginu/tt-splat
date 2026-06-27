"""3DGS-MCMC (fixed gaussian count) for the matrix-native blend: SGLD noise, contribution-preserving
relocation (the trivial o->o/n split for WSR), opacity/scale regularization,
and an MCMC training loop. No densify/prune/resize; relocation recycles dead slots in place.
"""
import math

import torch

from . import geometry, metrics, train
from .render import render


def op_sigmoid(x, k=100.0, x0=0.995):
    """Sharp gate ~1 only for x>x0 (i.e. opacity < ~0.005), used to noise only near-dead gaussians."""
    return torch.sigmoid(k * (x - x0))


def reg_loss(model, lambda_o=0.01, lambda_s=0.01):
    """Opacity-L1 (drives low-contributors to dead) + scale-L1 (discourages oversized gaussians)."""
    o = torch.sigmoid(model.opacity_raw)
    s = torch.exp(model.log_scales)
    return lambda_o * o.abs().mean() + lambda_s * s.abs().mean()


@torch.no_grad()
def add_sgld_noise(model, lr, noise_lr=5e5):
    """Covariance-shaped Langevin noise on means, gated to low-opacity gaussians."""
    o = torch.sigmoid(model.opacity_raw)                       # [G]
    s = torch.exp(model.log_scales)                            # [G,3]
    R = geometry.quat_to_rotmat(model.quats)                   # [G,3,3]
    L = R * s[:, None, :]                                      # covariance sqrt (L Lᵀ = Σ)
    eps = torch.randn_like(model.means3d)                      # [G,3]
    weight = op_sigmoid(1.0 - o) * (noise_lr * lr)             # [G]
    model.means3d.add_(torch.einsum("gij,gj->gi", L, eps) * weight[:, None])


def _logit(p):
    return torch.log(p / (1.0 - p))


@torch.no_grad()
def relocate(model, dead_thr=0.005, offset=0.0):
    """Move dead (o<thr) slots onto live gaussians sampled ∝ opacity. Contribution-preserving via the
    WSR o->o/n split. Returns (n_moved, touched_indices) for optimizer-state reset."""
    o = torch.sigmoid(model.opacity_raw)
    dead = torch.where(o < dead_thr)[0]
    alive = torch.where(o >= dead_thr)[0]
    if dead.numel() == 0 or alive.numel() == 0:
        return 0, torch.empty(0, dtype=torch.long, device=model.means3d.device)

    probs = o[alive] / o[alive].sum()
    targets = alive[torch.multinomial(probs, dead.numel(), replacement=True)]   # one target per dead
    uniq, counts = torch.unique(targets, return_counts=True)
    n_at = {int(t): int(c) + 1 for t, c in zip(uniq, counts)}                   # incl. the original

    new_o = {int(t): (o[int(t)] / n_at[int(t)]).clamp(1e-6, 1 - 1e-6) for t in uniq}
    for t in uniq:                                                              # split the targets
        model.opacity_raw[int(t)] = _logit(new_o[int(t)])
    for d, t in zip(dead.tolist(), targets.tolist()):                           # copy onto dead slots
        model.means3d[d] = model.means3d[t] + offset * torch.randn(
            3, dtype=model.means3d.dtype, device=model.means3d.device)
        model.log_scales[d] = model.log_scales[t]
        model.quats[d] = model.quats[t]
        model.color_dc[d] = model.color_dc[t]
        model.opacity_sh[d] = model.opacity_sh[t]
        model.opacity_raw[d] = _logit(new_o[t])

    touched = torch.unique(torch.cat([dead, uniq]))
    return dead.numel(), touched


_PER_GAUSSIAN = ("means3d", "log_scales", "quats", "opacity_raw", "color_dc", "opacity_sh")


@torch.no_grad()
def reset_adam_state(opt, model, idx):
    """Zero Adam moments for relocated gaussian indices (per-index, not a full rebuild)."""
    if idx.numel() == 0:
        return
    for name in _PER_GAUSSIAN:
        p = getattr(model, name)
        st = opt.state.get(p)
        if st and "exp_avg" in st:
            st["exp_avg"][idx] = 0
            st["exp_avg_sq"][idx] = 0


def train_mcmc(model, cameras, gt_images, arm, iters=600, lr=None, relocate_every=100,
               dead_thr=0.005, offset=0.005, lambda_o=0.01, lambda_s=0.01, noise_lr=5e5,
               k=4.0, blur_eps=0.3, near=0.2, log_every=0):
    """MCMC fixed-count training. Returns (loss_history, total_relocations)."""
    lr = {**train.DEFAULT_LR, **(lr or {})}
    opt = torch.optim.Adam(model.param_groups(lr))
    history, relocations = [], 0
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        photo = 0.0
        for cam, gt in zip(cameras, gt_images):
            photo = photo + metrics.loss_fn(render(model, cam, arm, k=k, blur_eps=blur_eps, near=near), gt)
        loss = photo / len(cameras) + reg_loss(model, lambda_o, lambda_s)
        loss.backward()
        opt.step()
        add_sgld_noise(model, lr["means"], noise_lr)
        if relocate_every and it > 0 and it % relocate_every == 0:
            moved, touched = relocate(model, dead_thr, offset)
            if moved:
                reset_adam_state(opt, model, touched)
                relocations += moved
        history.append(float(loss.detach()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    iter {it:4d}  loss {history[-1]:.5f}  relocations {relocations}")
    return history, relocations
