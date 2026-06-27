"""Image metrics + the training loss for the matrix-native reference. Images are [H,W,3] in [0,1]."""
import torch
from pytorch_msssim import ssim as _ssim


def _nchw(img):
    return img.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)


def psnr(pred, gt):
    mse = (pred.clamp(0.0, 1.0) - gt.clamp(0.0, 1.0)).pow(2).mean()
    return -10.0 * torch.log10(mse.clamp(min=1e-12))


def ssim(pred, gt, win_size=11):
    win = min(win_size, pred.shape[0], pred.shape[1])
    if win % 2 == 0:
        win -= 1
    return _ssim(_nchw(pred), _nchw(gt), data_range=1.0, win_size=max(win, 3))


def l1(pred, gt):
    return (pred - gt).abs().mean()


def loss_fn(pred, gt, lambda_ssim=0.2):
    """(1-λ)·L1 + λ·(1-SSIM), the standard 3DGS photometric loss."""
    return (1.0 - lambda_ssim) * l1(pred, gt) + lambda_ssim * (1.0 - ssim(pred, gt))
