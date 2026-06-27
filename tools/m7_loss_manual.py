"""HOST oracle: manual loss gradient dL/dimg for (1-λ)·L1 + λ·(1-SSIM), with the SSIM
Gaussian window expressed as BANDED-MATRIX GEMMs (device-ready form: filter(x) = M_h x M_wᵀ per channel,
backward = M_hᵀ g M_w). No autograd. Verified against (a) pytorch_msssim loss_fn (forward) and
(b) torch autograd (backward) so the derivation is correct before porting to ttnn.

Why GEMM-form: the 11×11 separable Gaussian on a fixed-size image is a banded [Hout×Hin] matrix applied
along H then a [Wout×Win] along W — both GEMMs (matrix engine), no conv2d needed on device.

Run (CPU, no device):  podman-compose --profile hw run --rm hw python3 tools/m7_loss_manual.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
import torch.nn.functional as F
from spike import metrics

C1 = 0.01 ** 2
C2 = 0.03 ** 2
WIN = 11
SIGMA = 1.5
LAMBDA = 0.2


def gauss_1d(win=WIN, sigma=SIGMA):
    c = torch.arange(win, dtype=torch.float64) - win // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def band_matrix(n_in, g):
    """[n_out, n_in] valid-conv banded matrix: row o = g placed at columns o..o+win-1."""
    win = g.numel()
    n_out = n_in - win + 1
    M = torch.zeros(n_out, n_in, dtype=torch.float64)
    for o in range(n_out):
        M[o, o:o + win] = g
    return M


def filt(x, Mh, Mw):
    """x [C,H,W] -> [C,Hout,Wout] separable Gaussian via GEMMs: Mh @ x @ Mwᵀ per channel."""
    # along H: [Hout,H]@[C,H,W] -> [C,Hout,W]
    t = torch.einsum('oh,chw->cow', Mh, x)
    # along W: [C,Hout,W]@[Wout,W]ᵀ -> [C,Hout,Wout]
    return torch.einsum('cow,vw->cov', t, Mw)


def filt_T(g, Mh, Mw):
    """transpose of filt: [C,Hout,Wout] -> [C,H,W] via Mhᵀ g Mw."""
    t = torch.einsum('oh,cov->chv', Mh, g)        # Mhᵀ @ g : sum over Hout (o) -> Hin (h)
    return torch.einsum('chv,vw->chw', t, Mw)     # @ Mw   : sum over Wout (v) -> Win (w)


def ssim_fwd(x, y, Mh, Mw):
    """Returns S (ssim map [C,Hout,Wout]) + the intermediates needed for the manual backward."""
    fx = filt(x, Mh, Mw)
    fy = filt(y, Mh, Mw)
    fx2 = filt(x * x, Mh, Mw)
    fy2 = filt(y * y, Mh, Mw)
    fxy = filt(x * y, Mh, Mw)
    mux, muy = fx, fy
    mux2, muy2 = mux * mux, muy * muy
    sx = fx2 - mux2
    sy = fy2 - muy2
    sxy = fxy - mux * muy
    A1 = 2 * mux * muy + C1
    A2 = 2 * sxy + C2
    B1 = mux2 + muy2 + C1
    B2 = sx + sy + C2
    S = (A1 * A2) / (B1 * B2)
    return S, dict(fx=fx, muy=muy, mux=mux, A1=A1, A2=A2, B1=B1, B2=B2)


def loss_manual(x, y, Mh, Mw):
    """(1-λ)L1 + λ(1-mean S); returns (loss, dL/dx) with dL/dx fully manual."""
    N = x.numel()
    # L1
    l1 = (x - y).abs().mean()
    g_l1 = (1.0 - LAMBDA) * torch.sign(x - y) / N

    # SSIM
    S, c = ssim_fwd(x, y, Mh, Mw)
    Ns = S.numel()
    ssim = S.mean()

    mux, muy, A1, A2, B1, B2 = c['mux'], c['muy'], c['A1'], c['A2'], c['B1'], c['B2']
    D = B1 * B2
    # partial dS w.r.t the three conv outputs fx, fx2, fxy (y is constant)
    dS_dfx2 = -S / B2
    dS_dfxy = 2 * A1 / D
    dS_dfx = (2.0 / D) * (muy * (A2 - A1) - S * mux * (B2 - B1))

    # d(mean S)/dx_i = (1/Ns)[ filtᵀ(dS_dfx) + 2x·filtᵀ(dS_dfx2) + y·filtᵀ(dS_dfxy) ]
    t1 = filt_T(dS_dfx, Mh, Mw)
    t2 = filt_T(dS_dfx2, Mh, Mw)
    t3 = filt_T(dS_dfxy, Mh, Mw)
    dmeanS_dx = (t1 + 2 * x * t2 + y * t3) / Ns
    g_ssim = LAMBDA * (-dmeanS_dx)

    loss = (1.0 - LAMBDA) * l1 + LAMBDA * (1.0 - ssim)
    return loss, g_l1 + g_ssim


def main():
    torch.manual_seed(0)
    res = 96
    # random-ish images in [0,1] (a real render+gt would do; structure is what matters for grad check)
    x = torch.rand(res, res, 3, dtype=torch.float64).clamp(0, 1)
    y = torch.rand(res, res, 3, dtype=torch.float64).clamp(0, 1)

    g = gauss_1d()
    Mh = band_matrix(res, g)
    Mw = band_matrix(res, g)

    # ---- forward: my SSIM vs pytorch_msssim ----
    Sx = x.permute(2, 0, 1).contiguous()
    Sy = y.permute(2, 0, 1).contiguous()
    S, _ = ssim_fwd(Sx, Sy, Mh, Mw)
    my_ssim = S.mean().item()
    ref_ssim = metrics.ssim(x.float(), y.float()).item()
    print(f"SSIM  mine={my_ssim:.6f}  pytorch_msssim={ref_ssim:.6f}  |Δ|={abs(my_ssim-ref_ssim):.2e}")

    my_loss, g_manual = loss_manual(Sx, Sy, Mh, Mw)
    ref_loss = metrics.loss_fn(x.float(), y.float(), lambda_ssim=LAMBDA).item()
    print(f"LOSS  mine={my_loss.item():.6f}  loss_fn={ref_loss:.6f}  |Δ|={abs(my_loss.item()-ref_loss):.2e}")

    # ---- backward: manual vs autograd of MY loss (exact) and of the REAL loss_fn ----
    xa = Sx.clone().requires_grad_(True)
    Sa, _ = ssim_fwd(xa, Sy, Mh, Mw)
    la = (1.0 - LAMBDA) * (xa - Sy).abs().mean() + LAMBDA * (1.0 - Sa.mean())
    la.backward()
    g_auto_mine = xa.grad
    rel_mine = ((g_manual - g_auto_mine).norm() / g_auto_mine.norm()).item()
    print(f"grad  manual vs autograd(my loss)   rel={rel_mine:.3e}  {'PASS' if rel_mine < 1e-9 else 'FAIL'}")

    # vs autograd of the actual training loss_fn (pytorch_msssim) -- validates we match what we train on
    xb = x.float().clone().requires_grad_(True)
    lb = metrics.loss_fn(xb, y.float(), lambda_ssim=LAMBDA)
    lb.backward()
    g_auto_real = xb.grad.permute(2, 0, 1).contiguous().double()
    rel_real = ((g_manual - g_auto_real).norm() / g_auto_real.norm()).item()
    print(f"grad  manual vs autograd(loss_fn)   rel={rel_real:.3e}  "
          f"{'PASS' if rel_real < 1e-3 else 'FAIL (SSIM impl differs)'}")


if __name__ == "__main__":
    main()
