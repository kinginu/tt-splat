"""Device loss gradient dL/dimg on the Blackhole, verified vs the host oracle
(tools/loss_manual.py, itself matched to the training loss_fn). Computes gC = dL/dimg for
(1-λ)·L1 + λ·(1-SSIM) entirely on device so the resident trainer never round-trips C/gC through host.

Design: the SSIM 11×11 separable Gaussian is banded-matrix GEMMs (filt = Mh·x·Mwᵀ, backward = Mhᵀ·g·Mw);
y (gt) is constant per view so its filtered maps (muy, muy2, sy) are precomputed host-side and uploaded.
Only the grad is needed (not the loss value) → no device reduction; the 1/Ns and 1/N scales are host scalars.

Run inside the hw container (needs the device free):
    podman-compose --profile hw run --rm hw python3 tools/loss_device.py --res 96
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from loss_manual import gauss_1d, band_matrix, ssim_fwd, loss_manual, C1, C2, LAMBDA

DEV = None
DT = ttnn.float32


def u(t):
    return ttnn.from_torch(t.contiguous().float(), dtype=DT, layout=ttnn.TILE_LAYOUT, device=DEV)


def dn(t):
    return ttnn.to_torch(t).float()


def mm(a, b):
    return ttnn.matmul(a, b)


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    args = ap.parse_args()
    res = args.res

    torch.manual_seed(0)
    x_h = torch.rand(3, res, res, dtype=torch.float64).clamp(0, 1)   # [3,H,W] (channel-batched)
    y_h = torch.rand(3, res, res, dtype=torch.float64).clamp(0, 1)

    g = gauss_1d()
    Mh = band_matrix(res, g)        # [Hout, Hin]
    Mw = band_matrix(res, g)        # [Wout, Win]
    Hout, Win = Mh.shape[0], Mw.shape[1]

    # ---- host oracle grad (== autograd of the real loss_fn, verified in step 1) ----
    _, g_ref = loss_manual(x_h, y_h, Mh, Mw)     # [3,H,W]

    # ---- precompute y-only (constant) maps on host, upload (muy, muy2, sy) ----
    from loss_manual import filt as hfilt
    muy = hfilt(y_h, Mh, Mw)
    muy2 = muy * muy
    sy = hfilt(y_h * y_h, Mh, Mw) - muy2

    DEV = ttnn.open_device(device_id=0)
    try:
        # constant device tensors
        Mh_b = u(Mh.unsqueeze(0).expand(3, *Mh.shape))      # [3,Hout,Hin]
        MwT_b = u(Mw.t().unsqueeze(0).expand(3, Win, Mw.shape[0]))   # [3,Win,Wout]
        MhT_b = u(Mh.t().unsqueeze(0).expand(3, Mh.shape[1], Hout))  # [3,Hin,Hout]
        Mw_b = u(Mw.unsqueeze(0).expand(3, *Mw.shape))      # [3,Wout,Win]
        x = u(x_h)
        y = u(y_h)
        muy_d, muy2_d, sy_d = u(muy), u(muy2), u(sy)

        def filt(t):     # Mh · t · Mwᵀ  -> [3,Hout,Wout]
            return mm(mm(Mh_b, t), MwT_b)

        def filt_T(t):   # Mhᵀ · t · Mw  -> [3,Hin,Win]
            return mm(mm(MhT_b, t), Mw_b)

        # ---- SSIM forward on device (x-dependent maps) ----
        fx = filt(x)
        fx2 = filt(ttnn.mul(x, x))
        fxy = filt(ttnn.mul(x, y))
        mux = fx
        mux2 = ttnn.mul(mux, mux)
        sx = ttnn.sub(fx2, mux2)
        sxy = ttnn.sub(fxy, ttnn.mul(mux, muy_d))
        A1 = ttnn.add(ttnn.mul(ttnn.mul(mux, muy_d), 2.0), C1)
        A2 = ttnn.add(ttnn.mul(sxy, 2.0), C2)
        B1 = ttnn.add(ttnn.add(mux2, muy2_d), C1)
        B2 = ttnn.add(ttnn.add(sx, sy_d), C2)
        D = ttnn.mul(B1, B2)
        S = ttnn.div(ttnn.mul(A1, A2), D)

        # ---- partial derivs (mirror the host oracle) ----
        dS_dfx2 = ttnn.neg(ttnn.div(S, B2))
        dS_dfxy = ttnn.div(ttnn.mul(A1, 2.0), D)
        # dS_dfx = (2/D)*( muy*(A2-A1) - S*mux*(B2-B1) )
        term = ttnn.sub(ttnn.mul(muy_d, ttnn.sub(A2, A1)),
                        ttnn.mul(ttnn.mul(S, mux), ttnn.sub(B2, B1)))
        dS_dfx = ttnn.mul(ttnn.div(term, D), 2.0)

        # ---- d(mean S)/dx = (1/Ns)[ filtᵀ(dS_dfx) + 2x·filtᵀ(dS_dfx2) + y·filtᵀ(dS_dfxy) ] ----
        Ns = float(S.shape[0] * S.shape[1] * S.shape[2])
        t1 = filt_T(dS_dfx)
        t2 = filt_T(dS_dfx2)
        t3 = filt_T(dS_dfxy)
        dmeanS_dx = ttnn.div(ttnn.add(ttnn.add(t1, ttnn.mul(ttnn.mul(x, t2), 2.0)),
                                      ttnn.mul(y, t3)), Ns)
        g_ssim = ttnn.mul(dmeanS_dx, -LAMBDA)

        # ---- L1 grad: (1-λ)·sign(x-y)/N ----
        N = float(x_h.numel())
        g_l1 = ttnn.mul(ttnn.sign(ttnn.sub(x, y)), (1.0 - LAMBDA) / N)

        gC = ttnn.add(g_l1, g_ssim)
        g_dev = dn(gC)

        # verify
        rel = ((g_dev.double() - g_ref).norm() / g_ref.norm()).item()
        # also check SSIM map matches (forward sanity)
        S_ref, _ = ssim_fwd(x_h, y_h, Mh, Mw)
        rel_S = ((dn(S).double() - S_ref).norm() / S_ref.norm()).item()
        print(f"== device loss-grad  res={res} ==")
        print(f"   SSIM map rel vs oracle: {rel_S:.3e}")
        print(f"   dL/dimg  rel vs oracle: {rel:.3e}   "
              f"{'PASS' if rel < 2e-2 else 'FAIL'}  (fp32 device; bf16-floor tol 2e-2)")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
