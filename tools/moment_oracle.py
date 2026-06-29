"""Oracle: forward fidelity of moment-based OIT reconstruction vs exact sorted transmittance.
Reports mean/max abs error in T for m in {2,4,6} and asserts m=4 mean < 0.05. PASS/FAIL exit."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import torch
from spike import arms

torch.manual_seed(42)
G = 16; P = 50
w_geo = torch.ones(P, G, dtype=torch.float64) * 0.5  # equal footprint
opacity_raw = torch.randn(G, dtype=torch.float64) * 0.5  # moderate opacity
z = torch.rand(G, dtype=torch.float64) * 6.0 + 0.5   # depths [0.5, 6.5]
color = torch.rand(G, 3, dtype=torch.float64)
w_b = torch.tensor(1e-6, dtype=torch.float64)
c_b = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

# Exact transmittance: T_i = prod_{j: z_j < z_i} (1 - alpha_j) -- sorted, brute force
o = torch.sigmoid(opacity_raw)
alpha = (o[None, :] * w_geo).clamp(1e-6, 1.0 - 1e-4)  # [P,G]
order = torch.argsort(z)                                 # sort once for reference


def exact_transmittance(alpha, order):
    """Exact front-to-back transmittance T_i = Prod_{j in front}(1-alpha_j). Returns [P,G]."""
    G = alpha.shape[1]
    P = alpha.shape[0]
    T_exact = torch.zeros(P, G, dtype=alpha.dtype)
    for p in range(P):
        t = 1.0
        for gi in order:
            T_exact[p, gi] = t
            t = t * (1.0 - alpha[p, gi].item())
    return T_exact


T_ref = exact_transmittance(alpha, order)   # [P, G]

all_ok = True
for m in [2, 4, 6]:
    # get MO's transmittance
    zw = arms._depth_warp(z)
    zp = torch.stack([zw ** n for n in range(m + 1)], dim=-1)
    a = -torch.log1p(-alpha)
    b = (a @ zp).double()
    A_frac = arms._moment_reconstruct(b, zw, m=m)
    T_mo = torch.exp(-b[:, :1] * A_frac)

    err = (T_mo - T_ref).abs()
    print(f"m={m}: mean_err={err.mean():.4f}  max_err={err.max():.4f}")
    if m == 4 and err.mean() > 0.05:
        print(f"FAIL m=4 mean error {err.mean():.4f} > 0.05")
        all_ok = False

print("ORACLE", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
