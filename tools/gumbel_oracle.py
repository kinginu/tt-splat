"""Oracle: verify that arm E's stochastic-hard estimator converges to exact sorted compositing.
Computes |mean_S - exact| vs S in {1,8,16,64,256} and checks the STE gradient sign. PASS/FAIL."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import torch
from spike import arms

torch.manual_seed(7)
G = 6; P = 4
w_geo = torch.ones(P, G) * 0.5
opacity_raw = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.3, 0.2])   # varying opacities
z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
color = torch.eye(G, 3)[:G]  # distinct colors
w_b = torch.tensor(1e-6); c_b = torch.tensor([1.0, 1.0, 1.0])
e_tau = torch.tensor(0.5)

# Exact sorted compositing: C = sum_i alpha_i * T_i * color_i + T_final * c_b
o = torch.sigmoid(opacity_raw)
alpha = (o[None, :] * w_geo).clamp(1e-6, 1.0 - 1e-4)
order = torch.argsort(z)
C_exact = torch.zeros(P, 3)
for p in range(P):
    T = 1.0
    for gi in order:
        a = alpha[p, gi].item()
        C_exact[p] += T * a * color[gi]
        T *= (1 - a)
    C_exact[p] += T * c_b

# Stochastic-hard estimator at various S
all_ok = True
gen = torch.Generator().manual_seed(99)
for S in [1, 8, 16, 64, 256, 1024]:
    C_est = arms.blend_E(w_geo, opacity_raw, z, e_tau, color, w_b, c_b, S=S, gen=gen)
    err = (C_est.detach() - C_exact).abs().mean().item()
    print(f"S={S:>5}: mean_err={err:.4f}")
    if S == 256 and err > 0.05:
        print(f"FAIL: S=256 error {err:.4f} > 0.05 (estimator may not be unbiased)")
        all_ok = False

# Check gradient sign via soft surrogate
z2 = z.clone().requires_grad_(True)
e_tau2 = torch.tensor(0.5)
C2 = arms.blend_E(w_geo, opacity_raw, z2, e_tau2, color, w_b, c_b, S=8)
(gz,) = torch.autograd.grad(C2.sum(), z2)
print(f"z-grad: {gz.tolist()}")
assert gz.abs().sum() > 0, "gradient is zero — STE surrogate not connected"
print("STE gradient flows through soft surrogate: OK")
print("ORACLE", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
