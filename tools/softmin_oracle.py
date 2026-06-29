"""Oracle: verify the analytic backward for the softmin (Boltzmann depth weight) arm SM.

Softmin per-gaussian weight:
    z_ref = min(z).detach()                    (detached front anchor)
    rho_g = exp(-(z_g - z_ref) / tau)
    W[p,g] = o_g * w_geo[p,g] * rho_g

Analytic grads (treat rho as the cut point; grad_rho_g = dL/drho_g from WSR autograd):
    d rho_g / d z_g  = -rho_g / tau
    d rho_g / d tau  =  rho_g * (z_g - z_ref) / tau^2

    grad_z_g   += grad_rho_g * (-rho_g / tau)
    grad_tau    = sum_g grad_rho_g * rho_g * (z_g - z_ref) / tau^2

Verified against autograd through blend_SM; rel tolerance < 1e-5. PASS/FAIL exit.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import torch
from spike import arms

torch.manual_seed(42)
G, P = 64, 200
w_geo = torch.rand(P, G, dtype=torch.float64)
opacity_raw = torch.randn(G, dtype=torch.float64)
color = torch.rand(G, 3, dtype=torch.float64)
z_raw = torch.rand(G, dtype=torch.float64) * 4.0 + 1.0   # camera depths ~[1,5]
w_b = torch.tensor(0.3, dtype=torch.float64)
c_b = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
target = torch.rand(P, 3, dtype=torch.float64)

tau = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
z = z_raw.clone().requires_grad_(True)

# ---- autograd reference through arms.blend_SM ----
out = arms.blend_SM(w_geo, opacity_raw, z, tau, color, w_b, c_b)
loss = ((out - target) ** 2).mean()
gz_auto, gt_auto = torch.autograd.grad(loss, [z, tau])

# ---- manual path: cut at rho, get dL/drho via autograd, then apply analytic formula ----
z2 = z_raw.clone().requires_grad_(True)
tau_d = tau.detach()
z_ref = z2.min().detach()
rho = torch.exp(-(z2 - z_ref) / tau_d).requires_grad_(True)
o = torch.sigmoid(opacity_raw)
W = o[None, :] * w_geo * rho[None, :]
num = W @ color + w_b * c_b[None, :]
den = W.sum(1, keepdim=True) + w_b
out2 = num / den
loss2 = ((out2 - target) ** 2).mean()
(grad_rho,) = torch.autograd.grad(loss2, [rho])   # dL/drho, shape [G]

rho_d = rho.detach()
inv_tau = 1.0 / tau_d
z_rel = z_raw - z_ref                              # (z - z_ref), detached

gz_manual = grad_rho * (-rho_d * inv_tau)                        # dL/dz per gaussian
gt_manual = (grad_rho * rho_d * z_rel * (inv_tau ** 2)).sum()    # dL/dtau scalar


def rel(a, b):
    return (abs(a - b) / (abs(b) + 1e-12)).item()


all_ok = True
for i in range(G):
    r = rel(gz_manual[i], gz_auto[i])
    if r >= 1e-5:
        print(f"z[{i}] grad MISMATCH: auto {gz_auto[i].item():+.6e}  manual {gz_manual[i].item():+.6e}  rel {r:.2e}")
        all_ok = False

r_tau = rel(gt_manual, gt_auto)
print(f"tau  grad: auto {gt_auto.item():+.6e}  manual {gt_manual.item():+.6e}  rel {r_tau:.2e}")
if r_tau >= 1e-5:
    all_ok = False

print("ORACLE", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
