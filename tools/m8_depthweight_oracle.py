"""Oracle: verify the analytic backward for the depth-weight lever.

depth weight (per-gaussian, sort-free, global beta/tau, multiplies o):
    rho_g = sigmoid(beta * (tau - z_g))
    w_g   = o_g * w_geo_pg * rho_g            (folded into keo = keep*o*rho on device)

The device trainer already has gkeo = dL/d(keo) where keo = keep*o*rho. The NEW grads:
    grad_rho_g = gkeo_g * keep_g * o_g
    pre_g      = beta*(tau - z_g);  d rho/d pre = rho*(1-rho)
    grad_pre_g = grad_rho_g * rho_g*(1-rho_g)
    grad_beta  = sum_g grad_pre_g * (tau - z_g)
    grad_tau   = sum_g grad_pre_g * beta

This isolates exactly the new code's math against autograd (treat rho as the cut point).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import torch
from spike import arms

torch.manual_seed(0)
G, P = 64, 200
w_geo = torch.rand(P, G)
opacity_raw = torch.randn(G)
color = torch.rand(G, 3)
z = torch.rand(G) * 4.0 + 1.0            # camera depths ~[1,5]
w_b = torch.tensor(0.3); c_b = torch.tensor([1.0, 1.0, 1.0])
target = torch.rand(P, 3)

beta = torch.tensor(1.3, requires_grad=True)
tau = torch.tensor(2.7, requires_grad=True)

# ---- autograd reference through arms.blend_B (the spike's validated arm B) ----
out = arms.blend_B(w_geo, opacity_raw, z, beta, tau, color, w_b, c_b)
loss = ((out - target) ** 2).mean()
gb_auto, gt_auto = torch.autograd.grad(loss, [beta, tau])

# ---- my manual path: cut at rho, get dL/drho via autograd, then apply MY formula ----
o = torch.sigmoid(opacity_raw)
pre = beta.detach() * (tau.detach() - z)
rho = torch.sigmoid(pre).requires_grad_(True)
W = o[None, :] * w_geo * rho[None, :]
num = W @ color + w_b * c_b[None, :]
den = W.sum(1, keepdim=True) + w_b
out2 = num / den
loss2 = ((out2 - target) ** 2).mean()
(grad_rho,) = torch.autograd.grad(loss2, [rho])     # = dL/drho (= what gkeo*keep*o reduces to: here keep=1)

# MY analytic beta/tau grads from grad_rho:
grad_pre = grad_rho * rho.detach() * (1 - rho.detach())
gb_manual = (grad_pre * (tau.detach() - z)).sum()
gt_manual = (grad_pre * beta.detach()).sum()

def rel(a, b): return (abs(a - b) / (abs(b) + 1e-12)).item()
print(f"beta grad: auto {gb_auto.item():+.6e}  manual {gb_manual.item():+.6e}  rel {rel(gb_manual, gb_auto):.2e}")
print(f"tau  grad: auto {gt_auto.item():+.6e}  manual {gt_manual.item():+.6e}  rel {rel(gt_manual, gt_auto):.2e}")
ok = rel(gb_manual, gb_auto) < 1e-5 and rel(gt_manual, gt_auto) < 1e-5
print("ORACLE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
