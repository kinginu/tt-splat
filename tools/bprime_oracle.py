"""Oracle: verify analytic gradients for arm BP (pairwise soft-occlusion GEMM).

arm BP forward:
    o_j    = sigmoid(opacity_raw_j)
    beta_j = -log(1 - o_j)                           absorbance
    S_ij   = sigmoid((z_i - z_j) / tau)   with diag=0   [G,G] pairwise visibility matrix
    logT_i = -(S @ beta)_i                           log transmittance
    rho_i  = exp(logT_i)                             per-gaussian transmittance
    W_pj   = o_j * w_geo_pj * rho_j                  WSR weight (two paths for opacity_raw)

Gradients checked analytically vs autograd:

grad_tau, grad_z:  only go through S (no direct path), fully analytic:
    grad_logT[i]  = grad_rho[i] * rho[i]
    grad_S[i,j]   = -grad_logT[i] * beta[j]          (from GEMM)
    D[i,j]        = (z[i] - z[j]) / tau
    grad_d[i,j]   = grad_S[i,j] * S[i,j] * (1 - S[i,j])
    grad_z[i]    += sum_j(grad_d[i,j]) / tau          row (z_i is in numerator)
    grad_z[j]    -= sum_i(grad_d[i,j]) / tau          col (z_j is in denominator)
    grad_tau       = -(1/tau) * sum(grad_d * D)

grad_op:  opacity_raw flows through TWO paths:
    path A  o -> W directly  (W = o * w_geo * rho)
    path B  o -> beta -> rho -> W
    We verify each path's contribution separately, then sum.
    Path B analytic:  grad_beta[j] = -(S.T @ grad_logT)[j]
                      grad_op_B[j] = grad_beta[j] * o[j]   (chain: beta=-log1p(-o), o=sigmoid(op))
    Path A:  computed via autograd with rho held fixed (cuts exactly path B).

Assert all max_rel < 1e-5. PASS/FAIL exit.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import torch
from spike import arms

torch.manual_seed(42)
G, P = 16, 32
dtype = torch.float64

w_geo = torch.rand(P, G, dtype=dtype)
opacity_raw = torch.randn(G, dtype=dtype)
color = torch.rand(G, 3, dtype=dtype)
z_val = torch.rand(G, dtype=dtype) * 4.0 + 1.0    # camera depths ~[1,5]
w_b = torch.tensor(0.3, dtype=dtype)
c_b = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
target = torch.rand(P, 3, dtype=dtype)

tau_val = torch.tensor(1.5, dtype=dtype)


# ---- autograd reference through arms.blend_BP ----
opacity_raw_ag = opacity_raw.clone().requires_grad_(True)
z_ag = z_val.clone().requires_grad_(True)
tau_ag = tau_val.clone().requires_grad_(True)

out_ag = arms.blend_BP(w_geo, opacity_raw_ag, z_ag, tau_ag, color, w_b, c_b)
loss_ag = ((out_ag - target) ** 2).mean()
loss_ag.backward()
grad_op_auto = opacity_raw_ag.grad.clone()
grad_z_auto = z_ag.grad.clone()
grad_tau_auto = tau_ag.grad.clone()


# ---- shared constants (no grad) ----
eps = 1e-4
o = torch.sigmoid(opacity_raw.detach())
beta_det = -torch.log1p(-(o.clamp(max=1.0 - eps)))
tau_det = tau_val.detach()
z_det = z_val.detach()

eye = torch.eye(G, dtype=dtype)
S_mat = torch.sigmoid((z_det[:, None] - z_det[None, :]) / tau_det) * (1.0 - eye)
logT = -(S_mat @ beta_det)
rho_det = torch.exp(logT)


# ---- step 1: get dL/drho via autograd (rho leaf; o fixed, direct path excluded) ----
rho_leaf = rho_det.clone().requires_grad_(True)
W1 = o[None, :] * w_geo * rho_leaf[None, :]
num1 = W1 @ color + w_b * c_b[None, :]
den1 = W1.sum(1, keepdim=True) + w_b
out1 = num1 / den1
loss1 = ((out1 - target) ** 2).mean()
(grad_rho,) = torch.autograd.grad(loss1, [rho_leaf])


# ---- step 2: get dL/do_direct via autograd (o leaf; rho fixed, path B excluded) ----
o_leaf = o.clone().requires_grad_(True)
W2 = o_leaf[None, :] * w_geo * rho_det[None, :]
num2 = W2 @ color + w_b * c_b[None, :]
den2 = W2.sum(1, keepdim=True) + w_b
out2 = num2 / den2
loss2 = ((out2 - target) ** 2).mean()
(grad_o_direct,) = torch.autograd.grad(loss2, [o_leaf])


# ---- analytic formulas ----
with torch.no_grad():
    # grad_tau and grad_z (only through rho path, no direct o path)
    grad_logT = grad_rho * rho_det                       # [G]
    grad_S = -grad_logT[:, None] * beta_det[None, :]     # [G,G]
    grad_beta = -(S_mat.T @ grad_logT)                   # [G]

    D = (z_det[:, None] - z_det[None, :]) / tau_det      # [G,G]
    grad_d = grad_S * S_mat * (1.0 - S_mat)              # [G,G]

    grad_z_analytic = (grad_d.sum(dim=1) - grad_d.sum(dim=0)) / tau_det   # [G]
    grad_tau_analytic = (-(1.0 / tau_det) * (grad_d * D).sum())

    # grad_op: path B (through beta->rho) + path A (direct o in W)
    # path B: d(beta)/d(op) = d(-log(1-o))/d(o) * d(sigmoid)/d(op) = 1/(1-o)*o*(1-o) = o
    grad_op_path_B = grad_beta * o                        # [G]
    # path A: d(o)/d(op) = o * (1 - o)  (sigmoid derivative)
    grad_op_path_A = grad_o_direct * o * (1.0 - o)       # [G]
    grad_op_analytic = grad_op_path_B + grad_op_path_A   # [G]


def rel(a, b):
    return ((a - b).abs() / (b.abs() + 1e-12)).max().item()


rel_tau = rel(grad_tau_analytic, grad_tau_auto)
rel_z = rel(grad_z_analytic, grad_z_auto)
rel_op = rel(grad_op_analytic, grad_op_auto)

print(f"grad_tau: auto {grad_tau_auto.item():+.6e}  analytic {grad_tau_analytic.item():+.6e}  rel {rel_tau:.2e}")
print(f"grad_z:   max_rel {rel_z:.2e}")
print(f"grad_op:  max_rel {rel_op:.2e}")

ok = rel_tau < 1e-5 and rel_z < 1e-5 and rel_op < 1e-5
print("ORACLE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
