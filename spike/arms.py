"""The blend arms for the matrix-native reference.

All arms share the geometry + poly-splat (`w_geo`); only the weight build / blend differs:
  A   depth-free WSR (the locked weight)
  B   + per-gaussian depth weight (sort-free, global beta/tau)
  C0  capacity control: same opacity_sh tensor as C but DC-only (no view dependence)
  C   + view-dependent opacity (SH deg-2)
  SZ  + soft-Z visibility prepass (per-pixel local occlusion, GEMM-foldable; OIT rung 4)
  RV  + revealage compositing (background show-through; OIT rung b / WBOIT)
  D   sorted alpha-exp standard 3DGS (reference upper bound; exp + sort allowed here only)

The SZ/RV arms are the GEMM-embedded occlusion levers (the OIT-ladder side-project): they buy back
what dropping the depth sort lost, while keeping the Q=Φθ GEMM and the WSR GEMMs intact.

`w_b` is passed already-positive (caller applies softplus). `c_b` is the background color.
"""
import torch

from . import sh


def _wsr(W, color, w_b, c_b):
    """Weighted Sum Rendering with learnable background. W[P,G], color[G,3] -> [P,3]."""
    num = W @ color + w_b * c_b[None, :]
    den = W.sum(dim=1, keepdim=True) + w_b
    return num / den


def blend_A(w_geo, opacity_raw, color, w_b, c_b):
    o = torch.sigmoid(opacity_raw)
    return _wsr(o[None, :] * w_geo, color, w_b, c_b)


def blend_B(w_geo, opacity_raw, depth, depth_beta, depth_tau, color, w_b, c_b):
    o = torch.sigmoid(opacity_raw)
    rho = torch.sigmoid(depth_beta * (depth_tau - depth))      # sort-free, global depth bias
    return _wsr(o[None, :] * w_geo * rho[None, :], color, w_b, c_b)


def blend_C0(w_geo, opacity_sh, color, w_b, c_b):
    o = torch.sigmoid(sh.opacity_dc(opacity_sh))               # DC-only: no view dependence
    return _wsr(o[None, :] * w_geo, color, w_b, c_b)


def blend_C(w_geo, opacity_sh, dirs, color, w_b, c_b):
    o = torch.sigmoid(sh.sh_eval_deg2(opacity_sh, dirs))       # view-dependent opacity
    return _wsr(o[None, :] * w_geo, color, w_b, c_b)


def blend_SZ(w_geo, opacity_raw, depth, softz_beta, color, w_b, c_b, eps=1e-6):
    """soft-Z visibility prepass (OIT ladder rung 4): per-PIXEL local occlusion, sort-free, GEMM-bound.

    Two passes, both matrix-dominant:
      pass1  zstar(p) = Σ_i (o_i w_geo) z_i / Σ_i (o_i w_geo)   -- visible front depth, a GEMM ratio
      pass2  h[p,i]   = σ(-β (z_i - zstar(p)))                  -- attenuate gaussians behind the front
             C = WSR(o · w_geo · h)
    Unlike arm B's GLOBAL σ(β(τ-z)) front-bias, zstar is per-pixel, so this models occlusion that
    varies across the image (the residual a global front-bias could not buy). Depth is a visibility gate (detached):
    it steers blending, not geometry. β learnable (host-Adam scalar). Still folds into the WSR
    columns -> Q=Φθ GEMM untouched; cost is +1 [P,G] reduction (zstar) + 1 SFPU σ + the 2nd WSR."""
    o = torch.sigmoid(opacity_raw)
    z = depth.detach()                                         # gate, not a geometry path
    W0 = o[None, :] * w_geo                                    # [P,G] base WSR weights
    den0 = W0.sum(dim=1, keepdim=True) + eps                   # [P,1]
    zstar = (W0 @ z[:, None]) / den0                           # [P,1] Σwz/Σw, per-pixel front depth
    h = torch.sigmoid(-softz_beta * (z[None, :] - zstar))      # [P,G] front wins
    return _wsr(W0 * h, color, w_b, c_b)


def blend_RV(w_geo, opacity_raw, color, w_b, c_b, eps=1e-6):
    """Revealage compositing (OIT ladder rung b / Weighted-Blended OIT): restore background show-through.

    Plain WSR normalizes by Σw, so it ALWAYS fully replaces the background (no thin-edge / silhouette
    alpha). Carry a separate per-pixel revealage R(p)=Π(1-a_i)=exp(Σ log(1-a_i)) (order-independent;
    the product becomes an additive reduction = a GEMM, exp is per-pixel ONCE, not P×G), then composite
    C = C_avg·(1-R) + bg·R. This is the background/alpha axis (OIT loss #2), orthogonal to occlusion."""
    o = torch.sigmoid(opacity_raw)
    W = o[None, :] * w_geo                                     # [P,G] pseudo per-(p,g) alpha
    den = W.sum(dim=1, keepdim=True) + eps                     # [P,1]
    C_avg = (W @ color) / den                                  # object color where covered
    a = W.clamp(0.0, 1.0 - 1e-4)
    R = torch.exp(torch.log1p(-a).sum(dim=1, keepdim=True))    # [P,1] revealage (transmittance)
    return C_avg * (1.0 - R) + c_b[None, :] * R


def render_D(Q, opacity_raw, color, depth, c_b, keep=None):
    """Standard 3DGS reference: alpha = o*exp(-Q/2), sort by center depth, front-to-back composite.
    Sorting index is detached (non-diff); values flow grads. Exclusive transmittance (T_0=1)."""
    o = torch.sigmoid(opacity_raw)
    alpha = o[None, :] * torch.exp(-0.5 * Q)                   # [P,G]
    if keep is not None:
        alpha = alpha * keep[None, :].to(alpha.dtype)
    alpha = alpha.clamp(max=0.9999)
    order = torch.argsort(depth).detach()                     # per-view, pixel-independent
    alpha = alpha[:, order]
    c_ord = color[order]
    P = Q.shape[0]
    # transmittance via log-space cumsum (cheap, stable backward) instead of cumprod:
    # T_i = prod_{j<i}(1-alpha_j) = exp(cumsum(log(1-alpha))_{<i})
    cs = torch.cumsum(torch.log((1.0 - alpha).clamp(min=1e-6)), dim=1)
    zero = torch.zeros(P, 1, dtype=alpha.dtype, device=alpha.device)
    T = torch.exp(torch.cat([zero, cs[:, :-1]], dim=1))        # exclusive (T_0 = 1)
    weights = T * alpha
    T_final = torch.exp(cs[:, -1])[:, None]                    # transmittance past all gaussians
    return weights @ c_ord + T_final * c_b[None, :]
