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


def _depth_warp(depth, near=0.5, far=8.0):
    """Map camera z to [0,1] for numerical stability of power basis."""
    return ((depth - near) / (far - near)).clamp(0.0, 1.0)


def _moment_reconstruct(b, zw, m=4, eps=1e-6):
    """Per-(pixel,gaussian) CDF fraction Â(z_g) in [0,1] using m+1 power moments.
    b: [P, m+1], zw: [G] in [0,1]. Returns A_hat [P, G].

    Strategy: minimum-norm / least-squares weight recovery on the Vandermonde system
    zp.T @ w[p] = b_hat[p], then exact exclusive step CDF in depth order.

    For G < m+1 (e.g. 2-gaussian toy): overdetermined → least-squares via Gram [G,G].
      Recovers w ≈ a[g]/b0 exactly when the system is consistent.
    For G >= m+1 (e.g. 16-gaussian oracle): underdetermined → minimum-norm via Gram [m+1,m+1].
      w[g] = polynomial(zw[g]); when a[g]/b0 is approximately uniform the step CDF ≈ empirical CDF.

    Sort is detached so grads flow through the recovered weights w (which depend on zw via zp
    and b_hat), keeping the depth channel differentiable (C3).
    """
    P = b.shape[0]
    G = zw.shape[0]
    b0 = b[:, 0:1].clamp(min=eps)    # [P, 1]
    b_hat = b / b0                     # [P, m+1]

    # Vandermonde power basis: zp[g, n] = zw[g]^n
    zp = torch.stack([zw ** n for n in range(m + 1)], dim=-1)  # [G, m+1]

    reg = eps * float(max(G, m + 1))

    if G >= m + 1:
        # Underdetermined: minimum-norm solution w = zp @ solve(zp.T @ zp, b_hat.T)
        gram = zp.T @ zp + reg * torch.eye(m + 1, dtype=b.dtype, device=b.device)
        c = torch.linalg.solve(gram, b_hat.T)   # [m+1, P]
        w = (zp @ c).T                           # [P, G]
    else:
        # Overdetermined: least-squares w = solve(zp @ zp.T, zp @ b_hat.T)
        gram2 = zp @ zp.T + reg * torch.eye(G, dtype=b.dtype, device=b.device)
        w = torch.linalg.solve(gram2, (zp @ b_hat.T)).T  # [P, G]

    # Exclusive step CDF: sort by depth (detached), exclusive cumsum of w
    order = zw.argsort().detach()               # [G]
    w_sorted = w[:, order]                      # [P, G] sorted by depth
    zeros = torch.zeros(P, 1, dtype=b.dtype, device=b.device)
    cum_excl = torch.cat([zeros, w_sorted[:, :-1].cumsum(dim=1)], dim=1)  # [P, G]
    A_hat = torch.zeros(P, G, dtype=b.dtype, device=b.device)
    A_hat[:, order] = cum_excl

    return A_hat.clamp(0.0, 1.0)


def blend_MO(w_geo, opacity_raw, depth, color, w_b, c_b, m=4, eps=1e-6):
    """Moment-based OIT (candidate #3): per-pixel transmittance from m power moments of the
    (depth, absorbance) measure; reconstruct T_i = exp(-b0 * Â(z_i)), composite OIT-over.
    Sort-free (moments = GEMM, reconstruction = polynomial eval per pixel); z ATTACHED (C3).
    OIT-style composite (not WSR): background gets true residual transmittance exp(-b0).
    Reference: Münstermann et al. 2018, Moment-Based Order-Independent Transparency."""
    o = torch.sigmoid(opacity_raw)
    alpha = (o[None, :] * w_geo).clamp(eps, 1.0 - 1e-4)       # [P, G]
    a = -torch.log1p(-alpha)                                    # [P, G] per-(p,g) absorbance
    zw = _depth_warp(depth)                                     # [G] in [0,1]; attached
    zp = torch.stack([zw ** n for n in range(m + 1)], dim=-1)  # [G, m+1]
    b = a @ zp                                                  # [P, m+1] moments (GEMM)
    A_frac = _moment_reconstruct(b, zw, m=m, eps=eps)          # [P, G] CDF fraction in [0,1]
    T = torch.exp(-b[:, 0:1] * A_frac)                         # [P, G] transmittance
    W = alpha * T                                               # [P, G]
    num = W @ color                                             # [P, 3]
    T_bg = torch.exp(-b[:, 0:1])                               # [P, 1] residual transmittance
    return num + T_bg * c_b[None, :]


def blend_C0(w_geo, opacity_sh, color, w_b, c_b):
    o = torch.sigmoid(sh.opacity_dc(opacity_sh))               # DC-only: no view dependence
    return _wsr(o[None, :] * w_geo, color, w_b, c_b)


def blend_C(w_geo, opacity_sh, dirs, color, w_b, c_b):
    o = torch.sigmoid(sh.sh_eval_deg2(opacity_sh, dirs))       # view-dependent opacity
    return _wsr(o[None, :] * w_geo, color, w_b, c_b)


def blend_SM(w_geo, opacity_raw, depth, softmin_tau, color, w_b, c_b):
    """Softmin / Boltzmann depth weight (candidate #2): per-gaussian front-weight
    rho = exp(-(z - z_ref)/tau), z_ref = detached min(z). tau->0 = front-takes-all (hard
    occlusion), tau->inf = uniform = arm A. Sort-free (per-gaussian exp = lane-wise), z stays
    ATTACHED so occlusion produces a z-force (C3). One learnable scalar tau>0."""
    o = torch.sigmoid(opacity_raw)
    tau = softmin_tau.clamp(min=1e-3)
    z_ref = depth.min().detach()                       # front anchor; subgradient detached
    rho = torch.exp(-(depth - z_ref) / tau)            # [G]
    return _wsr(o[None, :] * w_geo * rho[None, :], color, w_b, c_b)


def blend_BP(w_geo, opacity_raw, depth, bp_tau, color, w_b, c_b, eps=1e-4):
    """Pairwise soft-occlusion GEMM ('arm B′', candidate #5): per-gaussian transmittance
    rho_i = exp(-(S beta)_i), S_ij = sigmoid((z_i - z_j)/tau) (j in front of i, diag zeroed),
    beta_j = -log(1-o_j). tau->0 = exact alpha-transmittance Prod(1-o_j). Sort-free (S = [G,G]
    GEMM, exp lane-wise); z ATTACHED (C3). One learnable scalar tau>0."""
    o = torch.sigmoid(opacity_raw)
    tau = bp_tau.clamp(min=1e-3)
    beta = -torch.log1p(-(o.clamp(max=1.0 - eps)))                 # [G]
    S = torch.sigmoid((depth[:, None] - depth[None, :]) / tau)     # [G,G]
    S = S * (1.0 - torch.eye(S.shape[0], dtype=S.dtype, device=S.device))   # zero diagonal
    rho = torch.exp(-(S @ beta))                                   # [G]
    return _wsr(o[None, :] * w_geo * rho[None, :], color, w_b, c_b)


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


def blend_E(w_geo, opacity_raw, depth, e_tau, color, w_b, c_b, S=8, gen=None):
    """Stochastic transparency / Gumbel-softmin (candidate #8, 'arm E'). Forward = stochastic-hard
    (Bernoulli keep prob alpha_pg, nearest-kept argmin, mean of S samples) = unbiased alpha
    compositing with crisp occlusion edges. Backward via softmin surrogate (STE): value=hard,
    grad=soft. z attached through soft path (C3). Sort-free: argmin is per-pixel, not global."""
    o = torch.sigmoid(opacity_raw)
    alpha = (o[None, :] * w_geo).clamp(1e-6, 1.0 - 1e-4)        # [P,G]

    # Hard (unbiased, no grad): stochastic argmin
    P, G = alpha.shape
    results = []
    with torch.no_grad():
        for _ in range(S):
            if gen is not None:
                noise = torch.rand(P, G, dtype=alpha.dtype, device=alpha.device, generator=gen)
            else:
                noise = torch.rand(P, G, dtype=alpha.dtype, device=alpha.device)
            keep = (noise < alpha).float()                         # [P,G] Bernoulli samples
            d_masked = depth.unsqueeze(0).expand(P, G).clone()
            d_masked = d_masked.masked_fill(keep == 0, float("inf"))
            winner = d_masked.argmin(dim=1)                       # [P]
            all_inf = d_masked.isinf().all(dim=1)                 # [P]
            C_s = color[winner]                                   # [P,3]
            if all_inf.any():
                C_s = C_s.clone()
                C_s[all_inf] = c_b.to(C_s.dtype)
            results.append(C_s)
    hard = torch.stack(results).mean(0)                           # [P,3]

    # Soft (differentiable, z attached): softmin surrogate
    tau = e_tau.clamp(min=1e-3)
    z_ref = depth.min().detach()
    rho = torch.exp(-(depth - z_ref) / tau)                       # [G]
    soft = _wsr(o[None, :] * w_geo * rho[None, :], color, w_b, c_b)   # [P,3]

    # STE: value = hard (unbiased), grad = soft (differentiable)
    return soft + (hard - soft).detach()                          # [P,3]


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
