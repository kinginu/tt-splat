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


def _moment_reconstruct(b, zw, m=4, eps=1e-6, recon="mboit", tau_softcmp=0.05):
    """Per-(pixel,gaussian) CDF fraction Â(z_g) in [0,1] using m+1 power moments.
    b: [P, m+1], zw: [G] in [0,1]. Returns A_hat [P, G].

    Strategy: minimum-norm / least-squares weight recovery on the Vandermonde system
    zp.T @ w[p] = b_hat[p], then sort-free CDF reconstruction via one of two variants:

    recon="softcmp" (Variant A): pairwise sigmoid S[g,h]=σ((zw_g−zw_h)/τ) ("h in front of g"),
        A_frac = w @ Sᵀ.  O(G²) GEMMs, no sort.  At τ→0 bit-identical to argsort+cumsum.
    recon="mboit"   (Variant B, default): Münstermann et al. 2018 power-moment OIT closed-form.
        Build (m/2+1)×(m/2+1) Hankel system from normalised moments, solve for polynomial
        coefficients, find roots via companion-matrix eigvals, count fraction < zw[g].
        O(P·m²) + O(P·(m/2)³ eigvals), no sort, matrix-engine friendly.

    For G < m+1 (e.g. 2-gaussian toy): overdetermined → least-squares via Gram [G,G].
      Recovers w ≈ a[g]/b0 exactly when the system is consistent.
    For G >= m+1 (e.g. 16-gaussian oracle): underdetermined → minimum-norm via Gram [m+1,m+1].
      w[g] = polynomial(zw[g]); when a[g]/b0 is approximately uniform the step CDF ≈ empirical CDF.

    grads flow through recovered weights w (which depend on zw via zp and b_hat), keeping the
    depth channel differentiable (C3).  Neither variant calls argsort/sort/cumsum.
    Reference: Münstermann et al. 2018, "Moment-Based Order-Independent Transparency".
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

    if recon == "softcmp":
        # Variant A: soft-compare GEMM.
        # S[g,h] = σ((zw_g − zw_h)/τ)  ≈  indicator[h is in front of g].
        # A_frac[p,g] = Σ_h w[p,h] * S[g,h]  =  (w @ Sᵀ)[p,g].
        # No sort, no cumsum.  At τ→0 recovers the exact sorted CDF.
        S = torch.sigmoid((zw[:, None] - zw[None, :]) / tau_softcmp)   # [G,G]
        S = S * (1.0 - torch.eye(G, dtype=zw.dtype, device=zw.device))  # zero self-compare
        A_hat = (w @ S.T).clamp(0.0, 1.0)   # [P,G]

    elif recon == "mboit":
        # Variant B: Münstermann 2018 power-moment MBOIT closed-form.
        # Normalized moments c[p,n] = b[p,n]/b0[p];  c[:,0] = 1 by construction.
        c = b_hat  # [P, m+1]

        half = m // 2  # for m=4: half=2

        # Build (half+1)×(half+1) Hankel matrix per pixel: H[i,j] = c[:,i+j]
        H = torch.stack([
            torch.stack([c[:, i + j] for j in range(half + 1)], dim=-1)
            for i in range(half + 1)
        ], dim=-2)  # [P, half+1, half+1]

        # Regularise for numerical stability
        reg_H = eps * torch.eye(half + 1, dtype=b.dtype, device=b.device).unsqueeze(0)
        H_reg = H + reg_H  # [P, half+1, half+1]

        # Solve lower-left half×half sub-block for polynomial coefficients β.
        # The monic polynomial z^half + β[half-1]*z^(half-1) + ... + β[0]
        # has roots in [0,1] that are the canonical depth knots (CDF steps).
        H_sub = H_reg[:, :half, :half]   # [P, half, half]
        rhs2 = -H_reg[:, :half, half]    # [P, half]  (lower-right column, negated)
        beta_lo = torch.linalg.solve(H_sub, rhs2.unsqueeze(-1)).squeeze(-1)  # [P, half]

        # Build companion matrix for  z^half + beta_lo[half-1]*z^(half-1) + ... + beta_lo[0].
        # Companion layout: sub-diagonal of 1s, last column = -beta_lo.
        comp = torch.zeros(P, half, half, dtype=b.dtype, device=b.device)
        if half > 1:
            comp[:, 1:, :-1] = torch.eye(half - 1, dtype=b.dtype, device=b.device).unsqueeze(0)
        comp[:, :, -1] = -beta_lo  # [P, half]

        # Eigenvalues = polynomial roots.  Take real parts, clamp to [0,1].
        eigs = torch.linalg.eigvals(comp)        # [P, half] complex
        roots = eigs.real.clamp(0.0, 1.0)        # [P, half]

        # Recover Gaussian-quadrature weights for the canonical atomic measure via overdetermined
        # Vandermonde lstsq: use all m moments c_0..c_{m-1} for better weight conditioning.
        # V[p, n, k] = roots[p,k]^n  (n=0..m-1, k=0..half-1) → [P, m, half]
        # Solve V @ w_q ≈ c[:,0:m] in min-norm least-squares sense.
        # Weights w_q satisfy Σ_k w_q[k]*root_k^n ≈ c_n; they sum to c_0=1.
        n_rows = min(m, m + 1)  # use all available moment rows
        V_r = torch.stack([roots ** n for n in range(n_rows)], dim=1)  # [P, n_rows, half]
        b_vand = c[:, :n_rows]  # [P, n_rows]: c_0..c_{n_rows-1}
        # lstsq: min_w ||V@w - b||^2  (overdetermined when n_rows > half)
        w_q = torch.linalg.lstsq(V_r, b_vand.unsqueeze(-1)).solution.squeeze(-1)  # [P, half]
        w_q = w_q.clamp(min=0.0)
        w_q = w_q / (w_q.sum(dim=-1, keepdim=True).clamp(min=eps))   # normalize to sum=1

        # Exclusive CDF: A_frac[p,g] = Σ_{k: root_k < zw_g} w_q[p,k]
        # roots: [P, half, 1]  vs  zw: [1, 1, G]
        # Use a small offset (EPS_CDF) to guard against the self-root edge case: when G≤half,
        # the roots are at the gaussian depths up to ~1e-4 float error, so root_k ≈ zw_g would
        # be incorrectly counted as "in front of" g.  EPS_CDF > max expected root error ensures
        # the self-root is excluded; it's small enough not to drop genuinely frontal roots
        # (G>half case: roots are Gauss-quadrature nodes, far from gaussian depths).
        EPS_CDF = 1e-3
        diff = roots.unsqueeze(-1) - zw[None, None, :]  # [P, half, G]: root_k - zw_g
        in_front = diff < -EPS_CDF                       # root strictly before z_g, not at it
        A_hat = (in_front.float() * w_q.unsqueeze(-1)).sum(dim=1)  # [P, G]
        A_hat = A_hat.clamp(0.0, 1.0)

    else:
        raise ValueError(f"Unknown recon={recon!r}; expected 'softcmp' or 'mboit'")

    return A_hat


def blend_MO(w_geo, opacity_raw, depth, color, w_b, c_b, m=4, eps=1e-6,
             recon="mboit", tau_softcmp=0.05):
    """Moment-based OIT (candidate #3): per-pixel transmittance from m power moments of the
    (depth, absorbance) measure; reconstruct T_i = exp(-b0 * Â(z_i)), composite OIT-over.
    Sort-free (moments = GEMM, reconstruction = sort-free variant); z ATTACHED (C3).
    OIT-style composite (not WSR): background gets true residual transmittance exp(-b0).

    recon="mboit"   (default): Münstermann 2018 Hankel+companion-matrix CDF.  O(P·m²), no sort.
    recon="softcmp": pairwise sigmoid A_frac=w@Sᵀ, S[g,h]=σ((zw_g−zw_h)/τ).  O(P·G²), no sort.
    tau_softcmp: temperature for softcmp variant (default 0.05; τ→0 = exact sorted CDF).

    Reference: Münstermann et al. 2018, Moment-Based Order-Independent Transparency."""
    o = torch.sigmoid(opacity_raw)
    alpha = (o[None, :] * w_geo).clamp(eps, 1.0 - 1e-4)       # [P, G]
    a = -torch.log1p(-alpha)                                    # [P, G] per-(p,g) absorbance
    zw = _depth_warp(depth)                                     # [G] in [0,1]; attached
    zp = torch.stack([zw ** n for n in range(m + 1)], dim=-1)  # [G, m+1]
    b = a @ zp                                                  # [P, m+1] moments (GEMM)
    A_frac = _moment_reconstruct(b, zw, m=m, eps=eps,
                                 recon=recon, tau_softcmp=tau_softcmp)  # [P, G]
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
                n_inf = int(all_inf.sum())
                C_s[all_inf] = c_b.to(C_s.dtype)[None, :].expand(n_inf, -1)
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
