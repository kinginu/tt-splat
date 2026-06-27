"""The novel matrix-native per-pixel forward primitives for the matrix-native reference.

- `quad_form`   : raw quadratic form Q = d^T M d (NO /k inside — spec D3).
- `theta_from_conic` : the locked monomial map; used by test_transform to assert Q == Phi.theta.
- `poly_splat_wgeo`  : the poly-splat shape factor max(0, 1 - Q/k)^2, with the keep mask.
- `color_from_dc`    : SH degree-0 color.
"""
import torch

from . import sh


def quad_form(px, py, mu2d, conic_abc):
    """px,py [P]; mu2d [G,2]; conic_abc [G,3] -> Q [P,G] (RAW Mahalanobis, no /k)."""
    dx = px[:, None] - mu2d[None, :, 0]
    dy = py[:, None] - mu2d[None, :, 1]
    a = conic_abc[:, 0][None, :]
    b = conic_abc[:, 1][None, :]
    c = conic_abc[:, 2][None, :]
    return a * dx * dx + 2 * b * dx * dy + c * dy * dy


def theta_from_conic(conic_abc, mu2d, k):
    """Locked monomial map. Returns theta_Q[G,6], theta_u[G,6] over
    Phi = [px^2, py^2, px*py, px, py, 1]. theta_u folds u = 1 - Q/k."""
    a, b, c = conic_abc[:, 0], conic_abc[:, 1], conic_abc[:, 2]
    mux, muy = mu2d[:, 0], mu2d[:, 1]
    theta_Q = torch.stack([
        a,
        c,
        2 * b,
        -2 * (a * mux + b * muy),
        -2 * (b * mux + c * muy),
        a * mux * mux + 2 * b * mux * muy + c * muy * muy,
    ], dim=-1)
    bump = torch.zeros_like(theta_Q)
    bump[..., 5] = 1.0
    theta_u = (-1.0 / k) * theta_Q + bump
    return theta_Q, theta_u


def phi(px, py):
    """Monomial basis Phi(pixel) -> [P,6]."""
    return torch.stack([px * px, py * py, px * py, px, py, torch.ones_like(px)], dim=-1)


def quad_form_tilelocal(lx, ly, conic_abc, mu2d, tile_origin):
    """Q over a tile as the EXACT GEMM  Q = Phi(lx,ly) @ theta_Q^T  with the tile origin
    folded into the gaussian mean (mu_local = mu2d - tile_origin) so pixel coords stay tile-local
    (lx,ly in 0..15 -> px^2 <= 225). Identical to quad_form() in exact arithmetic because the
    displacement dx = lx - mu_local = px - mu is invariant under the shift; but numerically stable
    in bf16, where the global form catastrophically cancels (a*px^2 ~ 1e4 terms summing to O(1) Q).

    lx,ly [P] tile-local pixel coords; conic_abc [G,3]; mu2d [G,2] global; tile_origin [2] -> Q [P,G].
    """
    mu_local = mu2d - tile_origin                              # fold the offset into theta (k unused)
    theta_Q, _ = theta_from_conic(conic_abc, mu_local, k=1.0)  # [G,6]
    return phi(lx, ly) @ theta_Q.transpose(-1, -2)            # [P,6] @ [6,G] -> [P,G]


def poly_splat_wgeo(Q, k, keep=None):
    """Poly-splat shape factor w_geo = max(0, 1 - Q/k)^2  [P,G]; keep [G] gates culled gaussians."""
    w_geo = (1.0 - Q / k).clamp(min=0.0) ** 2
    if keep is not None:
        w_geo = w_geo * keep[None, :].to(w_geo.dtype)
    return w_geo


def color_from_dc(color_dc):
    """SH degree-0 color: clamp(0.5 + C0*dc, min=0)  [G,3] (min-only clamp, matches gsplat)."""
    return (0.5 + sh.C0 * color_dc).clamp(min=0.0)
