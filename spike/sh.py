"""Real spherical-harmonics basis (gsplat/inria convention).

Degree 0 is used for color (DC only, all arms). Degree 2 (9 coeffs) is used for arm C's
view-dependent opacity (scalar channel). `opacity_dc` is the view-independent control used
by arm C0 (same param tensor as C, but only the DC term active).
"""
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]


def sh_eval_deg2(coeffs, dirs):
    """coeffs [G,9] (scalar channel), dirs [G,3] (normalized internally) -> [G]."""
    d = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    r = C0 * coeffs[..., 0]
    r = r - C1 * y * coeffs[..., 1] + C1 * z * coeffs[..., 2] - C1 * x * coeffs[..., 3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    r = (r + C2[0] * xy * coeffs[..., 4] + C2[1] * yz * coeffs[..., 5]
         + C2[2] * (2.0 * zz - xx - yy) * coeffs[..., 6]
         + C2[3] * xz * coeffs[..., 7] + C2[4] * (xx - yy) * coeffs[..., 8])
    return r


def opacity_dc(coeffs):
    """View-independent (DC-only) opacity logit from the deg-2 coeff tensor (arm C0)."""
    return C0 * coeffs[..., 0]
