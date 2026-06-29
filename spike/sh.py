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


C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def eval_sh_color(deg, coeffs, dirs):
    """View-dependent RGB color from SH coefficients (inria/gsplat convention).

    coeffs [..., (deg+1)^2, 3] (coeffs[...,0,:] = DC), dirs [..., 3] (normalized internally).
    Returns clamp(0.5 + SH(dir), 0) [..., 3]. At deg=0 this equals forward.color_from_dc(coeffs[...,0,:]).
    """
    d = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]                       # [...,1] -> broadcast over 3 ch
    r = C0 * coeffs[..., 0, :]
    if deg >= 1:
        r = r - C1 * y * coeffs[..., 1, :] + C1 * z * coeffs[..., 2, :] - C1 * x * coeffs[..., 3, :]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy * coeffs[..., 4, :] + C2[1] * yz * coeffs[..., 5, :]
             + C2[2] * (2.0 * zz - xx - yy) * coeffs[..., 6, :]
             + C2[3] * xz * coeffs[..., 7, :] + C2[4] * (xx - yy) * coeffs[..., 8, :])
    if deg >= 3:
        r = (r + C3[0] * y * (3 * xx - yy) * coeffs[..., 9, :] + C3[1] * xy * z * coeffs[..., 10, :]
             + C3[2] * y * (4 * zz - xx - yy) * coeffs[..., 11, :]
             + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * coeffs[..., 12, :]
             + C3[4] * x * (4 * zz - xx - yy) * coeffs[..., 13, :]
             + C3[5] * z * (xx - yy) * coeffs[..., 14, :]
             + C3[6] * x * (xx - 3 * yy) * coeffs[..., 15, :])
    return (r + 0.5).clamp(min=0.0)
