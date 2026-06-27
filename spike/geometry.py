"""Shared 3DGS geometry for the matrix-native reference (standard math — formulas from gsplat).

This is the "G-setup" stage (per-gaussian, cheap): quaternion->rotation, 3D covariance,
and the EWA projection world->2D conic. Dtype-agnostic (used in fp64 for the lock tests,
fp32 in the train loop). All the novel matrix-native math lives in forward.py / arms.py, not here.
"""
import torch


def quat_to_rotmat(quats):
    """quats [...,4] in [w,x,y,z] order (normalized internally) -> R [...,3,3]."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def cov3d(scales, R):
    """3D covariance Sigma3 = R diag(scales^2) R^T. scales [...,3] (already exp'd), R [...,3,3]."""
    s2 = scales * scales
    return (R * s2[..., None, :]) @ R.transpose(-1, -2)


def project_ewa(means3d, cov3d_w, R_v, t_v, fx, fy, cx, cy, blur_eps=0.3, near=0.2):
    """EWA projection of 3D gaussians to 2D conics.

    Returns:
        mu2d      [G,2]  2D means in pixel coords
        conic_abc [G,3]  conic = Sigma2d^-1 as (a, b, c) = (M00, M01, M11)
        depth     [G]    camera-space z
        keep      [G]    bool, depth > near
    """
    mu_cam = means3d @ R_v.transpose(-1, -2) + t_v               # [G,3]
    depth = mu_cam[:, 2]
    keep = depth > near
    z = depth.clamp(min=near)                                    # z-guard: masked g can't emit inf/nan
    mu2d = torch.stack([fx * mu_cam[:, 0] / z + cx,
                        fy * mu_cam[:, 1] / z + cy], dim=-1)      # [G,2]

    Sigma_cam = torch.einsum('ij,gjk,lk->gil', R_v, cov3d_w, R_v)  # R_v Sigma3 R_v^T  [G,3,3]
    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([fx / z, zero, -fx * mu_cam[:, 0] / (z * z)], dim=-1),
        torch.stack([zero, fy / z, -fy * mu_cam[:, 1] / (z * z)], dim=-1),
    ], dim=-2)                                                   # [G,2,3]
    Sigma2d = torch.einsum('gij,gjk,glk->gil', J, Sigma_cam, J)  # J Sigma_cam J^T  [G,2,2]
    eye2 = torch.eye(2, dtype=Sigma2d.dtype, device=Sigma2d.device)
    Sigma2d = Sigma2d + blur_eps * eye2                          # low-pass on covariance

    s00, s01, s11 = Sigma2d[:, 0, 0], Sigma2d[:, 0, 1], Sigma2d[:, 1, 1]
    det = (s00 * s11 - s01 * s01).clamp(min=1e-12)               # closed-form 2x2 inverse
    conic_abc = torch.stack([s11 / det, -s01 / det, s00 / det], dim=-1)  # (a, b, c)
    return mu2d, conic_abc, depth, keep
