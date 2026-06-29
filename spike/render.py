"""End-to-end render: gaussian model + camera + arm -> image [H,W,3]. 

Ties the shared geometry (geometry.project_ewa) to the novel per-pixel forward
(forward.quad_form + poly_splat) to the chosen blend (arms.*). Differentiable throughout.
"""
import torch
import torch.nn.functional as F

from . import arms, camera as cam_mod, forward, geometry, sh

ARMS = ("A", "B", "C0", "C", "SZ", "RV", "SM", "BP", "E", "D")


def render(model, cam, arm, k=4.0, blur_eps=0.3, near=0.2, sh_degree=0):
    R = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), R)
    mu2d, conic_abc, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, blur_eps, near)

    px, py = cam_mod.pixel_grid(cam.H, cam.W, dtype=model.means3d.dtype, device=model.means3d.device)
    Q = forward.quad_form(px, py, mu2d, conic_abc)               # [P,G]
    w_geo = forward.poly_splat_wgeo(Q, k, keep)
    if sh_degree > 0 and hasattr(model, "color_rest"):           # view-dependent SH colour
        coeffs = torch.cat([model.color_dc[:, None, :], model.color_rest], dim=1)   # [G,16,3]
        color = sh.eval_sh_color(sh_degree, coeffs, model.means3d - cam.center[None, :])
    else:
        color = forward.color_from_dc(model.color_dc)            # [G,3] DC-only
    w_b = F.softplus(model.w_b_raw)
    c_b = model.c_b

    if arm == "A":
        img = arms.blend_A(w_geo, model.opacity_raw, color, w_b, c_b)
    elif arm == "B":
        img = arms.blend_B(w_geo, model.opacity_raw, depth,
                           model.depth_beta, model.depth_tau, color, w_b, c_b)
    elif arm == "C0":
        img = arms.blend_C0(w_geo, model.opacity_sh, color, w_b, c_b)
    elif arm == "C":
        dirs = model.means3d - cam.center[None, :]
        img = arms.blend_C(w_geo, model.opacity_sh, dirs, color, w_b, c_b)
    elif arm == "SM":
        img = arms.blend_SM(w_geo, model.opacity_raw, depth, model.softmin_tau, color, w_b, c_b)
    elif arm == "BP":
        img = arms.blend_BP(w_geo, model.opacity_raw, depth, model.bp_tau, color, w_b, c_b)
    elif arm == "E":
        img = arms.blend_E(w_geo, model.opacity_raw, depth, model.e_tau, color, w_b, c_b)
    elif arm == "SZ":
        img = arms.blend_SZ(w_geo, model.opacity_raw, depth, model.softz_beta, color, w_b, c_b)
    elif arm == "RV":
        img = arms.blend_RV(w_geo, model.opacity_raw, color, w_b, c_b)
    elif arm == "D":
        img = arms.render_D(Q, model.opacity_raw, color, depth, c_b, keep)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return img.reshape(cam.H, cam.W, 3)
