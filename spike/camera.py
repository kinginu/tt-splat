"""Camera conventions for the matrix-native reference.

NeRF-synthetic (Blender) ships `transform_matrix` = camera-to-world in the OpenGL
convention: the camera looks down -z, +y is up, +x is right. We render in the OpenCV
convention: camera looks down +z, +y is down. This module pins that conversion (the
highest silent-error-risk step in the pipeline, per the spec) and a point projector.

Derivation: with c2w = [[R_c | cam_pos],[0|1]] (OpenGL) and flip = diag(1,-1,-1),
a point in OpenCV cam coords p_cv maps to OpenGL cam coords as p_gl = flip @ p_cv, so
    p_world = R_c @ flip @ p_cv + cam_pos
    => world->cam:  R_v = (R_c @ flip)^T ,  t_v = -R_v @ cam_pos
"""
import torch


def opengl_c2w_to_opencv_w2c(c2w):
    """OpenGL camera-to-world (4x4) -> OpenCV world-to-camera (R_v[3,3], t_v[3])."""
    c2w = torch.as_tensor(c2w)
    R_c = c2w[:3, :3]
    cam_pos = c2w[:3, 3]
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=c2w.dtype, device=c2w.device))
    R_v = (R_c @ flip).transpose(-1, -2)          # world->cam rotation (OpenCV)
    t_v = -R_v @ cam_pos
    return R_v, t_v


def project_point(p_world, R_v, t_v, fx, fy, cx, cy):
    """Project a single world point. Returns (pixel[2], depth z). z>0 means in front."""
    p_cam = R_v @ p_world + t_v
    z = p_cam[2]
    x = fx * p_cam[0] / z + cx
    y = fy * p_cam[1] / z + cy
    return torch.stack([x, y]), z


class Camera:
    """An OpenCV pinhole camera (world->cam R_v[3,3], t_v[3]) with intrinsics + image size."""

    __slots__ = ("R_v", "t_v", "fx", "fy", "cx", "cy", "H", "W")

    def __init__(self, R_v, t_v, fx, fy, cx, cy, H, W):
        self.R_v, self.t_v = R_v, t_v
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.H, self.W = H, W

    @property
    def center(self):
        return -self.R_v.transpose(-1, -2) @ self.t_v


def look_at_opencv(eye, target, up=(0.0, 1.0, 0.0), dtype=torch.float32, device=None):
    """Build an OpenCV world->cam (R_v, t_v) for a camera at `eye` looking at `target`."""
    eye = torch.as_tensor(eye, dtype=dtype, device=device)
    target = torch.as_tensor(target, dtype=dtype, device=device)
    up = torch.as_tensor(up, dtype=dtype, device=device)
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R_v = torch.stack([x, y, z], dim=0)        # rows = cam axes -> world->cam
    t_v = -R_v @ eye
    return R_v, t_v


def pixel_grid(H, W, dtype=torch.float32, device="cpu"):
    """Row-major pixel centers. Returns px[P] (x=col+0.5), py[P] (y=row+0.5), P=H*W."""
    ii, jj = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                            torch.arange(W, dtype=dtype, device=device), indexing="ij")
    return (jj + 0.5).reshape(-1), (ii + 0.5).reshape(-1)
