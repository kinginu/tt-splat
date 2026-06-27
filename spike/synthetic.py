"""Synthetic multi-view scene for the data-free self-consistency oracle.

Build a known gaussian set (`make_gt_model`), orbit cameras around it (`make_orbit_cameras`),
and render ground-truth images under a chosen arm. A fresh model overfit to those images must
recover them — this validates render + autograd + Adam + arms end-to-end with NO external data.
"""
import math

import torch

from .camera import Camera, look_at_opencv
from .model import GaussianModel
from .render import render


def make_gt_model(seed=1, G=30, extent=0.5, dtype=torch.float32, device="cpu"):
    return GaussianModel(G, extent=extent, dtype=dtype, seed=seed, gt=True, init_scale=0.08, device=device)


def make_orbit_cameras(n=4, radius=2.5, elevation=0.6, res=40, fov=0.7, dtype=torch.float32, device="cpu"):
    fx = fy = 0.5 * res / math.tan(0.5 * fov)
    cx = cy = res / 2.0
    cams = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        eye = (radius * math.cos(th), elevation, radius * math.sin(th))
        R_v, t_v = look_at_opencv(eye, (0.0, 0.0, 0.0), dtype=dtype, device=device)
        cams.append(Camera(R_v, t_v, fx, fy, cx, cy, res, res))
    return cams


def render_gt_images(gt_model, cameras, arm, **kw):
    with torch.no_grad():
        return [render(gt_model, cam, arm, **kw) for cam in cameras]


def make_scene(arm="A", seed=1, G_gt=30, n_views=4, res=40, device="cpu"):
    """Returns (gt_model, cameras, gt_images) for the given arm (all on `device`)."""
    gt = make_gt_model(seed=seed, G=G_gt, device=device)
    cams = make_orbit_cameras(n=n_views, res=res, device=device)
    imgs = render_gt_images(gt, cams, arm)
    return gt, cams, imgs
