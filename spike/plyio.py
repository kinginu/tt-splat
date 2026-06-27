"""Minimal 3D Gaussian Splatting .ply writer (standard inria/gsplat field layout).

Maps our matrix-native GaussianModel params to the standard binary .ply so the trained model is a
real, inspectable artifact. Fields: x,y,z, nx,ny,nz, f_dc_0..2 (SH DC color), opacity (logit),
scale_0..2 (log), rot_0..3 (quat). DC-only (no f_rest_*); most viewers accept that.

CAVEAT (matrix-native): these gaussians are FIT TO OUR matrix-native RENDERER (poly-splat
`(1-Q/k)_+^2` + Weighted Sum Rendering, no exp, no depth sort). A STANDARD 3DGS viewer renders
with exp-splat + sorted alpha and will NOT reproduce our image. The faithful visualization is our
own render() (see tools/export_artifacts.py). The .ply is the trained model, not a drop-in for
standard viewers.
"""
import numpy as np
import torch

_FIELDS = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
           + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])


@torch.no_grad()
def save_ply(path, model):
    """Write `model` to `path` as a standard binary_little_endian 3DGS .ply. Returns N gaussians."""
    xyz = model.means3d.detach().float().cpu().numpy()
    n = xyz.shape[0]
    normals = np.zeros((n, 3), dtype=np.float32)
    f_dc = model.color_dc.detach().float().cpu().numpy()                       # SH DC color
    opacity = model.opacity_raw.detach().float().cpu().numpy().reshape(n, 1)   # logit (pre-sigmoid)
    scale = model.log_scales.detach().float().cpu().numpy()                    # log scale
    q = model.quats.detach().float()
    q = (q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)).cpu().numpy()      # normalized quat

    data = np.concatenate([xyz, normals, f_dc, opacity, scale, q], axis=1).astype("<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {f}\n" for f in _FIELDS)
              + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())
    return n
