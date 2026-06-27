"""NeRF-synthetic ("blender") dataset loader for the matrix-native reference.

Reads transforms_{split}.json + RGBA PNG frames, converts each pose from the Blender/OpenGL
camera-to-world convention to our OpenCV world-to-camera (via camera.opengl_c2w_to_opencv_w2c
— the SAME function the camera test pins), downscales, and composites RGBA over white (the
NeRF-synthetic background convention). Returns (cameras, images) ready for train.fit / render.
No COLMAP, no .ply.
"""
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from .camera import Camera, opengl_c2w_to_opencv_w2c


def intrinsics_from_fov(camera_angle_x, W, H):
    """Pinhole intrinsics from horizontal FOV (square pixels). FOV-preserving across resolution."""
    fx = 0.5 * W / math.tan(0.5 * camera_angle_x)
    return fx, fx, W / 2.0, H / 2.0


def composite_over_white(rgba):
    """rgba [...,4] in [0,1] -> rgb [...,3] composited over a white background."""
    rgb, a = rgba[..., :3], rgba[..., 3:4]
    return rgb * a + (1.0 - a)


def _load_image(path, res, dtype, keep_alpha=False):
    img = Image.open(path).convert("RGBA")
    if res is not None and (img.width != res or img.height != res):
        img = img.resize((res, res), Image.BILINEAR)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32)).to(dtype) / 255.0   # [H,W,4]
    return arr if keep_alpha else composite_over_white(arr)                        # [H,W,4] or [H,W,3]


def select_indices(n_frames, indices=None, stride=None, n=None):
    """Deterministic frame selection: explicit indices, or every `stride`-th, capped at `n`."""
    if indices is not None:
        return list(indices)
    idx = list(range(0, n_frames, stride)) if stride else list(range(n_frames))
    return idx[:n] if n is not None else idx


def load_blender(root, split="train", res=200, indices=None, stride=None, n=None,
                 device="cpu", dtype=torch.float32, keep_alpha=False):
    """Load a NeRF-synthetic split. Returns (cameras, images); images are [H,W,3] over white,
    or [H,W,4] RGBA (straight rgb + alpha) when keep_alpha=True (for random-bg compositing)."""
    with open(os.path.join(root, f"transforms_{split}.json")) as f:
        meta = json.load(f)
    frames = meta["frames"]
    idx = select_indices(len(frames), indices, stride, n)

    cameras, images = [], []
    for i in idx:
        fr = frames[i]
        c2w = torch.tensor(fr["transform_matrix"], dtype=dtype, device=device)
        R_v, t_v = opengl_c2w_to_opencv_w2c(c2w)

        fp = fr["file_path"]
        png = fp if fp.endswith(".png") else fp + ".png"
        png = os.path.normpath(os.path.join(root, png))
        img = _load_image(png, res, dtype, keep_alpha=keep_alpha).to(device)   # [H,W,3] or [H,W,4]

        H, W = img.shape[0], img.shape[1]
        fx, fy, cx, cy = intrinsics_from_fov(meta["camera_angle_x"], W, H)
        cameras.append(Camera(R_v, t_v, fx, fy, cx, cy, H, W))
        images.append(img)
    return cameras, images


# --- COLMAP (real scenes: Tanks&Temples / Mip360 / DeepBlending) -----------------------------------
# Reads the SfM output (sparse/0/{cameras,images}.bin) into our OpenCV Camera list + image tensors.
# COLMAP uses the SAME world->cam (R_v,t_v) convention as our Camera, so the mapping is direct.

def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def _read_colmap_cameras(path):
    import struct
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model = struct.unpack("<ii", f.read(8))
            W, H = struct.unpack("<QQ", f.read(16))
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model, 4)
            p = struct.unpack("<%dd" % nparams, f.read(8 * nparams))
            if model in (0, 2):       # SIMPLE_PINHOLE / SIMPLE_RADIAL: f, cx, cy
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:                      # PINHOLE / others: fx, fy, cx, cy
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            cams[cid] = (fx, fy, cx, cy, W, H)
    return cams


def _read_colmap_images(path):
    import struct
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)                                  # image_id
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)                          # skip points2D
            out.append((np.array(q), np.array(t), cid, name.decode()))
    return out


def load_colmap(root, downscale=1, n=None, stride=None, device="cpu", dtype=torch.float32):
    """Load a COLMAP scene (root/sparse/0 + root/images). Returns (cameras, images[H,W,3] in [0,1]).
    downscale halves resolution by that integer factor; n/stride subselect views (perf/eval)."""
    sp = os.path.join(root, "sparse", "0")
    cams = _read_colmap_cameras(os.path.join(sp, "cameras.bin"))
    imgs_meta = sorted(_read_colmap_images(os.path.join(sp, "images.bin")), key=lambda r: r[3])
    idx = select_indices(len(imgs_meta), None, stride, n)
    cameras, images = [], []
    for i in idx:
        q, t, cid, name = imgs_meta[i]
        fx, fy, cx, cy, W, H = cams[cid]
        im = Image.open(os.path.join(root, "images", name)).convert("RGB")
        if downscale > 1:
            W, H = W // downscale, H // downscale
            im = im.resize((W, H), Image.BILINEAR)
            s = 1.0 / downscale
            fx, fy, cx, cy = fx * s, fy * s, cx * s, cy * s
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(dtype).to(device)
        R_v = torch.from_numpy(_qvec2rotmat(q)).to(dtype).to(device)
        t_v = torch.from_numpy(t).to(dtype).to(device)
        cameras.append(Camera(R_v, t_v, float(fx), float(fy), float(cx), float(cy), int(H), int(W)))
        images.append(arr)
    return cameras, images


def load_colmap_points(root, max_pts=None):
    """Read sparse/0/points3D.bin -> (xyz[N,3], rgb[N,3] in [0,1]) for gaussian initialization."""
    import struct
    path = os.path.join(root, "sparse", "0", "points3D.bin")
    xyz, rgb = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                                  # point3D_id
            xyz.append(struct.unpack("<3d", f.read(24)))
            rgb.append(struct.unpack("<3B", f.read(3)))
            f.read(8)                                  # reprojection error
            tlen = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tlen)                           # skip track (image_id,point2D_idx)*tlen
    xyz = np.array(xyz, dtype=np.float32)
    rgb = np.array(rgb, dtype=np.float32) / 255.0
    if max_pts and len(xyz) > max_pts:
        sel = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb
