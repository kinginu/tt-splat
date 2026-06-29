"""Round-trip test for the COLMAP loader (spike/data.py: load_colmap + load_colmap_points). Writes a
synthetic COLMAP scene in the exact binary layout, reads it back, and checks cameras/poses/points.
Validates the struct offsets/types (the real-data run additionally confirms against actual COLMAP output)."""
import os, struct, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import torch
from PIL import Image
from spike import data as colmap_data


def write_scene(root):
    sd = os.path.join(root, "sparse", "0"); os.makedirs(sd); os.makedirs(os.path.join(root, "images"))
    # cameras.bin: 1 PINHOLE (model 1): fx,fy,cx,cy
    with open(os.path.join(sd, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 640, 480))                     # id, model=PINHOLE, W, H
        f.write(struct.pack("<dddd", 500.0, 510.0, 320.0, 240.0))         # fx, fy, cx, cy
    # images.bin: 2 images
    imgs = [(1, (1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3), 1, "a.png"),
            (2, (0.7071, 0.0, 0.7071, 0.0), (1.0, 2.0, 3.0), 1, "b.png")]
    with open(os.path.join(sd, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, q, t, cid, name in imgs:
            f.write(struct.pack("<i", iid)); f.write(struct.pack("<dddd", *q)); f.write(struct.pack("<ddd", *t))
            f.write(struct.pack("<i", cid)); f.write(name.encode() + b"\x00"); f.write(struct.pack("<Q", 0))
    # points3D.bin: 3 points
    pts = [(10, (1.0, 2.0, 3.0), (255, 128, 0)), (11, (-1.0, 0.0, 5.0), (0, 64, 200)),
           (12, (4.0, 4.0, 4.0), (10, 20, 30))]
    with open(os.path.join(sd, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, xyz, rgb in pts:
            f.write(struct.pack("<Q", pid)); f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", *rgb)); f.write(struct.pack("<d", 0.5)); f.write(struct.pack("<Q", 0))
    for name in ("a.png", "b.png"):
        Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8)).save(os.path.join(root, "images", name))
    return imgs, pts


def main():
    with tempfile.TemporaryDirectory() as root:
        imgs, pts = write_scene(root)
        cams, images = colmap_data.load_colmap(root)
        pxyz_np, prgb_np = colmap_data.load_colmap_points(root)
        pxyz, prgb = torch.from_numpy(pxyz_np), torch.from_numpy(prgb_np)
        ok = True
        # camera count + intrinsics
        ok &= len(cams) == 2 and len(images) == 2
        c0 = cams[0]; ok &= (c0.fx, c0.fy, c0.cx, c0.cy) == (500.0, 510.0, 320.0, 240.0)
        ok &= (c0.H, c0.W) == (480, 640)
        # pose: image "a.png" qvec=(1,0,0,0) -> R=I, t=(.1,.2,.3); sorted by name so cams[0]=a
        R0, t0 = c0.R_v.numpy(), c0.t_v.numpy()
        ok &= np.allclose(R0, np.eye(3), atol=1e-6) and np.allclose(t0, [0.1, 0.2, 0.3], atol=1e-6)
        # points round-trip
        ok &= pxyz.shape == (3, 3) and prgb.shape == (3, 3)
        ok &= np.allclose(pxyz.numpy(), [[1, 2, 3], [-1, 0, 5], [4, 4, 4]], atol=1e-6)
        ok &= np.allclose(prgb[0].numpy(), [255 / 255, 128 / 255, 0], atol=1e-3)
        print(f"cams={len(cams)} imgs={len(images)} pts={tuple(pxyz.shape)}  "
              f"intrinsics={(c0.fx, c0.fy, c0.cx, c0.cy)}  R0=I:{np.allclose(R0, np.eye(3), atol=1e-6)}")
        # init_from_points smoke
        from spike.model import GaussianModel
        m = GaussianModel(5)
        m.init_from_points(pxyz, prgb)
        ok &= m.means3d.shape == (5, 3) and torch.isfinite(m.log_scales).all()
        print("ROUNDTRIP", "PASS" if ok else "FAIL")
        return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
