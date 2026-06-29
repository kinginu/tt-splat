"""On-device geometry projection (quat->R, cov3d, EWA world->2D conic) -- the per-gaussian
"G-setup" stage (cheap, anywhere) run on the Blackhole so resident params never round-trip to the
host. Oracle: spike/geometry.py (fp32 CPU reference).

Key trick: the camera (R_v, t_v, fx, fy, cx, cy, blur_eps, near) is constant across gaussians within a
view, so every 3x3/2x2 matrix op collapses to ELEMENTWISE expressions on [G,1] tensors with scalar
coefficients -- no tiny per-gaussian matmuls (which waste ttnn's 32x32 tile). cov3d is folded into the
camera-space covariance: Sigma_cam = M diag(s^2) M^T with M = R_v R.

Outputs mu2d [G,2] and conic [G,3] on device; depth is returned to host (keep/binning are host).

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_geom_device.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike import geometry
from spike.model import GaussianModel


def A(a, b):
    return ttnn.add(a, b)


def M(a, b):
    return ttnn.mul(a, b)


def Sub(a, b):
    return ttnn.add(a, ttnn.neg(b))


def lin3(c0, x0, c1, x1, c2, x2):
    """c0*x0 + c1*x1 + c2*x2 with host-scalar c_i and device-tensor x_i."""
    return A(A(M(x0, c0), M(x1, c1)), M(x2, c2))


def geom_device(dev, dt, means3d, quats, scales, R_v, t_v, fx, fy, cx, cy, blur_eps, near):
    """All inputs torch; R_v[3,3], t_v[3], fx.. scalars. Returns (mu2d, conic) as torch via to_torch."""
    def u(col):
        return ttnn.from_torch(col.reshape(-1, 1).contiguous(), dtype=dt,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    mx, my, mz = u(means3d[:, 0]), u(means3d[:, 1]), u(means3d[:, 2])
    qw, qx, qy, qz = u(quats[:, 0]), u(quats[:, 1]), u(quats[:, 2]), u(quats[:, 3])
    sx, sy, sz = u(scales[:, 0]), u(scales[:, 1]), u(scales[:, 2])
    Rv = [[float(R_v[i, j]) for j in range(3)] for i in range(3)]
    tv = [float(t_v[i]) for i in range(3)]

    # --- normalize quaternion: q /= ||q|| ---
    nrm2 = A(A(M(qw, qw), M(qx, qx)), A(M(qy, qy), M(qz, qz)))
    inv = ttnn.rsqrt(A(nrm2, 1e-24))                       # 1/||q||
    w, x, y, z = M(qw, inv), M(qx, inv), M(qy, inv), M(qz, inv)

    # --- R (3x3) from quaternion, elementwise [G,1] ---
    xx, yy, zz = M(x, x), M(y, y), M(z, z)
    xy, xz, yz = M(x, y), M(x, z), M(y, z)
    wx, wy, wz = M(w, x), M(w, y), M(w, z)
    R00 = A(M(A(yy, zz), -2.0), 1.0)
    R11 = A(M(A(xx, zz), -2.0), 1.0)
    R22 = A(M(A(xx, yy), -2.0), 1.0)
    R01 = M(Sub(xy, wz), 2.0)
    R02 = M(A(xz, wy), 2.0)
    R10 = M(A(xy, wz), 2.0)
    R12 = M(Sub(yz, wx), 2.0)
    R20 = M(Sub(xz, wy), 2.0)
    R21 = M(A(yz, wx), 2.0)
    Rg = [[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]

    # --- M = R_v @ R  (R_v scalar) ; Mm[i][k] = sum_j Rv[i][j] R[j][k] ---
    Mm = [[lin3(Rv[i][0], Rg[0][k], Rv[i][1], Rg[1][k], Rv[i][2], Rg[2][k])
           for k in range(3)] for i in range(3)]
    s2 = [M(sx, sx), M(sy, sy), M(sz, sz)]                 # diag(scale^2)

    # --- Sigma_cam = M diag(s2) M^T  (symmetric 3x3) ---
    def SC(i, l):
        return A(A(M(M(Mm[i][0], Mm[l][0]), s2[0]), M(M(Mm[i][1], Mm[l][1]), s2[1])),
                 M(M(Mm[i][2], Mm[l][2]), s2[2]))
    SC00, SC01, SC02 = SC(0, 0), SC(0, 1), SC(0, 2)
    SC11, SC12, SC22 = SC(1, 1), SC(1, 2), SC(2, 2)

    # --- camera-space mean, depth guard, projection ---
    mcx = A(lin3(Rv[0][0], mx, Rv[0][1], my, Rv[0][2], mz), tv[0])
    mcy = A(lin3(Rv[1][0], mx, Rv[1][1], my, Rv[1][2], mz), tv[1])
    mcz = A(lin3(Rv[2][0], mx, Rv[2][1], my, Rv[2][2], mz), tv[2])
    depth = mcz
    z = A(ttnn.relu(A(mcz, -near)), near)                  # clamp(depth, min=near)
    zi = ttnn.reciprocal(z)
    mu2d_x = A(M(M(mcx, zi), fx), cx)
    mu2d_y = A(M(M(mcy, zi), fy), cy)

    # --- J (2x3): J00=fx/z, J02=-fx*mcx/z^2 ; J11=fy/z, J12=-fy*mcy/z^2 ---
    zi2 = M(zi, zi)
    J00 = M(zi, fx)
    J02 = M(M(M(mcx, zi2), fx), -1.0)
    J11 = M(zi, fy)
    J12 = M(M(M(mcy, zi2), fy), -1.0)

    # --- Sigma2d = J Sigma_cam J^T (+blur_eps on diag) ; rows of J have a zero (J01=J10=0) ---
    s00 = A(A(M(M(J00, J00), SC00), M(M(M(J00, J02), SC02), 2.0)), M(M(J02, J02), SC22))
    s11 = A(A(M(M(J11, J11), SC11), M(M(M(J11, J12), SC12), 2.0)), M(M(J12, J12), SC22))
    s01 = A(A(M(M(J00, J11), SC01), M(M(J00, J12), SC02)),
            A(M(M(J02, J11), SC12), M(M(J02, J12), SC22)))
    s00 = A(s00, blur_eps)
    s11 = A(s11, blur_eps)

    # --- conic = Sigma2d^-1 (closed-form 2x2) ---
    det = Sub(M(s00, s11), M(s01, s01))
    det = A(ttnn.relu(A(det, -1e-12)), 1e-12)             # clamp(det, min=1e-12)
    deti = ttnn.reciprocal(det)
    a = M(s11, deti)
    b = M(M(s01, deti), -1.0)
    c = M(s00, deti)

    mu2d = torch.cat([ttnn.to_torch(mu2d_x).float(), ttnn.to_torch(mu2d_y).float()], dim=1)
    conic = torch.cat([ttnn.to_torch(a).float(), ttnn.to_torch(b).float(),
                       ttnn.to_torch(c).float()], dim=1)
    return mu2d, conic, ttnn.to_torch(depth).float().reshape(-1)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def run(dev, dt, name, G=8000, res=128, seed=0):
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    # a representative camera (look down -z, centered)
    R_v = torch.eye(3)
    t_v = torch.tensor([0.0, 0.0, 4.0])
    fx = fy = float(res) * 1.2
    cx = cy = res / 2.0
    blur_eps, near = 0.3, 0.2
    scales = torch.exp(m.log_scales).detach()

    mu_ref, conic_ref, depth_ref, keep_ref = geometry.project_ewa(
        m.means3d.detach(), geometry.cov3d(scales, geometry.quat_to_rotmat(m.quats.detach())),
        R_v, t_v, fx, fy, cx, cy, blur_eps, near)
    mu_d, conic_d, depth_d = geom_device(
        dev, dt, m.means3d.detach(), m.quats.detach(), scales,
        R_v, t_v, fx, fy, cx, cy, blur_eps, near)

    print(f"[{name}] G={G} res={res}:")
    print(f"    mu2d  rel {rel(mu_d, mu_ref):.4f}")
    print(f"    conic rel {rel(conic_d, conic_ref):.4f}")
    print(f"    depth rel {rel(depth_d, depth_ref):.4f}")
    keep_d = depth_d > near
    agree = (keep_d == keep_ref).float().mean().item()
    print(f"    keep(host from device depth) agreement {agree*100:.1f}%")
    worst = max(rel(mu_d, mu_ref), rel(conic_d, conic_ref))
    print(f"    -> mu/conic worst rel {worst:.4f}  ({'PASS' if worst < 0.03 else 'see note'})")
    return worst


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        print("== on-device geometry projection vs spike/geometry.py (oracle) ==\n")
        run(dev, ttnn.bfloat16, "bf16")
        print()
        try:
            run(dev, ttnn.float32, "fp32")
        except Exception as e:  # noqa: BLE001
            print("  fp32 path:", type(e).__name__, str(e)[:100])
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
