"""Manual geometry BACKWARD (the geometry Jacobian), step 1 = derive in
pure torch (no autograd) and verify vs autograd. Once correct it ports to ttnn device elementwise (like
the geometry forward). Camera (Rv,tv,fx..) is per-view scalar. Inputs params (means, quats, log_scales) ->
conic[G,3], mu2d[G,2]; backward: (gconic, gmu2d) -> (gmeans, gquats, glog_scales).

Oracle: torch autograd of the same forward. Host-only.

Run (no device):
    podman run --rm -v $PWD:/workspace -w /workspace tt-splat:dev python3 tools/geom_bwd.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from spike import geometry
from spike.model import GaussianModel

NEAR, BLUR = 0.2, 0.3


def fwd(means, quat, scale, Rv, tv, fx, fy, cx, cy):
    """Manual forward, returns conic[G,3], mu2d[G,2], and a cache of intermediates for backward."""
    mx, my, mz = means[:, 0], means[:, 1], means[:, 2]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sx, sy, sz = scale[:, 0], scale[:, 1], scale[:, 2]
    n = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz).clamp(min=1e-12)
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    M = [[Rv[i][0] * R[0][k] + Rv[i][1] * R[1][k] + Rv[i][2] * R[2][k] for k in range(3)] for i in range(3)]
    s2 = [sx * sx, sy * sy, sz * sz]

    def SC(i, l):
        return M[i][0] * M[l][0] * s2[0] + M[i][1] * M[l][1] * s2[1] + M[i][2] * M[l][2] * s2[2]
    SC00, SC01, SC02, SC11, SC12, SC22 = SC(0, 0), SC(0, 1), SC(0, 2), SC(1, 1), SC(1, 2), SC(2, 2)
    mcx = Rv[0][0] * mx + Rv[0][1] * my + Rv[0][2] * mz + tv[0]
    mcy = Rv[1][0] * mx + Rv[1][1] * my + Rv[1][2] * mz + tv[1]
    mcz = Rv[2][0] * mx + Rv[2][1] * my + Rv[2][2] * mz + tv[2]
    zmask = (mcz > NEAR).float()
    z_ = mcz.clamp(min=NEAR)
    zi = 1.0 / z_
    mu2d = torch.stack([fx * mcx * zi + cx, fy * mcy * zi + cy], dim=-1)
    J00, J11 = fx * zi, fy * zi
    J02, J12 = -fx * mcx * zi * zi, -fy * mcy * zi * zi
    s00 = J00 * J00 * SC00 + 2 * J00 * J02 * SC02 + J02 * J02 * SC22 + BLUR
    s11 = J11 * J11 * SC11 + 2 * J11 * J12 * SC12 + J12 * J12 * SC22 + BLUR
    s01 = J00 * J11 * SC01 + J00 * J12 * SC02 + J02 * J11 * SC12 + J02 * J12 * SC22
    det = s00 * s11 - s01 * s01
    dmask = (det > 1e-12).float()
    det_c = det.clamp(min=1e-12)
    deti = 1.0 / det_c
    conic = torch.stack([s11 * deti, -s01 * deti, s00 * deti], dim=-1)
    cache = dict(w=w, x=x, y=y, z=z, n=n, R=R, M=M, s2=s2, sx=sx, sy=sy, sz=sz,
                 SC=[[SC00, SC01, SC02], [SC01, SC11, SC12], [SC02, SC12, SC22]],
                 mcx=mcx, mcy=mcy, zmask=zmask, z_=z_, zi=zi,
                 J00=J00, J02=J02, J11=J11, J12=J12, s00=s00, s11=s11, s01=s01,
                 det=det, dmask=dmask, det_c=det_c, deti=deti, Rv=Rv, fx=fx, fy=fy)
    return conic, mu2d, cache


def bwd(cache, ga, gb, gc, gmux, gmuy):
    c = cache
    deti, s00, s11, s01 = c["deti"], c["s00"], c["s11"], c["s01"]
    # step 12: conic
    gs11 = ga * deti
    gs01 = -gb * deti
    gs00 = gc * deti
    gdeti = ga * s11 + (-gb) * s01 + gc * s00
    # step 11: det
    gdet_c = -gdeti * deti * deti
    gdet = gdet_c * c["dmask"]
    gs00 = gs00 + gdet * s11
    gs11 = gs11 + gdet * s00
    gs01 = gs01 + gdet * (-2 * s01)
    # step 10: Sigma2d -> J, SC
    J00, J02, J11, J12 = c["J00"], c["J02"], c["J11"], c["J12"]
    SC = c["SC"]
    SC00, SC01, SC02, SC11, SC12, SC22 = SC[0][0], SC[0][1], SC[0][2], SC[1][1], SC[1][2], SC[2][2]
    gJ00 = gs00 * (2 * J00 * SC00 + 2 * J02 * SC02) + gs01 * (J11 * SC01 + J12 * SC02)
    gJ02 = gs00 * (2 * J00 * SC02 + 2 * J02 * SC22) + gs01 * (J11 * SC12 + J12 * SC22)
    gJ11 = gs11 * (2 * J11 * SC11 + 2 * J12 * SC12) + gs01 * (J00 * SC01 + J02 * SC12)
    gJ12 = gs11 * (2 * J11 * SC12 + 2 * J12 * SC22) + gs01 * (J00 * SC02 + J02 * SC22)
    gSC00 = gs00 * J00 * J00
    gSC11 = gs11 * J11 * J11
    gSC22 = gs00 * J02 * J02 + gs11 * J12 * J12 + gs01 * J02 * J12
    gSC02 = gs00 * 2 * J00 * J02 + gs01 * J00 * J12
    gSC12 = gs11 * 2 * J11 * J12 + gs01 * J02 * J11
    gSC01 = gs01 * J00 * J11
    # step 9 + mu2d: J, mu2d -> mcx, mcy, zi
    fx, fy = c["fx"], c["fy"]
    mcx, mcy, zi = c["mcx"], c["mcy"], c["zi"]
    gmcx = gmux * fx * zi + gJ02 * (-fx * zi * zi)
    gmcy = gmuy * fy * zi + gJ12 * (-fy * zi * zi)
    gzi = (gmux * fx * mcx + gmuy * fy * mcy
           + gJ00 * fx + gJ11 * fy
           + gJ02 * (-fx * mcx * 2 * zi) + gJ12 * (-fy * mcy * 2 * zi))
    # step 7: zi -> z_ -> mcz
    gz_ = -gzi * zi * zi
    gmcz = gz_ * c["zmask"]
    # step 6: mu_cam -> means
    Rv = c["Rv"]
    gmx = gmcx * Rv[0][0] + gmcy * Rv[1][0] + gmcz * Rv[2][0]
    gmy = gmcx * Rv[0][1] + gmcy * Rv[1][1] + gmcz * Rv[2][1]
    gmz = gmcx * Rv[0][2] + gmcy * Rv[1][2] + gmcz * Rv[2][2]
    # step 5: SC -> M, s2  (explicit unique-entry formulas; SC symmetric so use the 6 unique entries)
    M, s2 = c["M"], c["s2"]
    gM = [[torch.zeros_like(mcx) for _ in range(3)] for _ in range(3)]
    gs2 = [torch.zeros_like(mcx) for _ in range(3)]
    # diagonal SC(k,k) = sum_j M[k][j]^2 s2[j]
    for (gg, i) in ((gSC00, 0), (gSC11, 1), (gSC22, 2)):
        for k in range(3):
            gM[i][k] = gM[i][k] + gg * 2 * M[i][k] * s2[k]
            gs2[k] = gs2[k] + gg * M[i][k] * M[i][k]
    # off-diagonal SC(i,l) = sum_k M[i][k] M[l][k] s2[k], i<l
    for (gg, i, l) in ((gSC01, 0, 1), (gSC02, 0, 2), (gSC12, 1, 2)):
        for k in range(3):
            gM[i][k] = gM[i][k] + gg * M[l][k] * s2[k]
            gM[l][k] = gM[l][k] + gg * M[i][k] * s2[k]
            gs2[k] = gs2[k] + gg * M[i][k] * M[l][k]
    # step 4: s2 -> scale
    gsx = gs2[0] * 2 * c["sx"]; gsy = gs2[1] * 2 * c["sy"]; gsz = gs2[2] * 2 * c["sz"]
    # step 3: M = Rv R -> R
    gR = [[Rv[0][i] * gM[0][k] + Rv[1][i] * gM[1][k] + Rv[2][i] * gM[2][k] for k in range(3)] for i in range(3)]
    # step 2: R -> w,x,y,z
    w, x, y, z = c["w"], c["x"], c["y"], c["z"]
    gR00, gR01, gR02 = gR[0][0], gR[0][1], gR[0][2]
    gR10, gR11, gR12 = gR[1][0], gR[1][1], gR[1][2]
    gR20, gR21, gR22 = gR[2][0], gR[2][1], gR[2][2]
    gw = gR01 * (-2 * z) + gR02 * (2 * y) + gR10 * (2 * z) + gR12 * (-2 * x) + gR20 * (-2 * y) + gR21 * (2 * x)
    gx = (gR01 * (2 * y) + gR02 * (2 * z) + gR10 * (2 * y) + gR11 * (-4 * x) + gR12 * (-2 * w)
          + gR20 * (2 * z) + gR21 * (2 * w) + gR22 * (-4 * x))
    gy = (gR00 * (-4 * y) + gR01 * (2 * x) + gR02 * (2 * w) + gR10 * (2 * x) + gR12 * (2 * z)
          + gR20 * (-2 * w) + gR21 * (2 * z) + gR22 * (-4 * y))
    gz = (gR00 * (-4 * z) + gR01 * (-2 * w) + gR02 * (2 * x) + gR10 * (2 * w) + gR11 * (-4 * z)
          + gR12 * (2 * y) + gR20 * (2 * x) + gR21 * (2 * y))
    # step 1: normalize q -> raw q
    n = c["n"]
    dot = w * gw + x * gx + y * gy + z * gz
    gqw = (gw - w * dot) / n
    gqx = (gx - x * dot) / n
    gqy = (gy - y * dot) / n
    gqz = (gz - z * dot) / n
    gmeans = torch.stack([gmx, gmy, gmz], dim=-1)
    gquat = torch.stack([gqw, gqx, gqy, gqz], dim=-1)
    gscale = torch.stack([gsx, gsy, gsz], dim=-1)
    return gmeans, gquat, gscale


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def main():
    torch.manual_seed(0)
    G = 8000
    m = GaussianModel(G, extent=1.5, seed=0)
    means = m.means3d.detach().clone().requires_grad_(True)
    quat = m.quats.detach().clone().requires_grad_(True)
    logs = m.log_scales.detach().clone().requires_grad_(True)
    Rv_t = torch.tensor([[0.9, 0.1, -0.05], [-0.08, 0.95, 0.2], [0.05, -0.18, 0.98]])  # arbitrary-ish
    tv_t = torch.tensor([0.1, -0.2, 4.0])
    fx = fy = 128 * 1.2
    cx = cy = 64.0
    Rv = [[float(Rv_t[i, j]) for j in range(3)] for i in range(3)]
    tv = [float(tv_t[i]) for i in range(3)]

    # autograd oracle
    scale = torch.exp(logs)
    conic_o, mu_o, _ = fwd(means, quat, scale, Rv, tv, fx, fy, cx, cy)
    gconic = torch.randn_like(conic_o)
    gmu = torch.randn_like(mu_o)
    (conic_o * gconic).sum().add_((mu_o * gmu).sum()).backward()
    gmeans_a, gquat_a, glogs_a = means.grad.clone(), quat.grad.clone(), logs.grad.clone()

    # manual
    with torch.no_grad():
        scale2 = torch.exp(logs)
        conic_m, mu_m, cache = fwd(means, quat, scale2, Rv, tv, fx, fy, cx, cy)
        gmeans_m, gquat_m, gscale_m = bwd(cache, gconic[:, 0], gconic[:, 1], gconic[:, 2], gmu[:, 0], gmu[:, 1])
        glogs_m = gscale_m * scale2   # scale = exp(log) -> dscale/dlog = scale

    print("== manual geometry backward vs autograd (oracle) ==")
    print(f"   fwd match: conic {rel(conic_m, conic_o):.2e}  mu2d {rel(mu_m, mu_o):.2e}")
    print(f"   gmeans rel {rel(gmeans_m, gmeans_a):.3e}")
    print(f"   gquats rel {rel(gquat_m, gquat_a):.3e}")
    print(f"   glog_scales rel {rel(glogs_m, glogs_a):.3e}")
    worst = max(rel(gmeans_m, gmeans_a), rel(gquat_m, gquat_a), rel(glogs_m, glogs_a))
    print(f"   -> worst grad rel {worst:.3e}  ({'PASS' if worst < 1e-4 else 'FAIL'})")


if __name__ == "__main__":
    main()
