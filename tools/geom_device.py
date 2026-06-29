"""Port the geometry fwd+bwd to ttnn device elementwise (the math is verified host-side in
geom_bwd.py). This attacks the now-dominant host geometry cost (after bin-every-5 made binning
non-dominant). Camera is per-view scalar (single-view verify; batch/trace later). Masks (zmask, dmask)
are read back to host for now (small; replaced by device compares when traced).

Oracle: torch autograd of the same geometry. Verifies conic/mu2d (fwd) and gmeans/gquats/glog_scales
(bwd) on silicon.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/geom_device.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike.model import GaussianModel
import geom_bwd as gbh           # host reference (forward + autograd oracle in main)

NEAR, BLUR = 0.2, 0.3


def A(a, b):
    return ttnn.add(a, b)


def M(a, b):
    return ttnn.mul(a, b)


def S(a, b):
    return ttnn.add(a, ttnn.neg(b))


def lin3(c0, x0, c1, x1, c2, x2):
    return A(A(M(x0, c0), M(x1, c1)), M(x2, c2))


def device_fwd_core(cols, Rv, tv, fx, fy, cx, cy, concat_out=True):
    """Pure device math on already-uploaded tensors. cols = (mx,my,mz,qw,qx,qy,qz,sx,sy,sz) device [P,1].
    Rv[i][j]/tv[i]/fx.. may be python scalars OR [P,1] device tensors. Returns conic, mu2d, cache.
    concat_out=False (large-G/scaling path): returns conic=mu2d=None and puts the conic/mu COLUMNS in
    cache (ca,cb,cc,mu2d_x,mu2d_y) so the downstream gather embeds [G,1] columns separately, avoiding the
    per-gaussian [G,k] concat that runs on ~3 cores and overflows L1 at G>~185k."""
    mx, my, mz, qw, qx, qy, qz, sx, sy, sz = cols
    n2 = A(A(M(qw, qw), M(qx, qx)), A(M(qy, qy), M(qz, qz)))
    inv = ttnn.rsqrt(A(n2, 1e-24))                               # 1/||q||
    w, x, y, z = M(qw, inv), M(qx, inv), M(qy, inv), M(qz, inv)
    xx, yy, zz = M(x, x), M(y, y), M(z, z)
    xy, xz, yz = M(x, y), M(x, z), M(y, z)
    wx, wy, wz = M(w, x), M(w, y), M(w, z)
    R = [[A(M(A(yy, zz), -2.0), 1.0), M(S(xy, wz), 2.0), M(A(xz, wy), 2.0)],
         [M(A(xy, wz), 2.0), A(M(A(xx, zz), -2.0), 1.0), M(S(yz, wx), 2.0)],
         [M(S(xz, wy), 2.0), M(A(yz, wx), 2.0), A(M(A(xx, yy), -2.0), 1.0)]]
    Mm = [[lin3(Rv[i][0], R[0][k], Rv[i][1], R[1][k], Rv[i][2], R[2][k]) for k in range(3)] for i in range(3)]
    s2 = [M(sx, sx), M(sy, sy), M(sz, sz)]

    def SC(i, l):
        return A(A(M(M(Mm[i][0], Mm[l][0]), s2[0]), M(M(Mm[i][1], Mm[l][1]), s2[1])),
                 M(M(Mm[i][2], Mm[l][2]), s2[2]))
    SC00, SC01, SC02, SC11, SC12, SC22 = SC(0, 0), SC(0, 1), SC(0, 2), SC(1, 1), SC(1, 2), SC(2, 2)
    mcx = A(lin3(Rv[0][0], mx, Rv[0][1], my, Rv[0][2], mz), tv[0])
    mcy = A(lin3(Rv[1][0], mx, Rv[1][1], my, Rv[1][2], mz), tv[1])
    mcz = A(lin3(Rv[2][0], mx, Rv[2][1], my, Rv[2][2], mz), tv[2])
    z_ = A(ttnn.relu(A(mcz, -NEAR)), NEAR)
    zi = ttnn.reciprocal(z_)
    mu2d_x = A(M(M(mcx, zi), fx), cx)
    mu2d_y = A(M(M(mcy, zi), fy), cy)
    zi2 = M(zi, zi)
    J00, J11 = M(zi, fx), M(zi, fy)
    J02, J12 = M(M(M(mcx, zi2), fx), -1.0), M(M(M(mcy, zi2), fy), -1.0)
    s00 = A(A(A(M(M(J00, J00), SC00), M(M(M(J00, J02), SC02), 2.0)), M(M(J02, J02), SC22)), BLUR)
    s11 = A(A(A(M(M(J11, J11), SC11), M(M(M(J11, J12), SC12), 2.0)), M(M(J12, J12), SC22)), BLUR)
    s01 = A(A(M(M(J00, J11), SC01), M(M(J00, J12), SC02)), A(M(M(J02, J11), SC12), M(M(J02, J12), SC22)))
    det = S(M(s00, s11), M(s01, s01))
    det_c = A(ttnn.relu(A(det, -1e-12)), 1e-12)
    deti = ttnn.reciprocal(det_c)
    ca, cb, cc = M(s11, deti), M(M(s01, deti), -1.0), M(s00, deti)   # conic columns [G,1]
    conic = ttnn.concat([ca, cb, cc], dim=-1) if concat_out else None
    mu2d = ttnn.concat([mu2d_x, mu2d_y], dim=-1) if concat_out else None
    zmask = ttnn.gtz(A(mcz, -NEAR))                              # device mask (no host readback)
    dmask = ttnn.gtz(A(det, -1e-12))
    cache = dict(ca=ca, cb=cb, cc=cc, mu2d_x=mu2d_x, mu2d_y=mu2d_y,   # columns for the large-G gather
                 w=w, x=x, y=y, z=z, inv=inv, R=R, M=Mm, s2=s2, sx=sx, sy=sy, sz=sz,
                 SC=[[SC00, SC01, SC02], [SC01, SC11, SC12], [SC02, SC12, SC22]],
                 mcx=mcx, mcy=mcy, mcz=mcz, zi=zi, J00=J00, J02=J02, J11=J11, J12=J12,
                 s00=s00, s11=s11, s01=s01, det=det, deti=deti, Rv=Rv, fx=fx, fy=fy,
                 zmask=zmask, dmask=dmask)
    return conic, mu2d, cache


def device_fwd(dev, dt, means, quat, scale, Rv, tv, fx, fy, cx, cy):
    def u(col):
        return ttnn.from_torch(col.reshape(-1, 1).contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    cols = (u(means[:, 0]), u(means[:, 1]), u(means[:, 2]),
            u(quat[:, 0]), u(quat[:, 1]), u(quat[:, 2]), u(quat[:, 3]),
            u(scale[:, 0]), u(scale[:, 1]), u(scale[:, 2]))
    return device_fwd_core(cols, Rv, tv, fx, fy, cx, cy)


def device_bwd_core(cache, ga, gb, gc, gmux, gmuy):
    """Pure device math on already-uploaded grads. cache from device_fwd_core; ga,gb,gc,gmux,gmuy [P,1].
    Returns dict of g-col device tensors (gmx..gsz)."""
    c = cache
    zmask_d, dmask_d = c["zmask"], c["dmask"]
    deti, s00, s11, s01 = c["deti"], c["s00"], c["s11"], c["s01"]
    # step 12
    gs11 = M(ga, deti)
    gs01 = M(ttnn.neg(gb), deti)
    gs00 = M(gc, deti)
    gdeti = A(A(M(ga, s11), M(ttnn.neg(gb), s01)), M(gc, s00))
    # step 11
    gdet_c = M(ttnn.neg(gdeti), M(deti, deti))
    gdet = M(gdet_c, dmask_d)
    gs00 = A(gs00, M(gdet, s11))
    gs11 = A(gs11, M(gdet, s00))
    gs01 = A(gs01, M(gdet, M(s01, -2.0)))
    # step 10
    J00, J02, J11, J12 = c["J00"], c["J02"], c["J11"], c["J12"]
    SC = c["SC"]
    SC00, SC01, SC02, SC11, SC12, SC22 = SC[0][0], SC[0][1], SC[0][2], SC[1][1], SC[1][2], SC[2][2]
    gJ00 = A(M(gs00, A(M(M(J00, 2.0), SC00), M(M(J02, 2.0), SC02))),
             M(gs01, A(M(J11, SC01), M(J12, SC02))))
    gJ02 = A(M(gs00, A(M(M(J00, 2.0), SC02), M(M(J02, 2.0), SC22))),
             M(gs01, A(M(J11, SC12), M(J12, SC22))))
    gJ11 = A(M(gs11, A(M(M(J11, 2.0), SC11), M(M(J12, 2.0), SC12))),
             M(gs01, A(M(J00, SC01), M(J02, SC12))))
    gJ12 = A(M(gs11, A(M(M(J11, 2.0), SC12), M(M(J12, 2.0), SC22))),
             M(gs01, A(M(J00, SC02), M(J02, SC22))))
    gSC00 = M(gs00, M(J00, J00))
    gSC11 = M(gs11, M(J11, J11))
    gSC22 = A(A(M(gs00, M(J02, J02)), M(gs11, M(J12, J12))), M(gs01, M(J02, J12)))
    gSC02 = A(M(gs00, M(M(J00, 2.0), J02)), M(gs01, M(J00, J12)))
    gSC12 = A(M(gs11, M(M(J11, 2.0), J12)), M(gs01, M(J02, J11)))
    gSC01 = M(gs01, M(J00, J11))
    # step 9 + mu2d
    fx, fy, mcx, mcy, zi = c["fx"], c["fy"], c["mcx"], c["mcy"], c["zi"]
    # forms below work whether fx/fy are python scalars (single view) or [P,1] tensors (batched)
    gmcx = A(M(M(gmux, zi), fx), M(gJ02, M(M(M(zi, zi), fx), -1.0)))
    gmcy = A(M(M(gmuy, zi), fy), M(gJ12, M(M(M(zi, zi), fy), -1.0)))
    gzi = A(A(A(M(M(gmux, mcx), fx), M(M(gmuy, mcy), fy)), A(M(gJ00, fx), M(gJ11, fy))),
            A(M(gJ02, M(M(M(mcx, zi), fx), -2.0)), M(gJ12, M(M(M(mcy, zi), fy), -2.0))))
    # step 7
    gz_ = M(ttnn.neg(gzi), M(zi, zi))
    gmcz = M(gz_, zmask_d)
    # step 6
    Rv = c["Rv"]
    gmx = A(A(M(gmcx, Rv[0][0]), M(gmcy, Rv[1][0])), M(gmcz, Rv[2][0]))
    gmy = A(A(M(gmcx, Rv[0][1]), M(gmcy, Rv[1][1])), M(gmcz, Rv[2][1]))
    gmz = A(A(M(gmcx, Rv[0][2]), M(gmcy, Rv[1][2])), M(gmcz, Rv[2][2]))
    # step 5: SC -> M, s2
    Mm, s2 = c["M"], c["s2"]
    gM = [[None] * 3 for _ in range(3)]
    gs2 = [None, None, None]
    for k in range(3):
        gM[0][k] = A(A(M(gSC00, M(M(Mm[0][k], 2.0), s2[k])), M(gSC01, M(Mm[1][k], s2[k]))),
                     M(gSC02, M(Mm[2][k], s2[k])))
        gM[1][k] = A(A(M(gSC11, M(M(Mm[1][k], 2.0), s2[k])), M(gSC01, M(Mm[0][k], s2[k]))),
                     M(gSC12, M(Mm[2][k], s2[k])))
        gM[2][k] = A(A(M(gSC22, M(M(Mm[2][k], 2.0), s2[k])), M(gSC02, M(Mm[0][k], s2[k]))),
                     M(gSC12, M(Mm[1][k], s2[k])))
        gs2[k] = A(A(A(M(gSC00, M(Mm[0][k], Mm[0][k])), M(gSC11, M(Mm[1][k], Mm[1][k]))),
                     M(gSC22, M(Mm[2][k], Mm[2][k]))),
                   A(A(M(gSC01, M(Mm[0][k], Mm[1][k])), M(gSC02, M(Mm[0][k], Mm[2][k]))),
                     M(gSC12, M(Mm[1][k], Mm[2][k]))))
    # step 4: s2 -> scale
    gsx, gsy, gsz = M(gs2[0], M(c["sx"], 2.0)), M(gs2[1], M(c["sy"], 2.0)), M(gs2[2], M(c["sz"], 2.0))
    # step 3: M = Rv R -> R
    gR = [[A(A(M(gM[0][k], Rv[0][i]), M(gM[1][k], Rv[1][i])), M(gM[2][k], Rv[2][i])) for k in range(3)]
          for i in range(3)]
    # step 2: R -> w,x,y,z
    w, x, y, z = c["w"], c["x"], c["y"], c["z"]
    gR00, gR01, gR02 = gR[0][0], gR[0][1], gR[0][2]
    gR10, gR11, gR12 = gR[1][0], gR[1][1], gR[1][2]
    gR20, gR21, gR22 = gR[2][0], gR[2][1], gR[2][2]
    gw = A(A(A(M(gR01, M(z, -2.0)), M(gR02, M(y, 2.0))), A(M(gR10, M(z, 2.0)), M(gR12, M(x, -2.0)))),
           A(M(gR20, M(y, -2.0)), M(gR21, M(x, 2.0))))
    gx = A(A(A(M(gR01, M(y, 2.0)), M(gR02, M(z, 2.0))), A(M(gR10, M(y, 2.0)), M(gR11, M(x, -4.0)))),
           A(A(M(gR12, M(w, -2.0)), M(gR20, M(z, 2.0))), A(M(gR21, M(w, 2.0)), M(gR22, M(x, -4.0)))))
    gy = A(A(A(M(gR00, M(y, -4.0)), M(gR01, M(x, 2.0))), A(M(gR02, M(w, 2.0)), M(gR10, M(x, 2.0)))),
           A(A(M(gR12, M(z, 2.0)), M(gR20, M(w, -2.0))), A(M(gR21, M(z, 2.0)), M(gR22, M(y, -4.0)))))
    gz = A(A(A(M(gR00, M(z, -4.0)), M(gR01, M(w, -2.0))), A(M(gR02, M(x, 2.0)), M(gR10, M(w, 2.0)))),
           A(A(M(gR11, M(z, -4.0)), M(gR12, M(y, 2.0))), A(M(gR20, M(x, 2.0)), M(gR21, M(y, 2.0)))))
    # step 1: normalize -> raw q ; gq = (g - qn*dot)*inv
    inv = c["inv"]
    dot = A(A(M(w, gw), M(x, gx)), A(M(y, gy), M(z, gz)))
    gqw = M(S(gw, M(w, dot)), inv)
    gqx = M(S(gx, M(x, dot)), inv)
    gqy = M(S(gy, M(y, dot)), inv)
    gqz = M(S(gz, M(z, dot)), inv)

    return dict(gmx=gmx, gmy=gmy, gmz=gmz, gqw=gqw, gqx=gqx, gqy=gqy, gqz=gqz,
                gsx=gsx, gsy=gsy, gsz=gsz)


def device_bwd(dev, dt, cache, gconic, gmu2d):
    """gconic[P,3], gmu2d[P,2] (torch) -> gmeans[P,3], gquat[P,4], gscale[P,3] (torch)."""
    def u(col):
        return ttnn.from_torch(col.reshape(-1, 1).contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    g = device_bwd_core(cache, u(gconic[:, 0]), u(gconic[:, 1]), u(gconic[:, 2]),
                        u(gmu2d[:, 0]), u(gmu2d[:, 1]))

    def d(t):
        return ttnn.to_torch(t).float().reshape(-1)
    gmeans = torch.stack([d(g["gmx"]), d(g["gmy"]), d(g["gmz"])], dim=-1)
    gquat = torch.stack([d(g["gqw"]), d(g["gqx"]), d(g["gqy"]), d(g["gqz"])], dim=-1)
    gscale = torch.stack([d(g["gsx"]), d(g["gsy"]), d(g["gsz"])], dim=-1)
    return gmeans, gquat, gscale


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def run(dev, dt, name, G=8000, seed=0):
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    means = m.means3d.detach().clone().requires_grad_(True)
    quat = m.quats.detach().clone().requires_grad_(True)
    logs = m.log_scales.detach().clone().requires_grad_(True)
    Rv_t = torch.tensor([[0.9, 0.1, -0.05], [-0.08, 0.95, 0.2], [0.05, -0.18, 0.98]])
    tv_t = torch.tensor([0.1, -0.2, 4.0])
    fx = fy = 128 * 1.2
    cx = cy = 64.0
    Rv = [[float(Rv_t[i, j]) for j in range(3)] for i in range(3)]
    tv = [float(tv_t[i]) for i in range(3)]

    scale = torch.exp(logs)
    conic_o, mu_o, hcache = gbh.fwd(means, quat, scale, Rv, tv, fx, fy, cx, cy)
    gconic = torch.randn_like(conic_o)
    gmu = torch.randn_like(mu_o)
    (conic_o * gconic).sum().add_((mu_o * gmu).sum()).backward()
    gmeans_a, gquat_a, glogs_a = means.grad, quat.grad, logs.grad

    with torch.no_grad():
        sc = torch.exp(logs)
        conic_d, mu_d, cache = device_fwd(dev, dt, means, quat, sc, Rv, tv, fx, fy, cx, cy)
        gm, gq, gs = device_bwd(dev, dt, cache, gconic, gmu)
        glogs_m = gs * sc

    cd, md = ttnn.to_torch(conic_d).float(), ttnn.to_torch(mu_d).float()
    print(f"[{name}] G={G}:")
    print(f"   fwd conic {rel(cd, conic_o):.3e}  mu2d {rel(md, mu_o):.3e}")
    print(f"   gmeans {rel(gm, gmeans_a):.3e}  gquats {rel(gq, gquat_a):.3e}  glog_scales {rel(glogs_m, glogs_a):.3e}")
    worst = max(rel(gm, gmeans_a), rel(gq, gquat_a), rel(glogs_m, glogs_a))
    print(f"   -> worst grad rel {worst:.3e}  ({'PASS' if worst < 0.03 else 'see note'})")


def _cams(N):
    cams = []
    for v in range(N):
        a = 0.3 * (v - N / 2)
        Rv = torch.tensor([[float(torch.cos(torch.tensor(a))), 0.1, float(-torch.sin(torch.tensor(a)))],
                           [-0.08, 0.95, 0.2], [float(torch.sin(torch.tensor(a))), -0.18, 0.98]])
        tv = torch.tensor([0.1 * v, -0.2, 4.0 + 0.1 * v])
        cams.append((Rv, tv, 128 * 1.2, 128 * 1.2, 64.0, 64.0))
    return cams


def run_batched(dev, dt, name, N=6, G=8000, seed=0):
    """Camera-as-tensors, all N views in one batched P=N*G call (the integration-ready path)."""
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    means = m.means3d.detach().clone().requires_grad_(True)
    quat = m.quats.detach().clone().requires_grad_(True)
    logs = m.log_scales.detach().clone().requires_grad_(True)
    cams = _cams(N)
    gconic_all = torch.randn(N * G, 3)
    gmu_all = torch.randn(N * G, 2)

    # oracle: per-view autograd, accumulate param grads
    scale = torch.exp(logs)
    gmeans_a = torch.zeros(G, 3); gquat_a = torch.zeros(G, 4); glogs_a = torch.zeros(G, 3)
    conic_os, mu_os = [], []
    for v, (Rv_t, tv_t, fx, fy, cx, cy) in enumerate(cams):
        means_v = means.detach().clone().requires_grad_(True)
        quat_v = quat.detach().clone().requires_grad_(True)
        logs_v = logs.detach().clone().requires_grad_(True)
        Rv = [[float(Rv_t[i, j]) for j in range(3)] for i in range(3)]
        tv = [float(tv_t[i]) for i in range(3)]
        co, mo, _ = gbh.fwd(means_v, quat_v, torch.exp(logs_v), Rv, tv, fx, fy, cx, cy)
        conic_os.append(co.detach()); mu_os.append(mo.detach())
        (co * gconic_all[v * G:(v + 1) * G]).sum().add_((mo * gmu_all[v * G:(v + 1) * G]).sum()).backward()
        gmeans_a += means_v.grad; gquat_a += quat_v.grad; glogs_a += logs_v.grad
    conic_o = torch.cat(conic_os); mu_o = torch.cat(mu_os)

    # device: tile params, camera-as-tensors [P,1], one batched call
    P = N * G
    means_t = means.detach().repeat(N, 1)
    quat_t = quat.detach().repeat(N, 1)
    scale_t = torch.exp(logs.detach()).repeat(N, 1)

    def ct(vals):  # per-view scalar -> [P,1] device tensor (view-major)
        col = torch.cat([torch.full((G, 1), float(x)) for x in vals])
        return ttnn.from_torch(col.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    Rv_T = [[ct([c[0][i, j] for c in cams]) for j in range(3)] for i in range(3)]
    tv_T = [ct([c[1][i] for c in cams]) for i in range(3)]
    fx_T = ct([c[2] for c in cams]); fy_T = ct([c[3] for c in cams])
    cx_T = ct([c[4] for c in cams]); cy_T = ct([c[5] for c in cams])
    with torch.no_grad():
        conic_d, mu_d, cache = device_fwd(dev, dt, means_t, quat_t, scale_t, Rv_T, tv_T, fx_T, fy_T, cx_T, cy_T)
        gm, gq, gs = device_bwd(dev, dt, cache, gconic_all, gmu_all)
    gm = gm.reshape(N, G, 3).sum(0); gq = gq.reshape(N, G, 4).sum(0)
    glogs_m = (gs.reshape(N, G, 3).sum(0)) * scale

    cd, md = ttnn.to_torch(conic_d).float(), ttnn.to_torch(mu_d).float()
    print(f"[{name} BATCHED N={N}] P={P}:")
    print(f"   fwd conic {rel(cd, conic_o):.3e}  mu2d {rel(md, mu_o):.3e}")
    print(f"   gmeans {rel(gm, gmeans_a):.3e}  gquats {rel(gq, gquat_a):.3e}  glog_scales {rel(glogs_m, glogs_a):.3e}")
    worst = max(rel(gm, gmeans_a), rel(gq, gquat_a), rel(glogs_m, glogs_a))
    print(f"   -> worst grad rel {worst:.3e}  ({'PASS' if worst < 0.03 else 'see note'})")


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        print("== device geometry fwd+bwd vs autograd (oracle) ==\n")
        run(dev, ttnn.float32, "fp32")
        print()
        run_batched(dev, ttnn.float32, "fp32")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
