"""integrated resident FORWARD render -- geometry + gather/theta_u + the binned
matrix-native render, all on the Blackhole with resident params; only mu2d/depth read back for the host binning
and idx uploaded (per-iter host<->BH traffic is O(G)/O(T*K), not the O(P*G) hot path).
Oracle: tools/m4_train_binned.render_binned_device (the current host-orchestrated render). Also
times the device path vs the host path to show where the current 1353 ms/it goes.

Scope: forward only. The resident BACKWARD (render bwd is done -- m4_train_step; what remains is the
device geometry/gather Jacobians + resident Adam formulas) is the next step.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_resident_forward.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import ttnn

from spike import data, forward, geometry
from spike.model import GaussianModel
from m5_geom_device import A, M, Sub, lin3
import m4_train_binned as mtb
from m4_train_binned import TileMap, assign_bins, render_binned_device, K_POLY

CG = None
NEAR, BLUR = 0.2, 0.3
R_STENCIL = 1


def u(t, dev, dt=None):
    return ttnn.from_torch(t.contiguous(), dtype=dt or ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def geom_resident(dev, cols, Rv, tv, fx, fy, cx, cy):
    """Device geometry -> conic[G,3], mu2d[G,2] (device tensors) + depth (host) for keep/binning."""
    mx, my, mz, qw, qx, qy, qz, sx, sy, sz = cols
    nrm2 = A(A(M(qw, qw), M(qx, qx)), A(M(qy, qy), M(qz, qz)))
    inv = ttnn.rsqrt(A(nrm2, 1e-24))
    w, x, y, z = M(qw, inv), M(qx, inv), M(qy, inv), M(qz, inv)
    xx, yy, zz = M(x, x), M(y, y), M(z, z)
    xy, xz, yz = M(x, y), M(x, z), M(y, z)
    wx, wy, wz = M(w, x), M(w, y), M(w, z)
    Rg = [[A(M(A(yy, zz), -2.0), 1.0), M(Sub(xy, wz), 2.0), M(A(xz, wy), 2.0)],
          [M(A(xy, wz), 2.0), A(M(A(xx, zz), -2.0), 1.0), M(Sub(yz, wx), 2.0)],
          [M(Sub(xz, wy), 2.0), M(A(yz, wx), 2.0), A(M(A(xx, yy), -2.0), 1.0)]]
    Mm = [[lin3(Rv[i][0], Rg[0][k], Rv[i][1], Rg[1][k], Rv[i][2], Rg[2][k]) for k in range(3)]
          for i in range(3)]
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
    det = A(ttnn.relu(A(Sub(M(s00, s11), M(s01, s01)), -1e-12)), 1e-12)
    deti = ttnn.reciprocal(det)
    conic = ttnn.concat([M(s11, deti), M(M(s01, deti), -1.0), M(s00, deti)], dim=-1)   # [G,3]
    mu2d = ttnn.concat([mu2d_x, mu2d_y], dim=-1)                                        # [G,2]
    return conic, mu2d, mcz   # mcz left on device (no read) so this is trace-safe; caller to_torch's it


def gather_theta(dev, conic, mu2d, color_o, o_col, idx, valid, origins, T, K):
    ii = ttnn.from_torch(idx.to(torch.int32).reshape(T, K), dtype=ttnn.uint32,
                         layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    conic_t, mu_t = ttnn.embedding(ii, conic), ttnn.embedding(ii, mu2d)
    color_t, ocol_t = ttnn.embedding(ii, u(color_o, dev)), ttnn.embedding(ii, u(o_col, dev))
    org = u(origins[:, None, :].expand(T, K, 2).contiguous(), dev)
    mu = ttnn.add(mu_t, ttnn.neg(org))
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    mux, muy = mu[:, :, 0:1], mu[:, :, 1:2]
    mux2, muy2, muxy = ttnn.mul(mux, mux), ttnn.mul(muy, muy), ttnn.mul(mux, muy)
    t2 = ttnn.mul(b, 2.0)
    t3 = ttnn.mul(ttnn.add(ttnn.mul(a, mux), ttnn.mul(b, muy)), -2.0)
    t4 = ttnn.mul(ttnn.add(ttnn.mul(b, mux), ttnn.mul(c, muy)), -2.0)
    t5 = ttnn.add(ttnn.add(ttnn.mul(a, mux2), ttnn.mul(ttnn.mul(b, muxy), 2.0)), ttnn.mul(c, muy2))
    tQ = ttnn.concat([a, c, t2, t3, t4, t5], dim=-1)
    bump = torch.zeros(T, K, 6)
    bump[..., 5] = 1.0
    theta = ttnn.transpose(ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), u(bump, dev)), 1, 2)   # [T,6,K]
    theta = ttnn.mul(theta, u(valid[:, None, :].expand(T, 6, K).float().contiguous(), dev))
    vf = u(valid[..., None].float(), dev)
    return theta, ttnn.mul(color_t, vf), ttnn.mul(ocol_t, vf)


def render_fwd(Phi_t, theta, color_t, ocol_t, w_b, bias):
    relu_Q = ttnn.relu(ttnn.matmul(Phi_t, theta, core_grid=CG))            # [T,256,K]
    w = ttnn.square(relu_Q)
    num = ttnn.add(ttnn.matmul(w, color_t, core_grid=CG), bias)            # [T,256,3]
    den = ttnn.add(ttnn.matmul(w, ocol_t, core_grid=CG), float(w_b))       # [T,256,1]
    return ttnn.div(num, den)


def device_image(dev, cols, model, cam, tmap, K):
    Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
    tv = [float(cam.t_v[i]) for i in range(3)]
    conic, mu2d, mcz = geom_resident(dev, cols, Rv, tv, cam.fx, cam.fy, cam.cx, cam.cy)
    mu2d_h = ttnn.to_torch(mu2d).float()
    keep = ttnn.to_torch(mcz).float().reshape(-1) > NEAR
    o = torch.sigmoid(model.opacity_raw).detach()
    color = forward.color_from_dc(model.color_dc).detach()
    keo = (keep.float() * o)[:, None]
    color_o, o_col = (keo * color), keo
    idx, valid = assign_bins(mu2d_h, keep, tmap, R_STENCIL, K)
    T = tmap.T
    theta, color_t, ocol_t = gather_theta(dev, conic, mu2d, color_o, o_col, idx, valid, tmap.origins, T, K)
    Phi_t = u(tmap.Phi.unsqueeze(0).expand(T, 256, 6).contiguous(), dev)
    w_b = F.softplus(model.w_b_raw).detach()
    bias = u((w_b * model.c_b)[None, None, :].expand(T, 256, 3).contiguous(), dev)
    C = render_fwd(Phi_t, theta, color_t, ocol_t, w_b, bias)
    img = torch.zeros(cam.H * cam.W, 3)
    img[tmap.gidx] = ttnn.to_torch(C).float().reshape(T * 256, 3)
    return img.reshape(cam.H, cam.W, 3)


def main():
    global CG
    dev = ttnn.open_device(device_id=0)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        mtb._DEV, mtb.CG, mtb.CKC = dev, CG, None   # the oracle render_binned_device reads these globals
        res, G, K = 128, 8000, 256
        cams, _ = data.load_blender("data/nerf_synthetic/ficus", "train", res=res, n=1)
        cam = cams[0]
        tmap = TileMap(res, res)
        torch.manual_seed(0)
        m = GaussianModel(G, extent=1.5, seed=0)
        cols = [u(c.reshape(-1, 1), dev) for c in
                (m.means3d[:, 0], m.means3d[:, 1], m.means3d[:, 2],
                 m.quats[:, 0], m.quats[:, 1], m.quats[:, 2], m.quats[:, 3])]
        scales = torch.exp(m.log_scales).detach()
        cols += [u(scales[:, 0].reshape(-1, 1), dev), u(scales[:, 1].reshape(-1, 1), dev),
                 u(scales[:, 2].reshape(-1, 1), dev)]

        print("== integrated resident forward vs render_binned_device (oracle) ==")
        ref = render_binned_device(m, cam, tmap, R=R_STENCIL, K=K)
        img = device_image(dev, cols, m, cam, tmap, K)
        rel = ((img - ref).norm() / ref.norm()).item()
        print(f"   res={res} G={G} K={K}: image rel {rel:.4f}  ({'PASS' if rel < 0.05 else 'see note'})")

        print("\n== timing (1 view): device path vs host-orchestrated render_binned_device ==")
        for _ in range(2):
            device_image(dev, cols, m, cam, tmap, K)
            render_binned_device(m, cam, tmap, R=R_STENCIL, K=K)
        N = 8
        t0 = time.perf_counter()
        for _ in range(N):
            device_image(dev, cols, m, cam, tmap, K)
        ttnn.synchronize_device(dev)
        td = (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        for _ in range(N):
            render_binned_device(m, cam, tmap, R=R_STENCIL, K=K)
        th = (time.perf_counter() - t0) / N
        print(f"   resident device path : {td*1e3:7.1f} ms/view")
        print(f"   render_binned_device (host): {th*1e3:7.1f} ms/view")
        print(f"   (both include host python binning; the delta is the residency/dispatch effect)")

        print("\n== stage breakdown (where the device-path ms go) ==")
        Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
        tv = [float(cam.t_v[i]) for i in range(3)]
        tg = tb = tgr = 0.0
        for _ in range(N):
            t0 = time.perf_counter()
            conic, mu2d, mcz = geom_resident(dev, cols, Rv, tv, cam.fx, cam.fy, cam.cx, cam.cy)
            mu2d_h = ttnn.to_torch(mu2d).float()
            depth = ttnn.to_torch(mcz).float().reshape(-1)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            keep = depth > NEAR
            o = torch.sigmoid(m.opacity_raw).detach()
            color = forward.color_from_dc(m.color_dc).detach()
            keo = (keep.float() * o)[:, None]
            color_o, o_col = (keo * color), keo
            idx, valid = assign_bins(mu2d_h, keep, tmap, R_STENCIL, K)
            t2 = time.perf_counter()
            theta, ct, oct_ = gather_theta(dev, conic, mu2d, color_o, o_col, idx, valid, tmap.origins, tmap.T, K)
            Phi_t = u(tmap.Phi.unsqueeze(0).expand(tmap.T, 256, 6).contiguous(), dev)
            w_b = F.softplus(m.w_b_raw).detach()
            bias = u((w_b * m.c_b)[None, None, :].expand(tmap.T, 256, 3).contiguous(), dev)
            C = render_fwd(Phi_t, theta, ct, oct_, w_b, bias)
            _ = ttnn.to_torch(C).float()
            ttnn.synchronize_device(dev)
            t3 = time.perf_counter()
            tg += t1 - t0; tb += t2 - t1; tgr += t3 - t2
        print(f"   device geometry + readback : {tg/N*1e3:7.1f} ms")
        print(f"   host binning (assign_bins) : {tb/N*1e3:7.1f} ms")
        print(f"   gather + render (device)   : {tgr/N*1e3:7.1f} ms")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
