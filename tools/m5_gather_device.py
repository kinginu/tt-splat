"""On-device per-tile gather + theta_u build. Given device-resident per-gaussian tensors
(conic[G,3], mu2d[G,2], color_o[G,3], o_col[G,1]) and the host-computed bin indices idx[T,K] (binning
stays host), gather each tile's K gaussians ON DEVICE (ttnn.embedding = row lookup) and build
theta_u[T,6,K] (the origin-folded quadratic operands) with elementwise ops -- no host round-trip of the
gathered operands. Oracle: tools/m4_train_binned._operands (the current host gather+build).

The only host<->device traffic this models: read back mu2d[G,2] for binning, upload idx[T,K] (uint32)
and the valid mask -- O(G)/O(T*K), not the O(P*G) hot path.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_gather_device.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import ttnn

from spike import data, forward, geometry
from spike.model import GaussianModel
from m4_train_binned import _operands, assign_bins, TileMap, K_POLY

R_STENCIL = 1


def host_pregather(model, cam, tmap, K):
    """Replicate _operands' pre-gather stage -> the per-gaussian tensors + bin indices."""
    Rm = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), Rm)
    mu2d, conic, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, 0.3, 0.2)
    color = forward.color_from_dc(model.color_dc)
    o = torch.sigmoid(model.opacity_raw)
    keo = (keep.float() * o)[:, None]
    color_o, o_col = keo * color, keo
    idx, valid = assign_bins(mu2d.detach(), keep.detach(), tmap, R_STENCIL, K)
    return (mu2d.detach(), conic.detach(), color_o.detach(), o_col.detach(),
            idx, valid, tmap.origins)


def gather_build_device(dev, dt, mu2d, conic, color_o, o_col, idx, valid, origins, K):
    T = idx.shape[0]

    def u(t):
        return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    def gather(table_t):                                   # [G,D] -> [T,K,D] via row lookup
        w = u(table_t)
        ii = ttnn.from_torch(idx.to(torch.int32).reshape(T, K), dtype=ttnn.uint32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        return ttnn.embedding(ii, w)

    conic_t = gather(conic)                                # [T,K,3]
    mu_t = gather(mu2d)                                    # [T,K,2]
    color_t = gather(color_o)                              # [T,K,3]
    ocol_t = gather(o_col)                                 # [T,K,1]

    # mu = mu_t - origins (per tile, broadcast over K) -- expand on host, upload
    org = u(origins[:, None, :].expand(T, K, 2).contiguous())
    mu = ttnn.add(mu_t, ttnn.neg(org))
    a, b, c = conic_t[:, :, 0:1], conic_t[:, :, 1:2], conic_t[:, :, 2:3]
    mux, muy = mu[:, :, 0:1], mu[:, :, 1:2]
    mux2, muy2, muxy = ttnn.mul(mux, mux), ttnn.mul(muy, muy), ttnn.mul(mux, muy)
    t2 = ttnn.mul(b, 2.0)
    t3 = ttnn.mul(ttnn.add(ttnn.mul(a, mux), ttnn.mul(b, muy)), -2.0)
    t4 = ttnn.mul(ttnn.add(ttnn.mul(b, mux), ttnn.mul(c, muy)), -2.0)
    t5 = ttnn.add(ttnn.add(ttnn.mul(a, mux2), ttnn.mul(ttnn.mul(b, muxy), 2.0)), ttnn.mul(c, muy2))
    tQ = ttnn.concat([a, c, t2, t3, t4, t5], dim=-1)       # [T,K,6]
    bump = torch.zeros(T, K, 6)
    bump[..., 5] = 1.0
    theta_tk6 = ttnn.add(ttnn.mul(tQ, -1.0 / K_POLY), u(bump))
    theta = ttnn.transpose(theta_tk6, 1, 2)                # [T,6,K]
    valid6 = u(valid[:, None, :].expand(T, 6, K).float().contiguous())
    theta = ttnn.mul(theta, valid6)

    vf = u(valid[..., None].float())                       # [T,K,1]
    color_o_t = ttnn.mul(color_t, vf)
    o_col_t = ttnn.mul(ocol_t, vf)
    return (ttnn.to_torch(theta).float(), ttnn.to_torch(color_o_t).float(),
            ttnn.to_torch(o_col_t).float())


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def run(dev, dt, name, scene="data/nerf_synthetic/ficus", res=128, G=8000, K=256, seed=0):
    cams, _ = data.load_blender(scene, "train", res=res, n=1)
    cam = cams[0]
    tmap = TileMap(res, res)
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)

    thU_ref, col_ref, ocol_ref, _ = _operands(m, cam, tmap, R_STENCIL, K)   # host oracle
    mu2d, conic, color_o, o_col, idx, valid, origins = host_pregather(m, cam, tmap, K)
    thU, col, ocol = gather_build_device(dev, dt, mu2d, conic, color_o, o_col, idx, valid, origins, K)

    print(f"[{name}] G={G} res={res} T={tmap.T} K={K}:")
    print(f"    theta_u    rel {rel(thU, thU_ref.detach()):.4f}")
    print(f"    color_o_t  rel {rel(col, col_ref.detach()):.4f}")
    print(f"    o_col_t    rel {rel(ocol, ocol_ref.detach()):.4f}")
    worst = max(rel(thU, thU_ref.detach()), rel(col, col_ref.detach()), rel(ocol, ocol_ref.detach()))
    print(f"    -> worst rel {worst:.4f}  ({'PASS' if worst < 0.03 else 'see note'})")
    return worst


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        print("== on-device gather + theta_u build vs _operands (oracle) ==\n")
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
