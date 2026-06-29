"""Host-side win: vectorize the binning. tools/m4_train_binned.assign_bins is a python loop over T
tiles (each a full-G mask + topk) ~10.8 ms/view -- the dominant cost in the (host-bound) training loop.
This replaces it with a fully-vectorized scatter: expand each gaussian to its
(2R+1)^2 stencil tiles -> (tile, gaussian, dist^2) pairs -> sort by (tile, dist) -> per-tile rank<K keeps
the K nearest -> scatter into idx[T,K]. WSR is order-independent, so only the per-tile SET must match the
oracle (order within a tile is free). Runs on host (CPU) -- no device.

Oracle: m4_train_binned.assign_bins (per-tile set equality). Metric: ms/view, both on CPU.

Run (no device needed):
    podman run --rm -v $PWD:/workspace -w /workspace tt-splat:dev python3 tools/m5_binning_vec.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from spike import geometry
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins


@torch.no_grad()
def assign_bins_vec(mu2d, keep, tmap, R, K):
    """Vectorized equivalent of assign_bins. idx[T,K] long, valid[T,K] bool. Per-tile SET matches."""
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    G = mu2d.shape[0]
    dev = mu2d.device
    gx = (mu2d[:, 0] / 16).floor().long().clamp(0, ntx - 1)
    gy = (mu2d[:, 1] / 16).floor().long().clamp(0, nty - 1)

    offs = torch.arange(-R, R + 1, device=dev)
    dxs, dys = torch.meshgrid(offs, offs, indexing="ij")
    dxs, dys = dxs.reshape(-1), dys.reshape(-1)                       # [S]
    tx = gx[:, None] + dxs[None, :]                                   # [G,S]
    ty = gy[:, None] + dys[None, :]
    vp = keep[:, None] & (tx >= 0) & (tx < ntx) & (ty >= 0) & (ty < nty)
    tile = ty * ntx + tx
    tcx, tcy = (tx.float() + 0.5) * 16.0, (ty.float() + 0.5) * 16.0
    dist = (mu2d[:, 0:1] - tcx) ** 2 + (mu2d[:, 1:2] - tcy) ** 2
    gid = torch.arange(G, device=dev)[:, None].expand(G, tx.shape[1])

    m = vp.reshape(-1)
    tile_f, gid_f, dist_f = tile.reshape(-1)[m], gid.reshape(-1)[m], dist.reshape(-1)[m]
    key = tile_f.to(torch.float64) * 1e9 + dist_f.to(torch.float64)   # tile primary, dist secondary
    order = torch.argsort(key)
    tile_s, gid_s = tile_f[order], gid_f[order]
    counts = torch.bincount(tile_s, minlength=T)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(tile_s.numel(), device=dev) - starts[tile_s]
    sel = rank < K
    idx = torch.zeros(T, K, dtype=torch.long, device=dev)
    valid = torch.zeros(T, K, dtype=torch.bool, device=dev)
    idx[tile_s[sel], rank[sel]] = gid_s[sel]
    valid[tile_s[sel], rank[sel]] = True
    return idx, valid


def check(res=128, G=8000, K=256, R=1, seed=0):
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    R_v, t_v = torch.eye(3), torch.tensor([0.0, 0.0, 4.0])
    fx = fy = res * 1.2
    cov = geometry.cov3d(torch.exp(m.log_scales), geometry.quat_to_rotmat(m.quats))
    mu2d, conic, depth, keep = geometry.project_ewa(m.means3d, cov, R_v, t_v, fx, fy, res / 2, res / 2, 0.3, 0.2)
    mu2d, keep = mu2d.detach(), keep.detach()
    tmap = TileMap(res, res)

    io, vo = assign_bins(mu2d, keep, tmap, R, K)
    iv, vv = assign_bins_vec(mu2d, keep, tmap, R, K)
    mism = 0
    for t in range(tmap.T):
        so = set(io[t][vo[t]].tolist())
        sv = set(iv[t][vv[t]].tolist())
        if so != sv:
            mism += 1
    occ = vo.sum().item()
    print(f"res={res} G={G} K={K} R={R}: T={tmap.T}, total per-tile slots filled {occ}")
    print(f"   per-tile SET match: {tmap.T - mism}/{tmap.T} tiles  ({'PASS' if mism == 0 else f'{mism} MISMATCH'})")
    print(f"   count-per-tile match: {'yes' if torch.equal(vo.sum(1), vv.sum(1)) else 'NO'}")

    def timed(fn, reps=30):
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3
    t_loop = timed(lambda: assign_bins(mu2d, keep, tmap, R, K))
    t_vec = timed(lambda: assign_bins_vec(mu2d, keep, tmap, R, K))
    print(f"   time: loop {t_loop:.2f} ms  ->  vectorized {t_vec:.2f} ms  ({t_loop / t_vec:.1f}x faster)")


def main():
    print("== binning vectorization vs assign_bins (oracle: per-tile set) ==")
    for res, G, K in ((128, 8000, 256), (96, 4000, 128)):
        check(res, G, K)
        print()


if __name__ == "__main__":
    main()
