"""VALIDATED reference (pure torch) for the DEVICE inv-build = the spec the ttnn port follows.

Sync-removal frontier: to kill the per-iter binning sync we must build inv[G,SMAX] (gaussian->slot
positions, consumed by the device grad-scatter scatter_dev) ON DEVICE, replacing host build_inv +
the idx download it forces. Approach (dodges the sort/counter traps): rank[T,G] scatter (rank[t,idx[t,k]]=k,
single-destination) + 9 stencil flat-gathers (each gaussian's <=9 candidate tiles are DETERMINISTIC at R=1).
This file is that approach in pure torch, PROVEN to reproduce build_inv's per-gaussian SET (scatter_dev sums
the slots, so set-equality suffices) on res{800,128,64}/G{1k,2k}: ALL PASS, max <=9 slots/gaussian.

ttnn-port plan (tools/bin_device.py, next): rank scatter = ttnn.scatter into [T,G+1] (invalid slots
redirected to a dummy column G; use stride G+1 in the flat index so no tile-unaligned slice); flat-gather =
ttnn.embedding on the flattened rank; index arithmetic on [1,G] fp32 (clamp/lt/floor). Known risks to resolve
on silicon: scatter index layout, [T,G+1] reshape across TILE padding. Oracle = m9_scatter_oracle.build_inv.

Run (pure torch, no device): python3 tools/bin_inv_ref.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import torch
from spike import geometry
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins
from m9_scatter_oracle import build_inv


def build_inv_device_ref(idx, valid, gx, gy, tmap, G, K):
    """Mirrors the planned ttnn ops: rank[T,G] scatter, then per-stencil-offset flat-gather."""
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    SMAX, sentinel = 9, T * K
    # rank[t,g] = slot k where idx[t,k]=g (valid); else K (= not selected)
    rank = torch.full((T, G), K, dtype=torch.long)
    tt = torch.arange(T)[:, None].expand(T, K)
    kk = torch.arange(K)[None, :].expand(T, K)
    vm = valid.bool()
    rank[tt[vm], idx[vm]] = kk[vm]
    # 9 stencil offsets: inv[g, j] = t_j*K + rank[t_j, g] if g selected in its j-th stencil tile
    inv = torch.full((G, SMAX), sentinel, dtype=torch.long)
    g = torch.arange(G)
    j = 0
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            txj, tyj = gx + ox, gy + oy
            inb = (txj >= 0) & (txj < ntx) & (tyj >= 0) & (tyj < nty)
            tj = tyj.clamp(0, nty - 1) * ntx + txj.clamp(0, ntx - 1)
            r = rank[tj, g]                                # flat-gather rank[t_j, g]
            sel = inb & (r < K)
            inv[sel, j] = tj[sel] * K + r[sel]
            j += 1
    return inv


def check(res, G, K, R=1, seed=0):
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    Rv, tv = torch.eye(3), torch.tensor([0.0, 0.0, 4.0])
    fx = fy = res * 1.2
    cov = geometry.cov3d(torch.exp(m.log_scales), geometry.quat_to_rotmat(m.quats))
    mu2d, conic, depth, keep = geometry.project_ewa(m.means3d, cov, Rv, tv, fx, fy, res / 2, res / 2, 0.3, 0.2)
    mu2d, keep = mu2d.detach(), keep.detach()
    tmap = TileMap(res, res)
    idx, valid = assign_bins(mu2d, keep, tmap, R, K)
    gx = (mu2d[:, 0] / 16).floor().long().clamp(0, tmap.ntx - 1)
    gy = (mu2d[:, 1] / 16).floor().long().clamp(0, tmap.nty - 1)
    SMAX = 9
    inv_o = build_inv(idx, valid, G, SMAX)                 # host oracle
    inv_d = build_inv_device_ref(idx, valid, gx, gy, tmap, G, K)
    TK = tmap.T * K
    # per-gaussian SET of non-sentinel slots (scatter_dev SUMS them -> order-free)
    mism = 0
    for gi in range(G):
        so = set(x for x in inv_o[gi].tolist() if x != TK)
        sd = set(x for x in inv_d[gi].tolist() if x != TK)
        if so != sd:
            mism += 1
    # also check Smax not exceeded by device build
    maxslots = int((inv_d != TK).sum(1).max())
    print(f"res={res} G={G} K={K}: per-gaussian SET match {G - mism}/{G} "
          f"-> {'PASS' if mism == 0 else f'{mism} MISMATCH'}   (device max slots/gaussian {maxslots})")
    return mism == 0


def main():
    ok = True
    for res, G, K in [(800, 2000, 128), (128, 2000, 128), (128, 1000, 128), (64, 2000, 128)]:
        ok &= check(res, G, K)
    print("ALL PASS" if ok else "SOME MISMATCH")


if __name__ == "__main__":
    main()
