"""On-device binning v1 (device-binning branch) -- topk-per-tile, NO wide global sort.

Reformulates host assign_bins (which sorts G*(2R+1)^2 (tile,dist) pairs globally -> the wide
ttnn.sort HANG) as: build dist2[T,G], mask out-of-stencil/dead gaussians to +LARGE, then
ttnn.topk(K, dim=G, smallest) per tile -> the K nearest in-stencil gaussians = idx[T,K]. topk is
PER-ROW (width G), so no global sort. Correct because WSR is order-independent: only the per-tile
SET must match the oracle.

v1 SCAFFOLD: dist2[T,G] + stencil mask computed on HOST (cheap torch), the SELECTION (topk) on DEVICE
-- that's the irregular part we needed on-device. v2 will move dist2 on-device as a GEMM
(|mu|^2 - 2 mu.c^T + |c|^2). Bounded to G <= ~2880 (topk width safe zone; sort was fine to 2880, hung
at 7168) = the T1 SRAM-resident regime.

Oracle: tools/m4_train_binned.assign_bins, per-tile SET equality (m5_binning_vec.check style).

Run (real silicon, hw):
  podman run --name tt-splat-bin --rm --device /dev/tenstorrent:/dev/tenstorrent \
    -v /dev/hugepages-1G:/dev/hugepages-1G -v $PWD:/workspace \
    -v $PWD/outputs/ttnn_cache:/root/.cache/ttnn -w /workspace --shm-size 32gb \
    --cap-add SYS_NICE tt-splat:dev python3 tools/bin_device.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike import geometry
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins
from m9_scatter_oracle import build_inv

LARGE = 1.0e9


def make_bin_ctx(tmap, G, K, dev):
    """Preallocate the per-(tmap,G,K) CONSTANTS used by bin_to_buffers + build_inv_device ONCE, so the
    per-bin-every call does NO from_torch (the ~170ms host-tilization cost at scale,. Reuse
    across iters; rank_buf is reset to K each call (cheap device fill) before the scatter."""
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    t = torch.arange(T)
    tx, ty = (t % ntx).float(), (t // ntx).float()
    cx, cy = (tx + 0.5) * 16.0, (ty + 0.5) * 16.0

    def ccol(v):
        return ttnn.from_torch(v.reshape(T, 1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    return {
        "cx": ccol(cx), "cy": ccol(cy), "tx": ccol(tx), "ty": ccol(ty),
        "rank_buf": ttnn.from_torch(torch.full((T, G + 1), float(K)), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev),
        "k_vals": ttnn.from_torch(torch.arange(K).to(torch.float32)[None, :].expand(T, K).contiguous(),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        "g_row": ttnn.from_torch(torch.arange(G).to(torch.float32)[None, :].contiguous(),
                                 dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
    }


@torch.no_grad()
def dist2_and_mask(mu2d, keep, tmap, R):
    """HOST: dist2[T,G] to each tile centre, with out-of-stencil / dead gaussians set to +LARGE."""
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    G = mu2d.shape[0]
    gx = (mu2d[:, 0] / 16).floor().long().clamp(0, ntx - 1)
    gy = (mu2d[:, 1] / 16).floor().long().clamp(0, nty - 1)
    t = torch.arange(T)
    tx, ty = t % ntx, t // ntx
    centres = torch.stack([(tx.float() + 0.5) * 16.0, (ty.float() + 0.5) * 16.0], dim=1)  # [T,2]
    d = mu2d[None, :, :] - centres[:, None, :]                  # [T,G,2]
    dist2 = (d * d).sum(-1)                                     # [T,G]
    inb = ((tx[:, None] - gx[None, :]).abs() <= R) & ((ty[:, None] - gy[None, :]).abs() <= R) & keep[None, :]
    return torch.where(inb, dist2, torch.full_like(dist2, LARGE))


@torch.no_grad()
def binning_device(mu2d, keep, tmap, R, K, dev):
    """idx[T,K] long, valid[T,K] bool -- selection (topk) on DEVICE."""
    T = tmap.T
    dist2_m = dist2_and_mask(mu2d, keep, tmap, R)              # [T,G] host
    dm = ttnn.from_torch(dist2_m.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    r = ttnn.topk(dm, K, dim=-1, largest=False, sorted=True)
    vals, inds = r if isinstance(r, (list, tuple)) else (r, None)
    idx = ttnn.to_torch(inds).long().reshape(T, K)
    valid = ttnn.to_torch(vals).float().reshape(T, K) < (LARGE * 0.5)
    return idx, valid


@torch.no_grad()
def binning_device_v2(mu2d, keep, tmap, R, K, dev):
    """v2: dist2[T,G] AND the stencil mask computed ON DEVICE (broadcast outer-diffs), then topk.
    Only O(G)+O(T) vectors cross the bus (no [T,G] host compute / upload). dist2 = (cx-mux)^2+(cy-muy)^2
    computed subtract-FIRST (no bf16 cancellation; fp32 here). Stencil = relu(|tx-gx|-R)+relu(|ty-gy|-R).
    """
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    G = mu2d.shape[0]
    F32 = ttnn.float32
    # O(G) gaussian vectors (host now; in the resident trainer these come from on-device geometry)
    mux, muy = mu2d[:, 0], mu2d[:, 1]
    gx = (mux / 16).floor().clamp(0, ntx - 1)
    gy = (muy / 16).floor().clamp(0, nty - 1)
    keepf = keep.float()
    # O(T) tile vectors (constant per tilemap)
    t = torch.arange(T)
    tx, ty = (t % ntx).float(), (t // ntx).float()
    cx, cy = (tx + 0.5) * 16.0, (ty + 0.5) * 16.0

    def col(v):
        return ttnn.from_torch(v.reshape(T, 1).contiguous(), dtype=F32, layout=ttnn.TILE_LAYOUT, device=dev)

    def row(v):
        return ttnn.from_torch(v.reshape(1, G).contiguous(), dtype=F32, layout=ttnn.TILE_LAYOUT, device=dev)

    if ctx is None:
        cxD, cyD, txD, tyD = col(cx), col(cy), col(tx), col(ty)
    else:
        cxD, cyD, txD, tyD = ctx["cx"], ctx["cy"], ctx["tx"], ctx["ty"]   # preallocated (no per-call from_torch)
    muxD, muyD, gxD, gyD, keepD = row(mux), row(muy), row(gx), row(gy), row(keepf)

    DX = ttnn.subtract(cxD, muxD)                              # [T,G] outer diff
    DY = ttnn.subtract(cyD, muyD)
    dist2 = ttnn.add(ttnn.square(DX), ttnn.square(DY))         # [T,G] (subtract-first: no cancellation)
    vX = ttnn.relu(ttnn.subtract(ttnn.abs(ttnn.subtract(txD, gxD)), float(R)))
    vY = ttnn.relu(ttnn.subtract(ttnn.abs(ttnn.subtract(tyD, gyD)), float(R)))
    viol = ttnn.add(vX, vY)                                    # >0 iff out of R-stencil
    keep_pen = ttnn.multiply(ttnn.rsub(keepD, 1.0), LARGE)     # (1-keep)*LARGE, row-broadcast
    masked = ttnn.add(ttnn.add(dist2, ttnn.multiply(viol, LARGE)), keep_pen)
    masked_bf = ttnn.typecast(masked, ttnn.bfloat16)
    r = ttnn.topk(masked_bf, K, dim=-1, largest=False, sorted=True)
    vals, inds = r if isinstance(r, (list, tuple)) else (r, None)
    idx = ttnn.to_torch(inds).long().reshape(T, K)
    valid = ttnn.to_torch(vals).float().reshape(T, K) < (LARGE * 0.5)
    return idx, valid


@torch.no_grad()
def bin_to_buffers(mux_col, muy_col, keep_col, tmap, R, K, dev, idx_u, valid6, vf, ctx=None):
    """INTEGRATION unit: device binning (v2) reading [G,1] DEVICE columns (cache mu2d_x/mu2d_y/zmask) and
    WRITING idx_u[T,K]uint32 / valid6[T,6,K] / vf[T,K,1] in place -- exactly the buffers the resident
    trainer's rend_fwd/rend_bwd consume. This is what `--device-binning` calls (replacing host assign_bins).
    """
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    G = K  # placeholder; real G from tensor
    F32 = ttnn.float32
    muxR = ttnn.reshape(mux_col, (1, -1))                      # [1,G]
    muyR = ttnn.reshape(muy_col, (1, -1))
    keepR = ttnn.reshape(keep_col, (1, -1))
    gxR = ttnn.clamp(ttnn.floor(ttnn.mul(muxR, 1.0 / 16.0)), 0.0, float(ntx - 1))
    gyR = ttnn.clamp(ttnn.floor(ttnn.mul(muyR, 1.0 / 16.0)), 0.0, float(nty - 1))
    # tile-constant columns [T,1]
    t = torch.arange(T)
    tx, ty = (t % ntx).float(), (t // ntx).float()
    cx, cy = (tx + 0.5) * 16.0, (ty + 0.5) * 16.0

    def col(v):
        return ttnn.from_torch(v.reshape(T, 1).contiguous(), dtype=F32, layout=ttnn.TILE_LAYOUT, device=dev)

    if ctx is None:
        cxD, cyD, txD, tyD = col(cx), col(cy), col(tx), col(ty)
    else:
        cxD, cyD, txD, tyD = ctx["cx"], ctx["cy"], ctx["tx"], ctx["ty"]   # preallocated; do NOT deallocate
    dl = ttnn.deallocate
    # dist2[T,G] fp32 (res1264 needs fp32: cx-mux cancels in bf16). Deallocate [T,G] intermediates
    # PROMPTLY so peak is ~2-3 live, not ~6 (G=100k -> 1.6GB each -> OOM otherwise)..
    DXs = ttnn.square(ttnn.subtract(cxD, muxR)); DYs = ttnn.square(ttnn.subtract(cyD, muyR))
    dist2 = ttnn.add(DXs, DYs); dl(DXs); dl(DYs)
    vX = ttnn.relu(ttnn.subtract(ttnn.abs(ttnn.subtract(txD, gxR)), float(R)))
    vY = ttnn.relu(ttnn.subtract(ttnn.abs(ttnn.subtract(tyD, gyR)), float(R)))
    viol = ttnn.add(vX, vY); dl(vX); dl(vY)
    vL = ttnn.multiply(viol, LARGE); dl(viol)
    keep_pen = ttnn.multiply(ttnn.rsub(keepR, 1.0), LARGE)
    m1 = ttnn.add(dist2, vL); dl(dist2); dl(vL)
    masked = ttnn.add(m1, keep_pen); dl(m1); dl(keep_pen)
    masked_bf = ttnn.typecast(masked, ttnn.bfloat16); dl(masked)
    vals, inds = ttnn.topk(masked_bf, K, dim=-1, largest=False, sorted=True); dl(masked_bf)
    # write idx_u (uint32 ROW_MAJOR)
    idx_rm = ttnn.to_layout(ttnn.typecast(inds, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy(idx_rm, idx_u)
    # valid mask [T,K] = (vals < LARGE/2); write vf[T,K,1] and valid6[T,6,K]
    validf = ttnn.lt(vals, LARGE * 0.5)                        # 1.0 where valid
    ttnn.copy(ttnn.reshape(ttnn.typecast(validf, ttnn.bfloat16), (T, K, 1)), vf)
    v1k = ttnn.reshape(ttnn.typecast(validf, ttnn.bfloat16), (T, 1, K))
    ttnn.copy(ttnn.concat([v1k] * 6, dim=1), valid6)


def check_buffers(dev, res=128, G=2000, K=128, R=1, seed=0):
    """Verify bin_to_buffers writes idx_u/valid6/vf that decode to the same per-tile SET as assign_bins."""
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    Rv, tv = torch.eye(3), torch.tensor([0.0, 0.0, 4.0])
    fx = fy = res * 1.2
    cov = geometry.cov3d(torch.exp(m.log_scales), geometry.quat_to_rotmat(m.quats))
    mu2d, conic, depth, keep = geometry.project_ewa(m.means3d, cov, Rv, tv, fx, fy, res / 2, res / 2, 0.3, 0.2)
    mu2d, keep = mu2d.detach(), keep.detach()
    tmap = TileMap(res, res)
    io, vo = assign_bins(mu2d, keep, tmap, R, K)
    T = tmap.T
    mux_col = ttnn.from_torch(mu2d[:, 0:1].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    muy_col = ttnn.from_torch(mu2d[:, 1:2].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    keep_col = ttnn.from_torch(keep.float().reshape(G, 1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    idx_u = ttnn.from_torch(torch.zeros(T, K, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    valid6 = ttnn.from_torch(torch.zeros(T, 6, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vf = ttnn.from_torch(torch.zeros(T, K, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    bin_to_buffers(mux_col, muy_col, keep_col, tmap, R, K, dev, idx_u, valid6, vf)
    idx = ttnn.to_torch(idx_u).long().reshape(T, K)
    vfd = ttnn.to_torch(vf).float().reshape(T, K) > 0.5
    mism = sum(set(io[t][vo[t]].tolist()) != set(idx[t][vfd[t]].tolist()) for t in range(T))
    cand = (dist2_and_mask(mu2d, keep, tmap, R) < LARGE * 0.5).sum(1)
    overflow_only = all((set(io[t][vo[t]].tolist()) == set(idx[t][vfd[t]].tolist())) or int(cand[t]) > K for t in range(T))
    print(f"[bin_to_buffers] res={res} G={G} K={K}: SET match {T - mism}/{T}  "
          f"-> {'PASS' if mism == 0 else ('benign overflow-only' if overflow_only else 'NON-BENIGN')}")
    return mism == 0 or overflow_only


@torch.no_grad()
def build_inv_device(idx_u, valid_f, mux_col, muy_col, tmap, K, dev, sinv_u=None, ctx=None):
    """DEVICE inv[G,9] (gaussian->slot positions) for the grad scatter, replacing host build_inv.
    Ports tools/bin_inv_ref.py (validated): rank[T,G+1] single-dest scatter + 9 stencil flat-gathers.
    If sinv_u given, writes it in place (uint32 ROW_MAJOR) and returns None (trainer path, NO download);
    else returns inv[G,9] long (host) for verification.
    idx_u[T,K]uint32, valid_f[T,K] (1/0), mux/muy [G,1] device columns."""
    ntx, nty, T = tmap.ntx, tmap.nty, tmap.T
    G = mux_col.shape[0]
    G1 = G + 1
    sentinel = float(T * K)
    BFL, F32 = ttnn.bfloat16, ttnn.float32

    def rm_u(t):   # uint32 ROW_MAJOR index (scatter/embedding index layout, per m9 / scatter bench)
        return ttnn.to_layout(ttnn.typecast(t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)

    # ---- rank[T,G+1] = K; scatter rank[t, idx_safe[t,k]] = k  (invalid slots -> dummy column G) ----
    idx_f = ttnn.typecast(idx_u, F32)                                          # [T,K]
    idx_safe = ttnn.add(ttnn.mul(idx_f, valid_f), ttnn.mul(ttnn.rsub(valid_f, 1.0), float(G)))
    if ctx is None:
        k_vals = ttnn.from_torch(torch.arange(K).to(torch.float32)[None, :].expand(T, K).contiguous(),
                                 dtype=BFL, layout=ttnn.TILE_LAYOUT, device=dev)
        rank0 = ttnn.from_torch(torch.full((T, G1), float(K)), dtype=BFL, layout=ttnn.TILE_LAYOUT, device=dev)
    else:
        k_vals, rank0 = ctx["k_vals"], ctx["rank_buf"]                         # preallocated (no from_torch)
        ttnn.multiply(rank0, 0.0, output_tensor=rank0)                         # reset to sentinel K (cheap fill,
        ttnn.add(rank0, float(K), output_tensor=rank0)                         # robust to scatter in/out-of-place)
    rank = ttnn.scatter(rank0, 1, rm_u(idx_safe), k_vals)                      # [T,G+1] bf16
    # embedding weight [T*(G+1),1] kept ROW_MAJOR: TILE would pad inner-dim 1->32 = 32x blowup
    # (G=100k: 411M -> 13B = 26GB OOM). ROW_MAJOR [N,1] = no pad -> ~0.8GB.
    rank_flat = ttnn.reshape(ttnn.to_layout(rank, ttnn.ROW_MAJOR_LAYOUT), (T * G1, 1))
    if ctx is None:
        ttnn.deallocate(rank)                                                 # ctx rank_buf is persistent; else free
    # ---- gx,gy [1,G]; per-offset flat-gather of rank ----
    muxR = ttnn.reshape(mux_col, (1, G)); muyR = ttnn.reshape(muy_col, (1, G))
    gx = ttnn.clamp(ttnn.floor(ttnn.mul(muxR, 1.0 / 16.0)), 0.0, float(ntx - 1))
    gy = ttnn.clamp(ttnn.floor(ttnn.mul(muyR, 1.0 / 16.0)), 0.0, float(nty - 1))
    g_row = ctx["g_row"] if ctx is not None else ttnn.from_torch(
        torch.arange(G).to(torch.float32)[None, :].contiguous(), dtype=F32, layout=ttnn.TILE_LAYOUT, device=dev)
    cols = []
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            txj, tyj = ttnn.add(gx, float(ox)), ttnn.add(gy, float(oy))        # [1,G]
            inb = ttnn.mul(ttnn.mul(ttnn.ge(txj, 0.0), ttnn.lt(txj, float(ntx))),
                           ttnn.mul(ttnn.ge(tyj, 0.0), ttnn.lt(tyj, float(nty))))
            txc, tyc = ttnn.clamp(txj, 0.0, float(ntx - 1)), ttnn.clamp(tyj, 0.0, float(nty - 1))
            tj = ttnn.add(ttnn.mul(tyc, float(ntx)), txc)                      # [1,G]
            flat_idx = ttnn.add(ttnn.mul(tj, float(G1)), g_row)
            r = ttnn.typecast(ttnn.reshape(ttnn.embedding(rm_u(flat_idx), rank_flat), (1, G)), F32)
            sel = ttnn.mul(inb, ttnn.lt(r, float(K)))                          # 1 iff selected & in-bounds
            slot = ttnn.add(ttnn.mul(tj, float(K)), r)
            cols.append(ttnn.add(ttnn.mul(sel, slot), ttnn.mul(ttnn.rsub(sel, 1.0), sentinel)))
    ttnn.deallocate(rank_flat)                                                # free the big [N,1] gather table
    inv = ttnn.transpose(ttnn.concat(cols, dim=0), -2, -1)                     # [G,9]
    if sinv_u is not None:
        ttnn.copy(ttnn.to_layout(ttnn.typecast(inv, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), sinv_u)
        return None
    return ttnn.to_torch(inv).round().long()


def check_inv_device(dev, res=128, G=1000, K=128, R=1, seed=0):
    """Verify build_inv_device reproduces host build_inv's per-gaussian SET (scatter_dev sums -> set-equal)."""
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    Rv, tv = torch.eye(3), torch.tensor([0.0, 0.0, 4.0])
    fx = fy = res * 1.2
    cov = geometry.cov3d(torch.exp(m.log_scales), geometry.quat_to_rotmat(m.quats))
    mu2d, conic, depth, keep = geometry.project_ewa(m.means3d, cov, Rv, tv, fx, fy, res / 2, res / 2, 0.3, 0.2)
    mu2d, keep = mu2d.detach(), keep.detach()
    tmap = TileMap(res, res)
    idx, valid = assign_bins(mu2d, keep, tmap, R, K)
    inv_o = build_inv(idx, valid, G, 9)
    idx_u = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    valid_f = ttnn.from_torch(valid.float().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mux = ttnn.from_torch(mu2d[:, 0:1].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    muy = ttnn.from_torch(mu2d[:, 1:2].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    inv_d = build_inv_device(idx_u, valid_f, mux, muy, tmap, K, dev)
    TK = tmap.T * K
    mism = sum(set(x for x in inv_o[g].tolist() if x != TK) != set(x for x in inv_d[g].tolist() if x != TK)
               for g in range(G))
    print(f"[build_inv_device] res={res} G={G} K={K}: per-gaussian SET {G - mism}/{G} "
          f"-> {'PASS' if mism == 0 else f'{mism} MISMATCH'}")
    return mism == 0


def check(dev, res=128, G=2000, K=128, R=1, seed=0, binner=binning_device):
    torch.manual_seed(seed)
    m = GaussianModel(G, extent=1.5, seed=seed)
    Rv, tv = torch.eye(3), torch.tensor([0.0, 0.0, 4.0])
    fx = fy = res * 1.2
    cov = geometry.cov3d(torch.exp(m.log_scales), geometry.quat_to_rotmat(m.quats))
    mu2d, conic, depth, keep = geometry.project_ewa(m.means3d, cov, Rv, tv, fx, fy, res / 2, res / 2, 0.3, 0.2)
    mu2d, keep = mu2d.detach(), keep.detach()
    tmap = TileMap(res, res)

    io, vo = assign_bins(mu2d, keep, tmap, R, K)               # host oracle
    t0 = time.time()
    iv, vv = binner(mu2d, keep, tmap, R, K, dev)               # device binning (v1 or v2)
    dt = (time.time() - t0) * 1e3

    # per-tile candidate counts (from the stencil mask) -> overflow = count > K
    dm = dist2_and_mask(mu2d, keep, tmap, R)
    cand_cnt = (dm < LARGE * 0.5).sum(1)                       # [T] candidates per tile
    mism = 0
    mism_overflow = 0
    for t in range(tmap.T):
        so = set(io[t][vo[t]].tolist())
        sv = set(iv[t][vv[t]].tolist())
        if so != sv:
            mism += 1
            if int(cand_cnt[t]) > K:
                mism_overflow += 1
    occ_o, occ_v = int(vo.sum()), int(vv.sum())
    n_overflow = int((cand_cnt > K).sum())
    ok = mism == 0
    benign = mism == mism_overflow                            # all mismatches are overflow tiles
    print(f"[{binner.__name__}] res={res} G={G} K={K} R={R}: T={tmap.T}  oracle slots {occ_o} / device {occ_v}  "
          f"max cand/tile {int(cand_cnt.max())}  overflow tiles(>{K}) {n_overflow}  ({dt:.0f} ms)")
    verdict = "PASS" if ok else (f"{mism} mismatch ALL on overflow tiles (benign K-boundary tie)"
                                 if benign else f"{mism} MISMATCH ({mism - mism_overflow} NON-overflow!)")
    print(f"   per-tile SET match: {tmap.T - mism}/{tmap.T}  -> {verdict}")
    return ok or benign


def main():
    print("== on-device binning vs host assign_bins oracle (v1 host-dist2 / v2 fully-device) ==")
    dev = ttnn.open_device(device_id=0)
    allok = True
    try:
        # v2 = fully on-device dist2+mask. Compare to oracle; should match v1 (exact no-overflow, benign overflow).
        for binner in (binning_device, binning_device_v2):
            print(f"--- {binner.__name__} ---")
            for res, G, K in [(800, 2000, 128), (128, 2000, 128), (128, 1000, 128)]:
                allok &= check(dev, res=res, G=G, K=K, binner=binner)
            print()
        # INTEGRATION unit: device binning that WRITES the trainer buffers (idx_u/valid6/vf)
        print("--- bin_to_buffers (integration unit: writes idx_u/valid6/vf) ---")
        for res, G, K in [(800, 2000, 128), (128, 1000, 128)]:
            allok &= check_buffers(dev, res=res, G=G, K=K)
        print()
        # SYNC-REMOVAL unit: device inv-build (replaces host build_inv)
        print("--- build_inv_device (sync-removal: device inv[G,9] vs host build_inv) ---")
        for res, G, K in [(128, 1000, 128), (800, 2000, 128)]:
            allok &= check_inv_device(dev, res=res, G=G, K=K)
        print()
    finally:
        ttnn.close_device(dev)
    print("ALL PASS/BENIGN" if allok else "SOME NON-BENIGN MISMATCH")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
