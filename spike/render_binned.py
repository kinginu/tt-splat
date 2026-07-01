"""Pure-torch "binned" render for arm PW: reproduces the Blackhole device's K=128 per-tile
occluder constraint so PW quality-vs-(iters,res) sweeps run in minutes on a GPU instead of
hours on hardware.

Dense `render(model, cam, "PW")` (spike/render.py + spike/arms.py::blend_PW) is O(P*G^2): the
pairwise soft-occlusion compare S=[G,G] is global over all gaussians, for every pixel. The
device instead assigns each 16x16 tile <=K local occluders (R=1 tile-neighbourhood binning) and
only compares within that tile -> O(T*256*K^2). At G=10000,K=128 that is ~(G/K)^2 ~= 6100x
cheaper, letting the same PW math run at res800/iters30000 in minutes.

`TileMap` and `assign_bins` below are ported verbatim (pure-torch logic only) from
`tools/m4_train_binned.py`, which cannot be imported directly here: it does `import ttnn` at
module scope, and ttnn is not installed in this (host/GPU) environment. The rest
(`quad_form_tilelocal`, `poly_splat_wgeo`, `_depth_warp`, `project_ewa`, `color_from_dc`) is
imported and reused as-is; only the per-tile gather + PW compositing is new, and it mirrors
`arms.blend_PW` exactly (same alpha/absorbance/S/T_bg formulas), just batched per tile instead
of globally.
"""
import torch

from . import arms, forward, geometry

_TILEMAP_CACHE = {}


class TileMap:
    """16x16-tile layout for an HxW image (H,W must be multiples of 16).

    - `lx`, `ly` [256]: tile-local pixel centers (0.5..15.5), shared by every tile.
    - `origins` [T,2]: (x,y) pixel-space origin of each tile's top-left corner.
    - `gidx` [T*256]: flattened (tile, tile-pixel) -> flat image pixel index (row-major H*W);
      a permutation of range(H*W) since the 16x16 tiles exactly partition the image.
    """

    def __init__(self, H, W, device=None):
        assert H % 16 == 0 and W % 16 == 0, f"TileMap requires H,W multiples of 16 (got {H},{W})"
        device = device or "cpu"
        self.H, self.W, self.nty, self.ntx = H, W, H // 16, W // 16
        self.T = self.nty * self.ntx
        p = torch.arange(256, device=device)
        self.lx = (p % 16).float() + 0.5
        self.ly = (p // 16).float() + 0.5
        t = torch.arange(self.T, device=device)
        ty, tx = t // self.ntx, t % self.ntx
        self.origins = torch.stack([(tx * 16).float(), (ty * 16).float()], dim=1)  # [T,2]
        gr = ty[:, None] * 16 + (p // 16)[None, :]
        gc = tx[:, None] * 16 + (p % 16)[None, :]
        self.gidx = (gr * W + gc).reshape(-1).long()


@torch.no_grad()
def assign_bins(mu2d, keep, tmap, R, K):
    """Vectorized per-tile gaussian index lists. Returns idx[T,K] long, valid[T,K] bool.

    Ported verbatim (device-parameterized) from tools/m4_train_binned.py::assign_bins.
    Fully vectorized: O(G*(2R+1)^2) pair construction + one combined argsort (by tile then
    dist-to-centre) + position-within-tile scatter. No Python loop over T tiles.
    Overflow (>K per tile) is handled by the distance-first sort: the first K elements in each
    tile group are the K nearest to tile centre -- same policy as the original topk fallback.
    """
    device = mu2d.device
    ntx, nty = tmap.ntx, tmap.nty
    gx = (mu2d[:, 0] / 16).floor().long().clamp(0, ntx - 1)
    gy = (mu2d[:, 1] / 16).floor().long().clamp(0, nty - 1)
    G_total = mu2d.shape[0]
    g_all = torch.arange(G_total, device=device)

    # Build (tile_id, g_id) pairs -- O(G*(2R+1)^2)
    tiles_l, gs_l = [], []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            tx_n = gx + dx
            ty_n = gy + dy
            ok = keep & (tx_n >= 0) & (tx_n < ntx) & (ty_n >= 0) & (ty_n < nty)
            tiles_l.append((ty_n * ntx + tx_n)[ok])
            gs_l.append(g_all[ok])

    idx = torch.zeros(tmap.T, K, dtype=torch.long, device=device)
    valid = torch.zeros(tmap.T, K, dtype=torch.bool, device=device)
    if not tiles_l or not any(t.numel() > 0 for t in tiles_l):
        return idx, valid

    all_tiles = torch.cat(tiles_l)
    all_gs = torch.cat(gs_l)

    # Sort by (tile_id, dist_to_tile_centre) so first K per tile group = K nearest.
    # Pack into int64: tile * 2048 + quantized_dist2.  At R=1 max dist2 ~= 1200 < 2048.
    tx_f = (all_tiles % ntx).float()
    ty_f = (all_tiles // ntx).float()
    dist2 = ((mu2d[all_gs, 0] - (tx_f + 0.5) * 16.0) ** 2 +
             (mu2d[all_gs, 1] - (ty_f + 0.5) * 16.0) ** 2)
    sort_key = all_tiles.long() * 2048 + dist2.long().clamp(0, 2047)
    order = torch.argsort(sort_key)
    all_tiles = all_tiles[order]
    all_gs = all_gs[order]

    # Position within tile group: first K are kept (= K nearest by construction)
    t_bnd = torch.searchsorted(all_tiles.contiguous(), torch.arange(tmap.T + 1, device=device))
    run_idx = torch.arange(len(all_tiles), dtype=torch.long, device=device)
    pos_within = run_idx - t_bnd[all_tiles]
    mask = pos_within < K
    flat_out = all_tiles[mask] * K + pos_within[mask]
    idx.view(-1)[flat_out] = all_gs[mask]
    valid.view(-1)[flat_out] = True
    return idx, valid


def _get_tilemap(H, W, device):
    key = (H, W, str(device))
    tmap = _TILEMAP_CACHE.get(key)
    if tmap is None:
        tmap = TileMap(H, W, device=device)
        _TILEMAP_CACHE[key] = tmap
    return tmap


def render_binned(model, cam, K=128, tau=0.01, k=4.0, blur_eps=0.3, near=0.2, R=1):
    """Binned torch PW render: `render(model, cam, "PW")`'s math (arms.blend_PW), but with the
    global O(G) occlusion set replaced by each 16x16 tile's <=K nearest gaussians (R=1 tile
    neighbourhood, device-faithful binning). O(T*256*K^2) instead of O(P*G^2).

    Only the bin *assignment* (`idx`) is detached (index selection is non-differentiable, same
    as the device); every gathered VALUE (conic, mu2d, opacity, color, depth/zw) stays attached,
    so gradients (incl. the PW z-force through S) flow to the per-gaussian parameters exactly as
    in the dense arm.
    """
    device = model.means3d.device

    # 1. geometry -- identical to spike/render.py::render
    Rm = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), Rm)
    mu2d, conic, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, blur_eps, near)
    o = torch.sigmoid(model.opacity_raw)                      # [G]
    color = forward.color_from_dc(model.color_dc)             # [G,3]
    zw = arms._depth_warp(depth)                               # [G] in [0,1], ATTACHED (C3)
    c_b = model.c_b

    # 2. bin (DETACHED indices; device does the same -- index assignment is non-diff)
    tmap = _get_tilemap(cam.H, cam.W, device)
    idx, valid = assign_bins(mu2d.detach(), keep.detach(), tmap, R, K)   # [T,K] long, [T,K] bool
    vmask = valid.to(mu2d.dtype)                                        # [T,K]

    # 3. per-tile gather (ATTACHED values -> gradients flow via advanced indexing)
    conic_t = conic[idx]        # [T,K,3]
    mu_t = mu2d[idx]             # [T,K,2]  (global mu; quad_form_tilelocal folds the tile origin)
    o_t = o[idx]                 # [T,K]
    col_t = color[idx]          # [T,K,3]
    zw_t = zw[idx]               # [T,K]

    # 4. tile-local poly-splat, batched over all T tiles (no python loop). `forward.quad_form_tilelocal`
    # itself assumes exactly-2D input (its `theta_from_conic` call does `conic_abc[:, 0]`, a
    # dim-1 index that is only equivalent to the last-dim channel select for 2D tensors) so it
    # can't be called directly on our [T,K,*] batch; flatten to [T*K,*] to reuse
    # `theta_from_conic`/`phi` (the primitives quad_form_tilelocal itself is built from)
    # unmodified, then reshape back and batch the final GEMM over T via broadcasting.
    T_, K_g = conic_t.shape[0], conic_t.shape[1]
    mu_local = (mu_t - tmap.origins[:, None, :]).reshape(T_ * K_g, 2)   # tile origin folded in
    conic_flat = conic_t.reshape(T_ * K_g, 3)
    theta_Q, _ = forward.theta_from_conic(conic_flat, mu_local, k=1.0)  # [T*K,6]
    theta_Q = theta_Q.reshape(T_, K_g, 6)
    Q_t = forward.phi(tmap.lx, tmap.ly) @ theta_Q.transpose(-1, -2)     # [256,6]@[T,6,K] -> [T,256,K]
    w_geo = forward.poly_splat_wgeo(Q_t, k) * vmask[:, None, :]         # [T,256,K]

    # 5. PW occlusion PER TILE -- O(K^2) not O(G^2); mirrors arms.blend_PW exactly.
    alpha = (o_t[:, None, :] * w_geo).clamp(1e-6, 1.0 - 1e-4)   # [T,256,K]
    a = -torch.log1p(-alpha)                                     # [T,256,K] absorbance
    K_ = zw_t.shape[1]
    S = torch.sigmoid((zw_t[:, :, None] - zw_t[:, None, :]) / tau)   # [T,K,K] S[t,i,j]=sig((z_i-z_j)/tau)
    eye = torch.eye(K_, dtype=S.dtype, device=device)
    S = S * (1.0 - eye) * vmask[:, None, :]                       # exclusive diag + mask padded occluders
    logT = -torch.matmul(a, S.transpose(-1, -2))                  # [T,256,K]
    Tr = torch.exp(logT.clamp(min=-30.0))
    W = alpha * Tr                                                # [T,256,K]
    num = torch.matmul(W, col_t)                                  # [T,256,3]
    T_bg = torch.exp(-a.sum(dim=-1, keepdim=True).clamp(min=0.0))  # [T,256,1]
    C_tiles = num + T_bg * c_b[None, None, :]                     # [T,256,3]  OIT-over composite

    # 6. reassemble tiles -> [H,W,3] via tmap.gidx (a permutation of range(H*W): the 16x16 tiling
    # partitions the image exactly once). index_copy keeps this differentiable w.r.t. C_tiles.
    img = torch.zeros(cam.H * cam.W, 3, dtype=C_tiles.dtype, device=device)
    img = img.index_copy(0, tmap.gidx.to(device), C_tiles.reshape(-1, 3))
    return img.reshape(cam.H, cam.W, 3)
