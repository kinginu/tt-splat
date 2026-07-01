# Hand-off — binned torch PW render (device-faithful, GPU-fast) for iters/res quality sweeps

> **One Sonnet agent, runs the sweeps on the GPU box.** Goal: a pure-torch, differentiable render that
> reproduces the Blackhole device's **K=128 per-tile occluder constraint** so we can answer "does
> increasing iters/res improve arm PW's quality?" in **minutes on a GPU** instead of hours on the
> device. Branch: `feat/pw-binned-torch` (off `feat/arm-pairwise-device`, which has host `blend_PW` +
> the device `render_fwd_pw` reference).

## Why this exists

- The current torch render (`spike/render.py`) is **dense `[P,G]`** → at res800/G10000 it OOMs, and the
  PW occlusion is **O(P·G²)** (`logT = a @ Sᵀ`, `S=[G,G]`) → hours even on a GPU at G=10000.
- The device avoids this with **per-tile K=128 binning** → occlusion is **O(K²)** per tile, i.e.
  `(G/K)² = (10000/128)² ≈ 6000×` cheaper. Reproducing that in torch makes GPU loops fast **and**
  keeps the result faithful to the device (same K=128 restriction), so conclusions transfer.
- This is the fast trial-and-error loop for the quality/scaling question (the current stage's real
  question), replacing 4-hour device runs.

## Building blocks (already exist — reuse, don't reinvent)

- `tools/m4_train_binned.py::assign_bins(mu2d, keep, tmap, R=1, K)` → `idx[T,K]` long, `valid[T,K]` bool
  (host torch, the exact device binning).
- `tools/m4_train_binned.py::TileMap(H, W)` → `T` tiles (16×16), tile origins / grid layout.
- `spike/forward.py::quad_form_tilelocal(lx, ly, conic_abc, mu2d, tile_origin)` → tile-local `Q` as a
  GEMM (numerically stable in the tile frame — use this, not the global `quad_form`).
- `spike/forward.py::poly_splat_wgeo(Q, k, keep)`; `spike/arms.py::blend_PW` (the PW math oracle);
  `spike/arms.py::_depth_warp`.
- `spike/geometry.py::project_ewa` (→ mu2d, conic, depth, keep), the same geometry `render()` uses.

## The binned render to implement

Add `render_binned(model, cam, K=128, tau=0.01, k=4.0, blur_eps=0.3, near=0.2) -> [H,W,3]`:

```
# 1. geometry (same as spike/render.py::render)
mu2d, conic, depth, keep = project_ewa(...)            # [G,2],[G,3],[G],[G]
o = sigmoid(opacity_raw); color = color_from_dc(color_dc); zw = _depth_warp(depth)   # [G],[G,3],[G]

# 2. bin (DETACHED — index assignment is non-diff, like the device; gathered VALUES stay attached)
tmap = TileMap(cam.H, cam.W); T = tmap.T
idx, valid = assign_bins(mu2d.detach(), keep.detach() > 0.5, tmap, 1, K)   # [T,K] long, [T,K] bool

# 3. per-tile gather (attached params -> gradients flow)
conic_t = conic[idx]           # [T,K,3]
mu_t    = mu2d[idx]            # [T,K,2]  (global mu; quad_form_tilelocal folds the tile origin)
o_t     = o[idx]              # [T,K]
col_t   = color[idx]         # [T,K,3]
zw_t    = zw[idx]            # [T,K]
vmask   = valid.float()      # [T,K]

# 4. tile-local poly-splat  (lx,ly in 0..15 for the 256 pixels of a tile; origins from tmap)
Q_t     = quad_form_tilelocal(lx, ly, conic_t, mu_t, tile_origin_t)   # [T,256,K]  (batch over T)
w_geo   = poly_splat_wgeo(Q_t, k) * vmask[:, None, :]                 # [T,256,K]  (mask padded slots)

# 5. PW occlusion PER TILE (the whole point — O(K²), not O(G²))
alpha   = (o_t[:, None, :] * w_geo).clamp(1e-6, 1-1e-4)               # [T,256,K]
a       = -log1p(-alpha)                                             # [T,256,K]
S       = sigmoid((zw_t[:, :, None] - zw_t[:, None, :]) / tau)       # [T,K,K]  S[t,i,j]=σ((z_i-z_j)/τ)
S       = S * (1 - eye(K)) * vmask[:, None, :]                       # exclusive + mask padded occluders
logT    = -matmul(a, S.transpose(-1,-2))                             # [T,256,K]
Tr      = exp(clamp(logT, min=-30))                                 # [T,256,K]
W       = alpha * Tr                                                 # [T,256,K]
num     = matmul(W, col_t)                                          # [T,256,3]
T_bg    = exp(-a.sum(-1, keepdim=True).clamp(min=0))                 # [T,256,1]
C_tiles = num + T_bg * c_b                                          # [T,256,3]  OIT-over

# 6. reassemble tiles -> image [H,W,3] (use tmap's tile row/col grid; inverse of the 16x16 tiling)
return scatter_tiles_to_image(C_tiles, tmap, cam.H, cam.W)
```

Notes:
- **Batch step 4 over all T tiles** (don't Python-loop tiles) — `quad_form_tilelocal` already returns a
  GEMM; give it the per-tile `lx,ly` (shared 0..15 grid) and per-tile `tile_origin`. Result `[T,256,K]`.
- `matmul(a[T,256,K], S.transpose[T,K,K])` and `matmul(W[T,256,K], col_t[T,K,3])` are batched over T —
  this is the O(T·256·K²) work (fits a GPU easily even at res800: T=2500, 256, K=128).
- Keep `zw`/`depth` **attached** through `S` (C3 z-force). Only `idx` (binning) is detached.
- Mirror `render_fwd_pw` in `tools/sweep_resident.py` for the exact composite (it is the device oracle).

## Integration

- New module `spike/render_binned.py` (or add to `spike/render.py`) with `render_binned(...)`.
- `spike/m05_spike.py`: add `--binned K` (default off). When set, occlusion arms (`PW`, and optionally
  `MO`/`BP`) render via `render_binned` instead of the dense path. `train.fit`/`eval_psnr` call it
  transparently (they already take a render callable / arm).
- Keep the dense path unchanged for small-scale/oracle use.

## Validation (do before the sweeps)

1. **binned == dense at G ≤ K**: with `G=100, K=128`, every tile's occluders = all gaussians →
   `render_binned` must match dense `blend_PW` (global) to tight rtol on a toy scene. (Proves the tiled
   math is correct.)
2. **binned ≈ device**: on a small scene (res128/G2000), `render_binned(K=128)` should match the device
   `render_fwd_pw` output (the device is busy — do this opportunistically, or compare to the device's
   known quality gate 18.80). At minimum, the toy occlusion probe (front red dominates, z-grad nonzero).
3. `python -m py_compile` + a 50-iter GPU smoke (loss ↓).

## The actual experiment (run on the GPU box — the payoff)

Sweep arm PW (binned) vs A and D, over iters × res, on lego + ficus:
```bash
for RES in 128 256 512 800; do for IT in 3000 10000 30000; do
  python -m spike.m05_spike --scene data/nerf_synthetic/lego --arms A,PW,D \
     --binned 128 --G 10000 --res $RES --iters $IT --seeds 1 --device cuda \
     --out outputs/m05_pwbin_lego_res${RES}_it${IT}
done; done
```
(res800/G10000/iter30000 should be **minutes–tens of minutes** on a GPU with the binned render, vs 4 h
on the device.) **Read-out:** does PW holdout PSNR climb with iters and with res, and how far below the
D ceiling does it plateau? That answers the current-stage question and tells us whether the device is
worth the long runs.

## Scope defaults (adjust if desired)
- **Occluders = K=128 per tile** (device-faithful). Optionally also sweep K∈{128,256} to test the
  restriction's effect at scale (the "does K suffice" question).
- **Full-batch views** (m05 default n_train=8) for a clean quality signal; a stochastic 1-view option
  matches the device trainer if wall-clock matters.
- `tau=0.01` (the value that won the bake-off).

## Report
Branch + sha; the `render_binned` implementation + how tiles are reassembled; validation results
(binned==dense at G≤K; device/toy match); the iters×res×scene PSNR table (vs A and D); and the
wall-clock per config (to confirm the GPU loop is minutes, not hours).
