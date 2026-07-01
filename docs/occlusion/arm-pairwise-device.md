# Hand-off — tt-nn device port of arm `PW` (per-tile pairwise soft-occlusion)

> **One Sonnet agent.** The GPU bake-off picked **PW** as the sort-free winner (lego 18.51 vs A 16.18 =
> 65% of the A→D gap, stable ±0.51; ficus 25.29 ≈ D). Port PW to the Blackhole traced path. This is the
> **hardest** port so far: PW is **per-pixel** (rewrites `rend_fwd`/`rend_bwd`, re-captures `rfid`/`rbid`),
> NOT a `keo` fold. Read `01-arm-authoring-contract.md`, `02-device-map.md`, and `arm-softmin.md §Device`
> (its z-force wiring is reused). §3-gated. Branch: `feat/arm-pairwise-device`.

## Base branch / reuse
- Base off the branch that has the host `blend_PW` **and** the validated SM device z-force wiring. If
  the SM device branch (`feat/arm-softmin-device`, SM port: means z-force rel ~8e-8, validated on hw) is
  available, base off it and merge `feat/arm-pairwise` (host PW); otherwise base off `feat/arm-pairwise`
  and **re-implement the SM z-force injection** per `arm-softmin.md §Device` (it is ~4 lines:
  `out["mx"] += M(grad_mcz, Rvb[2][0])`, my, mz). State which you did.
- Reuse: SM's host-Adam scalar pattern (`adam_sm`) for `pw_tau`; SM's `grad_mcz → means` via `Rvb[2][j]`.

## The oracle — host `blend_PW` (spike/arms.py, DO NOT change it)
```python
o = sigmoid(opacity_raw); alpha = (o[None,:]*w_geo).clamp(eps, 1-1e-4)   # [P,G]
a = -log1p(-alpha)                                                        # [P,G] true absorbance
zw = clamp((depth-0.5)/7.5, 0, 1)                                         # _depth_warp(near=0.5,far=8.0), ATTACHED
tau = pw_tau.clamp(min=1e-3)                                              # pw_tau init 0.01, learnable (lr["depth"])
S = sigmoid((zw[:,None]-zw[None,:])/tau) * (1-eye(G))                     # [G,G] S[g,h]="h in front of g", excl.
logT = -(a @ S.T)                                                         # [P,G]
T = exp(logT.clamp(min=-30)); W = alpha*T                                 # [P,G]
num = W @ color; T_bg = exp(-a.sum(1,keepdim).clamp(min=0))               # [P,3],[P,1]
C = num + T_bg * c_b                                                      # OIT-over (NO WSR div)
```
Every device kernel is validated against this. `pw_tau=0.01` is the value that won on GPU — use it.

## Device FORWARD — rewrite `render_fwd_cache` for the PW path (per-tile `[T,256,K]`)

Reuse the existing gather (`gather_theta_cols` → `theta`, `col_t[T,K,3]=keo·color`, `oc_t[T,K,1]=keo=keep·o`)
and poly-splat `w[T,256,K] = square(relu(matmul(Phi,thU)))`. Then REPLACE the WSR `num/den/div` with:
```
alpha_pg = M(w, oc_t)                              # [T,256,K]  = w_geo·keep·o  (broadcast oc over 256)
a_pg     = ttnn.neg(ttnn.log( clamp(1 - alpha_pg, min=1e-6) ))    # [T,256,K]  ≈ -log1p(-alpha)
z_t      = ttnn.embedding(idx_u, mcz)             # [T,K,1] gather camera-z per tile-slot
zw_t     = clamp((z_t - 0.5) * (1/7.5), 0, 1)     # [T,K,1] warped (near=0.5,far=8.0)
S        = ttnn.sigmoid( (zw_t[T,K,1] - zw_t[T,1,K]) * inv_tau )   # [T,K,K] outer-diff then σ; inv_tau=1/τ
S        = M(S, one_minus_I_KK)                   # zero diagonal ([K,K] const buffer)
logT     = ttnn.neg( ttnn.matmul(a_pg, T3(S), core_grid=CG) )     # [T,256,K] = -Σ_h a[·,h]·S[g,h]  (the pairwise GEMM)
Tt       = ttnn.exp( clamp(logT, min=-30) )       # [T,256,K]
num      = ttnn.matmul( M(w, Tt), col_t, core_grid=CG )           # [T,256,3]  Σ_g (w·T)·(keo·color)
a_sum    = ttnn.sum(a_pg, dim=K) ; T_bg = ttnn.exp(ttnn.neg(clamp(a_sum,min=0)))   # [T,256,1]
C        = ttnn.add( num, M(T_bg, cb_bcast) )     # [T,256,3]  OIT-over; cb_bcast = c_b broadcast
```
Cache `{alpha_pg, a_pg, S, zw_t, logT, Tt, w, col_t, T_bg}` for the backward. Note: NO `den`, NO `div`,
NO `wbb` for PW — the background is the `T_bg·c_b` term. **Re-capture `rfid`** with these buffers
pre-allocated.

## Device BACKWARD — rewrite the PW half of `render_bwd`/`geom_bwd`

Given `gC[T,256,3]` from the loss trace, chain (all per-tile):
```
gnum = gC ; gT_bg = ttnn.sum(M(gC, cb_bcast), dim=3)                      # [T,256,1]
# num = matmul(w·T, col_t):
gwT   = ttnn.matmul(gnum, T3(col_t))            # [T,256,K]  d/d(w·T)
gcol  = ttnn.matmul(T3(M(w,Tt)), gnum)          # [T,K,3]  -> into color/keo grad (gco path)
gw   += M(gwT, Tt) ; gT = M(gwT, w)             # split w·T ; gw ALSO feeds the theta backward (below)
# T = exp(logT): gT_from_T = M(gT, Tt) ; # from T_bg: ga_bg = -M(gT_bg, T_bg) broadcast over K
glogT = M(gT, Tt)
# logT = -matmul(a, S^T): ga += ttnn.neg(matmul(glogT, S)) ; gS = ttnn.neg(matmul(T3(glogT), a_pg))  # [T,K,K]
ga    = A( ttnn.neg(ttnn.matmul(glogT, S)), broadcast(ga_bg) )            # [T,256,K]
gS    = ttnn.neg( ttnn.matmul(T3(glogT), a_pg) )                          # [T,K,K]  (sum over the 256 pixels)
# S = σ(d)·(1-I): gd = M(gS, M(S, 1-S))         # d = (zw_g - zw_h)·inv_tau
# zw grads: g(zw) per slot = ( rowsum_h gd[g,h] - colsum_g gd[g,h] ) * inv_tau        # [T,K,1]
# tau grad: gtau_tile = -inv_tau * Σ_{g,h} gd·d   -> reduce -> gpwtau_buf (host-Adam, like adam_sm)
# zw = (z-0.5)/7.5: g(mcz)_occ = g(zw) * (1/7.5)   -> per tile-slot; SCATTER to per-gaussian gmcz_occ
# a = -log(1-alpha): galpha = M(ga, 1/(1-alpha_pg)) ; gw += M(galpha, oc_t) ; goc += reduce_256(M(galpha, w))
```
Then:
- `gw` (poly-splat grad, `[T,256,K]`) feeds the **existing** theta backward (`gthU = matmul(T3(Phi), M(gw, 2·relu_Q))`) unchanged — geometry/theta path is reused.
- `gcol`,`goc` feed the existing color/opacity scatter (`gco`/`gocl` → `geom_bwd`).
- `gmcz_occ` (scattered to per-gaussian, `[G,1]`) is added to the **means** gradient in `geom_bwd` via
  the SM wiring: `out["mx"] += M(gmcz_occ, Rvb[2][0])`, my, mz. **This is the C3 z-force.** Because PW's
  `z` enters only through `S` (in `rend_bwd`), `gmcz_occ` originates per-tile-slot and must be
  **scattered** to per-gaussian (mirror the existing host-scatter of `gco`/`gocl`) before `geom_bwd`.
- `gpwtau` → host-Adam `adam_pw` for `pw_tau` (mirror SM `adam_sm`; `inv_tau=1/τ` buffer refreshed).

**Re-capture `rbid`** with the new grad buffers. Add `--arm-pairwise` flag + `--pwtau0` (default 0.01) +
`--lr-pwtau`; mutually exclusive with `--depth-weight`/`--arm-softmin`.

## Acceptance gate (the decision this port settles)
1. **Device oracle** — new `tools/pw_device_oracle.py` (mirror `softmin_device_oracle.py`): on small
   `G ≤ 256`, one device fwd+bwd, check `logT`, `T`, `T_bg`, and grads (`grad_o`, `grad_τ`, and the
   **means z-force** `grad_mcz`) vs host `blend_PW` autograd to tight `rel`. **This is the correctness
   gate — run it first.** (Orchestrator runs on hw; you write it + py_compile it, you cannot run hw.)
2. **Hardware smoke**: `sweep_resident.py --arm-pairwise --G 1000 --iters 30 --res 128` — exit 0, loss ↓.
3. **⚠ Quality gate (the real test)**: a device-trained PW must reach the **GPU PW number (lego 18.51)**.
   The device uses **per-tile `[K,K]` (K=128) occluders**, but the GPU bake-off used **global `[G,G]`**
   (all G as occluders). If device PW < 18.51, the **K=128 occluder restriction is the prime suspect**
   (a pixel's true occluder culled from its tile's nearest-K) → raise `K`, or widen the occluder set.
   Report device-PW holdout vs 18.51; this settles the design-notes "does binning cap K suffice" question.

## Constraints / notes
- Sort-free on device: NO `argsort`/`sort`/`cumsum`-in-depth. The `[K,K]` `S` is a comparison GEMM.
- Keep arm-A / arm-B / SM paths bit-identical (new `if args.arm_pairwise:` branches only).
- Cost: per-tile `[256,K]@[K,K]` (logT) + `[256,K]@[K,3]` (num) GEMMs, K=128 — matrix-engine-bound,
  bounded, no BP saturation (true per-(p,g) `a`, small where `w_geo` small).
- You CANNOT run Blackhole hardware — the orchestrator runs the device oracle + smoke + quality gate.
  Do host `py_compile` + write `pw_device_oracle.py`; do NOT run podman.

## Commit + report
Branch `feat/arm-pairwise-device`; commit (co-author + session trailer). Report: sha; the fwd/bwd
changes + where you injected the z-force (and whether you reused the SM branch or re-implemented); how
`gmcz_occ` is scattered to per-gaussian; py_compile; the exact hw commands for the orchestrator
(`pw_device_oracle.py`, the smoke, and the quality run to compare vs lego 18.51); and every place you
were unsure (esp. the `[K,K]` transpose bookkeeping and the scatter of the z-force).
