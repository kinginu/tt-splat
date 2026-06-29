# Device (Blackhole / ttnn) render-path map — for the occlusion device ports

> Derived by reading `tools/{resident_traced,sweep_resident,geom_device,traced_fwd,m4_train_binned}.py`.
> **File:line refs drift** — re-grep the named functions before editing. This map exists so each
> hand-off's §Device can name exact interfaces and insertion points.

`sweep_resident.py` is the multi-view trainer; `resident_traced.py` holds the authoritative
`render_bwd`/`theta_bwd` (imported by sweep_resident) and the single-view traced pipeline. Both share
the same trace structure.

## 0. The big picture (what differs from the host `[P,G]` arm)

The device path is **binned per-tile `[T,K]`**, not dense `[P,G]`:
- Tile = `16×16` px (`assert H%16==0`); `T` = (H/16)(W/16) tiles; **256 px/tile**.
- `K = 128` (`--K`): per-tile cap on nearest gaussians (host `assign_bins`, R=1 stencil).
- The WSR weight per (pixel,gaussian) is `w[T,256,K]`; num/den are **GEMMs** `matmul(w, col)` / `matmul(w, oc)`.

The per-gaussian opacity column is **`keo = keep·o·(·rho)` `[G,1]`** — this is THE fold-in point an
arm-B-style per-gaussian multiplier targets. Color columns `co[ch] = keo·color[ch]`.

## 1. Forward

- **`geom_device.device_fwd_core(cols, Rv, tv, …)`** → `conic, mu2d, cache`.
  - `cache["mcz"]` = camera-space z `[G,1]` fp32 = `Rv[2,:]·means + tv[2]` (geom_device.py ~:69).
  - `cache["zmask"]` = front-of-camera mask `[G,1]` bf16.
- **`geom_fwd()`** (closure, trace `gfid`; resident_traced.py ~:327, sweep_resident.py ~:236):
  ```python
  keep = cache["zmask"]; o = ttnn.sigmoid(P["op"])         # [G,1]
  # arm-B branch (depth-weight):
  rho = ttnn.sigmoid(M(beta_buf, A(tau_buf, ttnn.neg(cache["mcz"]))))  # σ(β(τ−z)) [G,1]
  keo = M(M(keep, o), rho)                                  # else keo = M(keep,o)
  co  = [M(keo, color[ch]) for ch in (0,1,2)]               # [G,1] each
  ```
- **`traced_fwd.gather_theta_cols(...)`** → `theta[T,6,K], col_t[T,K,3], oc_t[T,K,1]` (per-tile gather
  via `ttnn.embedding(idx_u, col)`).
- **`render_fwd_cache(Phi, thU, col, oc, wbb, bias)`** (inner; resident_traced.py ~:365):
  ```python
  relu_Q = ttnn.relu(ttnn.matmul(Phi, thU, core_grid=CG))   # [T,256,K]
  w      = ttnn.square(relu_Q)                              # poly-splat weights
  den    = ttnn.add(ttnn.matmul(w, oc,  core_grid=CG), wbb) # [T,256,1] WSR denominator
  num    = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)# [T,256,3] WSR numerator
  C      = ttnn.div(num, den)                               # [T,256,3]
  ```

## 2. arm-B depth-weight (the per-gaussian fold-in template)

`rho` computed **on device** in `geom_fwd` from `cache["mcz"]`; folded `keo = M(M(keep,o),rho)`.
`beta_buf`,`tau_buf` are `[G,1]` fp32 broadcast buffers; **optimised by HOST Adam** over two python
floats (`adam_bt`, resident_traced.py ~:503 / sweep_resident.py ~:354). Allocation block:
```python
beta_buf  = u(torch.full((G,), beta0)); tau_buf = u(torch.full((G,), tau0))   # device [G,1]
gbeta_buf = u(torch.zeros(G)); gtau_buf = u(torch.zeros(G))                    # grad accumulators
bt = {"beta": beta0, "tau": tau0}; bt_m = {...0}; bt_v = {...0}                # host Adam state
```

## 3. Depth z

`cache["mcz"]` `[G,1]` fp32 device — **available**, lives inside the `gfid` trace. Host does not
download it routinely (costs `synchronize_device` + `dn()`).
- **Per-gaussian use** (B / softmin / B′): consume `mcz` directly in `geom_fwd` → `keo`. zero extra cost.
- **Per-pixel Σw·z** (SZ / moment): gather `z_t = ttnn.embedding(idx_u, mcz)` `[T,K,1]`, then
  `matmul(w[T,256,K], z_t)` → `[T,256,1]` inside `rend_fwd` (requires re-capturing `rfid`/`rbid`).
- **O(G²) pairwise** (B′ global): needs a `[G,G]` (or per-tile `[K,K]`) device op — a new trace/op.

## 4. Backward

- **`render_bwd(Phi, col, oc, gC, cache)`** (resident_traced.py ~:60) → `gthU[T,6,K], gcol[T,K,3], goc[T,K,1]`.
  `goc = matmul(T3(w), gden)` = grad into the gathered `keo` column (via WSR denominator).
- **`theta_bwd(gthU, …)`** → `gconic, gmu`.
- **`geom_bwd()`** (trace `gbid`; resident_traced.py ~:397) builds the per-gaussian **`gkeo` `[G,1]`**:
  ```python
  gkeo = A(A(M(gco_buf[0],color[0]), M(gco_buf[1],color[1])),
           A(M(gco_buf[2],color[2]), gocl_buf))      # THE hook point
  # arm-B: grad_rho = M(M(gkeo,keep),o); grad_pre = grad_rho·rho·(1−rho);
  #        gbeta = grad_pre·(tau−z); gtau = grad_pre·beta  → copy into gbeta_buf/gtau_buf
  ```
  **`gkeo` is the universal hook** for any per-gaussian occlusion multiplier `m_i` folded as
  `keo = keep·o·m`: `grad_m_i = gkeo_i · keep_i · o_i`, then chain through `m_i`'s own math.

## 5. Traced pipeline (per-iter replay)

```
execute(gfid)  → [sync@bin_every] host assign_bins + setbuf(idx_u,valid6,vf)
→ execute(rfid) → execute(lfid) → execute(rbid) → sync
→ HOST scatter (index_add → setbuf gcon_buf/gco_buf/gocl_buf)
→ execute(gbid) → adam_inplace() → adam_bt() [if depth-weight]
```
Capture order: `gfid, rfid, lfid, rbid, (sbid), gbid` (sweep_resident.py ~:424).

**Insertion points**
- **Per-gaussian multiplier** (B / softmin / B′-global): add the op in `geom_fwd` (fold into `keo`)
  + the backward in `geom_bwd` (off `gkeo`). **Zero re-capture if added before initial capture.**
- **Per-pixel state** (SZ / moment / B′-per-tile): add buffers + GEMMs in `rend_fwd` and grads in
  `rend_bwd`; **must re-capture `rfid`/`rbid`** with the new buffers pre-allocated.

## 6. Adding a new host-Adam scalar (τ, α, …)

1. argparse `--<name>-0`, `--lr-<name>`; allocate `name_buf` `[G,1]` (broadcast) + `gname_buf` `[G,1]`
   (grad accumulator) **before trace capture**; host dicts `val/m/v`.
2. Use `name_buf` in `geom_fwd` (same trace as `rho`).
3. In `geom_bwd`, `ttnn.copy(<grad expr>, gname_buf)`.
4. After `gbid`+sync: host `g = float(dn(gname_buf).sum())`; Adam step; `setbuf(name_buf, full((G,),val))`.

## Constants
- `M(a,b)=ttnn.mul`, `A(a,b)=ttnn.add`, `T3` = transpose helper, `u(t)` = host→device upload,
  `dn(t)` = device→host download, `setbuf` = in-place device copy, `CG` = CoreGrid.
- `B1,B2,EPS` (Adam) and `PN,B1,B2` exported from `resident_traced.py`.
- Tile 16×16, 256 px/tile, `K=128`, `idx_u[T,K]` uint32, `Phi[T,256,6]`, `valid6[T,6,K]`, `vf[T,K,1]`.
