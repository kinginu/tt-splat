# Hand-off — arm `BP` (pairwise soft-occlusion GEMM, "arm B′") · candidate #5

> **One Sonnet agent owns this file.** Read `01-arm-authoring-contract.md` + `02-device-map.md` first.
> §3-gated; ADD `BP` alongside arm A. Branch: `feat/arm-bprime`.
>
> **Device difficulty: MEDIUM** — per-gaussian ρ folds into `keo` (arm-B pattern), but ρ comes from an
> `O(G²)` (or per-tile `O(K²)`) `[G,G]` GEMM. **Best fit to the perf thesis** (keeps the matrix engine
> busy). Land **softmin (`SM`) first** — it de-risks the shared `keo`/`gkeo`/z-force device wiring.

## What it is

The sort-free realization of arm B's rank. The "number of gaussians in front of i" is a soft compare
matrix row-sum (a GEMM); weighting that sum by each occluder's absorbance recovers **true cumulative
transmittance**:

```
Sᵢⱼ = σ( (zᵢ − zⱼ) / τ )           # [G,G] soft "j is in front of i" (zⱼ<zᵢ ⇒ Sᵢⱼ→1); Sᵢᵢ := 0
βⱼ  = −log(1 − αⱼ),  αⱼ ≈ oⱼ        # per-gaussian absorbance (opacity proxy; footprint stays in w_geo)
log Tᵢ = −(S β)ᵢ                    # prefix-product → exp(−sum); the S β is the GEMM
ρᵢ = exp(log Tᵢ)                    # per-gaussian transmittance
Wᵢ = oᵢ · w_geo · ρᵢ                # fold ρ into the WSR column (keo); _wsr as usual
```

- **τ → 0**: `S` → hard step ⇒ `log Tᵢ → Σ_{j in front} log(1−αⱼ)` ⇒ `ρᵢ → Π_{j in front}(1−αⱼ)` =
  **exact alpha-transmittance**. This is the limit arm B (monotone sigmoid) can only approximate.
- **τ → ∞**: `S → ½` ⇒ near-uniform attenuation ⇒ degrades gracefully toward a global dim.

Diagonal `Sᵢᵢ = σ(0) = ½` is **self-occlusion** — zero it (`S ← S·(1−I)`; a gaussian does not occlude
itself; exclusive transmittance). Cost `O(G²)` but on BH's fastest resource. Cheaper than a
differentiable sort (it builds only the prefix-occlusion, not a permutation matrix).

## Host implementation (per `01-arm-authoring-contract.md`)

**`spike/arms.py`**:
```python
def blend_BP(w_geo, opacity_raw, depth, bp_tau, color, w_b, c_b, eps=1e-4):
    """Pairwise soft-occlusion GEMM ('arm B′', candidate #5): per-gaussian transmittance
    rho_i = exp(-(S beta)_i), S_ij = sigmoid((z_i - z_j)/tau) (j in front of i, diag zeroed),
    beta_j = -log(1-o_j). tau->0 = exact alpha-transmittance Prod(1-o_j). Sort-free (S = [G,G]
    GEMM, exp lane-wise); z ATTACHED (C3). One learnable scalar tau>0."""
    o = torch.sigmoid(opacity_raw)
    tau = bp_tau.clamp(min=1e-3)
    beta = -torch.log1p(-(o.clamp(max=1.0 - eps)))                 # [G]
    S = torch.sigmoid((depth[:, None] - depth[None, :]) / tau)     # [G,G]
    S = S * (1.0 - torch.eye(S.shape[0], dtype=S.dtype, device=S.device))   # zero diagonal
    rho = torch.exp(-(S @ beta))                                   # [G]
    return _wsr(o[None, :] * w_geo * rho[None, :], color, w_b, c_b)
```

**`spike/model.py`**: `self.bp_tau = nn.Parameter(torch.tensor(2.0, …))`; register under `lr["depth"]`.
Init `τ = 2.0` (soft regime, scene z-range ~4) — starts near a gentle global occlusion, learns down.

**`spike/render.py`**: `ARMS += ("BP",)`; `elif arm == "BP": img = arms.blend_BP(w_geo, model.opacity_raw, depth, model.bp_tau, color, w_b, c_b)`.

**`spike/m05_spike.py`**: add `"BP"` to `CANDIDATE_ARMS`.

**`spike/tests/test_toy_occlusion.py`** (2 gaussians, near red z=2 / far blue z=5, both o=0.9 ⇒ β≈2.3):
with τ small, `S[far,near]=σ((5−2)/τ)→1` so `ρ_far=exp(−β_near)≈exp(−2.3)≈0.1`, `ρ_near=exp(0)=1` ⇒
front (red) dominates:
```python
def test_bprime_occludes():
    s = _setup(); C = arms.blend_BP(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.5), s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.85 and C[2].item() < 0.15, C
    C2 = arms.blend_BP(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.2), s["color"], s["w_b"], s["c_b"])[0]
    assert C2[0].item() >= C[0].item()                 # smaller tau -> harder

def test_bprime_depth_order():
    s = _setup(); C = arms.blend_BP(s["w_geo"], s["opacity_raw"], _f([5.0,2.0]), _f(0.5), s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C

def test_bprime_zgrad():
    s = _setup(); z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_BP(s["w_geo"], s["opacity_raw"], z, _f(0.5), s["color"], s["w_b"], s["c_b"])
    (g,) = torch.autograd.grad(C.sum(), z); assert g.abs().sum() > 0
```

## Analytic backward (oracle + device)

`ρᵢ = exp(log Tᵢ)`, `log Tᵢ = −Σⱼ Sᵢⱼ βⱼ`, `Sᵢⱼ = σ(dᵢⱼ)`, `dᵢⱼ = (zᵢ−zⱼ)/τ`. WSR hook
`grad_ρᵢ = gkeoᵢ·keepᵢ·oᵢ`:

```
grad_logTᵢ = grad_ρᵢ · ρᵢ
grad_Sᵢⱼ   = −grad_logTᵢ · βⱼ
grad_βⱼ    = −Σᵢ grad_logTᵢ · Sᵢⱼ = −(Sᵀ grad_logT)ⱼ
grad_dᵢⱼ   = grad_Sᵢⱼ · Sᵢⱼ(1−Sᵢⱼ)
grad_zᵢ   += Σⱼ grad_dᵢⱼ / τ           # rowsum/τ ...
grad_zⱼ   += −Σᵢ grad_dᵢⱼ / τ          # ... minus colsum/τ  ⇒ grad_z = (rowsum − colsum)/τ
grad_τ     = −(1/τ) Σᵢⱼ grad_dᵢⱼ · dᵢⱼ
grad_opⱼ  += grad_βⱼ · oⱼ              # β=−log(1−o), o=σ(op) ⇒ chain = ·o ; ADD to existing o-grad
```
`grad_z` ADDs to the mcz→means path (C3 — see device note). **Oracle** `tools/bprime_oracle.py`:
autograd through `blend_BP` vs these formulas for `grad_τ, grad_z, grad_op`; `rel < 1e-5`.

## Device implementation (per `02-device-map.md`)

Per-gaussian ρ ⇒ folds into `keo` like arm B. The new piece is producing ρ from a GEMM. Two routes —
**do (A) first to validate the math, then (B) if `[G,G]` busts L1.**

**(A) Global `[G,G]` in `geom_fwd` (simplest, validate-first).** Using `z = cache["mcz"]` `[G,1]`:
```python
# S[G,G] = sigmoid((z_i - z_j)/tau): outer difference then SFPU sigmoid.
D  = ttnn.sub(z, T_(z)) * inv_tau          # [G,G] outer diff (z broadcast row - col); inv_tau scalar
S  = ttnn.sigmoid(D); S = M(S, one_minus_I)# zero diagonal ([G,G] const buffer 1-I)
beta = ttnn.neg(ttnn.log(ttnn.sub(1.0, o)))# [G,1]
logT = ttnn.neg(ttnn.matmul(S, beta, core_grid=CG))   # [G,1] GEMV — the matrix-engine step
rho  = ttnn.exp(logT)
keo  = M(M(keep, o), rho)
```
Backward in `geom_bwd` off `gkeo`, following the formulas above: `grad_logT = M(grad_rho, rho)`;
`grad_S = -outer(grad_logT, beta)`; `grad_beta = -matmul(T_(S), grad_logT)`; `grad_d = M(grad_S, M(S,1-S))`;
`grad_z = (rowsum(grad_d) - colsum(grad_d)) * inv_tau` → **add to mcz grad**; `grad_op += M(grad_beta, o)`;
`grad_tau = -inv_tau * sum(M(grad_d, D))` → `gtau_buf`.

**(B) Per-tile `[T,K,K]` in `rend_fwd` (bounded `O(K²)`, production).** After gather, `z_t[T,K]` (gather
`mcz` via `idx_u`); build `S_t[T,K,K]`, `logT_t[T,K,1] = −matmul(S_t, beta_t)`, `ρ_t = exp`, fold into
`oc_t` (and `col_t`). More accurate (tile-local occluders) but **re-captures `rfid`/`rbid`** and needs
the per-tile ρ grad scattered back. `K=128` ⇒ `[T,128,128]` per tile = bounded.

**L1 sizing decision (the deferred §-question):** at the target (res128, G2000), `[2000,2000]` bf16 ≈
8 MB — likely **does not** fit L1 ⇒ route (A) is a small-G validation harness, route (B) is production.
Measure: does `[G,G]` fit for the G you actually train? If not, ship (B). Report the crossover.

**Host-Adam scalar `τ`**: `--bptau-0` (2.0) + `--lr-bptau`; `inv_tau`/`inv_tau2` buffers; `gtau_buf`;
step `adam_bt`-style. **z-force C3 wiring**: same critical note as `arm-softmin.md` — add `grad_z`
into the mcz→means accumulator; share/verify the fix with softmin.

## Sizing / cost
- (A) `O(G²)` elementwise (outer-diff + σ) + a `[G,G]·[G,1]` GEMV fwd, transpose-GEMM bwd. Matrix-engine
  bound (good). (B) `O(T·K²)` — bounded, scales with tiles not G². Forward adds one GEMV; backward one
  transpose-GEMM + reductions.
- Strictly heavier than softmin; **the point** is it recovers *cumulative* occlusion (how much is
  packed in front), which a per-gaussian monotone weight cannot.

## Acceptance
- Host: `test_toy_occlusion` (+3) passes; `bprime_oracle.py` PASS (`grad_τ,grad_z,grad_op` rel<1e-5);
  arm A bit-identical; `m05_spike --arms A,B,SM,BP,D` includes `BP`.
- Device: route (A) small-G (`G≤256`) device grads match host oracle; smoke on hardware
  (`sweep_resident.py --arm-bprime --G 256 --iters 30 --res 64`) loss decreases; then route (B) at
  `G=2000` if `[G,G]` busts L1. Report the L1 crossover and (A)-vs-(B) quality delta.
- Result note: BP vs B vs SM holdout PSNR on lego solid-occluder; does the τ→0 exact-transmittance
  limit beat softmin's front-takes-all on stacked occluders.

## Risks / open
- **L1 fit** of `[G,G]` (the headline open question) — measured, with (B) as the fallback.
- Per-tile (B) uses only the K tile-local gaussians as occluders — an approximation vs global; validate
  it doesn't under-occlude when true occluders are culled from a tile's K-list.
- `exp(log T)` underflow for deep stacks — clamp `log T ≥ −30`; report if it bites.
