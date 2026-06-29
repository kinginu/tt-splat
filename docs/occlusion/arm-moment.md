# Hand-off — arm `MO` (moment-based OIT) · candidate #3

> **One Sonnet agent owns this file.** Read `01-arm-authoring-contract.md` + `02-device-map.md` first.
> §3-gated; ADD `MO` alongside arm A. Branch: `feat/arm-moment`.
>
> **Device difficulty: MEDIUM-HIGH** — **per-pixel** transmittance (modifies `rend_fwd`/`rend_bwd`,
> re-captures `rfid`/`rbid`), and a per-pixel reconstruction (small Cholesky) on SFPU/RISCV. Unlike
> softmin/B/B′ this does **not** fold into `keo`; it composites like OIT, not normalized WSR. Land
> after `SM`/`BP`. Reference: Münstermann, Krishnaswamy et al. 2018, *Moment-Based Order-Independent
> Transparency* (power-moment variant) — this hand-off follows that algorithm; cite it in code.

## What it is

Represent the per-pixel **transmittance function** `T(z)` by a few **power moments** of the
(depth, absorbance) measure, then reconstruct each gaussian's `Tᵢ = exp(−A(zᵢ))` from the moments —
**no sort, no prefix-product over a per-gaussian order**. Moments are pure GEMM; reconstruction is a
tiny per-pixel solve. It captures *where in depth the occluders sit* (the whole absorbance profile),
so it is strictly more expressive than a global monotone front-weight — a plausible **superset of
arm B**.

Per pixel `p`, with per-`(p,g)` alpha `α_pg = o_g·w_geo_pg` and absorbance `a_pg = −log(1−α_pg)`:

```
moments:  bₙ[p] = Σ_g a_pg · z_gⁿ ,   n = 0 … m            # m≈4 ⇒ [P, m+1], a [P,G]@[G,m+1] GEMM
recon:    Â_p(z) ≈ cumulative absorbance fraction in front of z, from {bₙ[p]}   # per-pixel solve
weight:   T_pg = exp( −b₀[p] · Â_p(z_g) ) ,   W_pg = α_pg · T_pg
composite (OIT over, NOT normalized WSR):
          C_p = Σ_g W_pg · c_g  +  exp(−b₀[p]) · c_b
```

`Σ_g α_pg T_pg ≈ 1 − exp(−b₀)` (telescoping), so the background gets the true residual transmittance
`exp(−b₀)` — this is proper alpha-compositing, reconstructed without ordering.

## Host implementation (per `01-arm-authoring-contract.md`)

> **Note the composite is OIT-style** (like `blend_RV`/`render_D`), **not** `_wsr`. Do not call `_wsr`.

```python
def blend_MO(w_geo, opacity_raw, depth, color, w_b, c_b, m=4, eps=1e-6):
    """Moment-based OIT (candidate #3): per-pixel transmittance from m power moments of the
    (depth, absorbance) measure; reconstruct T_i = exp(-A(z_i)), composite OIT-over. Sort-free
    (moments = GEMM, reconstruction = per-pixel Cholesky); z ATTACHED (C3). No learnable scalar in
    the base form (m is fixed); a learnable depth-warp is an optional extension."""
    o = torch.sigmoid(opacity_raw)
    alpha = (o[None, :] * w_geo).clamp(eps, 1.0 - 1e-4)            # [P,G]
    a = -torch.log1p(-alpha)                                      # [P,G] absorbance
    z = _depth_warp(depth)                                        # [G] -> [0,1]-ish; keep attached
    zp = torch.stack([z ** n for n in range(m + 1)], dim=-1)      # [G, m+1] power basis
    b = a @ zp                                                    # [P, m+1] moments (the GEMM)
    A_frac = moment_reconstruct(b, z, m, eps)                     # [P,G] reconstructed cum-absorbance fraction in front
    T = torch.exp(-(b[:, :1] * A_frac))                           # [P,G] transmittance at each z_g
    W = alpha * T                                                 # [P,G]
    num = torch.einsum("pg,gc->pc", W, color)                     # Σ α T c
    T_bg = torch.exp(-b[:, 0:1])                                  # [P,1] residual transmittance
    return num + T_bg * c_b[None, :]
```

- **`_depth_warp(z)`**: map camera z to a stable `[0,1]` range for the power basis (powers of raw z
  overflow/condition badly). Use `(z − z_near)/(z_far − z_near)` with detached per-image min/max, or a
  fixed near/far. State your choice; keep `z` attached through the warp for C3.
- **`moment_reconstruct(b, z, m)`** — **the meaty part.** Follow the power-moment reconstruction from
  Münstermann 2018: normalize moments by `b₀`, form the Hankel system from `{bₙ/b₀}`, Cholesky-solve
  the small `(m/2+1)×(m/2+1)` system per pixel, evaluate the bounded reconstruction of the CDF at each
  `z_g`, clamp to `[0,1]`. Implement it as a clean torch function; **validate it** (next section).
  If full MBOIT is too much for v1, ship the **trigonometric-moment / Fourier variant (candidate #7)**
  as a fallback (often simpler to bound) and note the swap.

**`spike/render.py`**: `ARMS += ("MO",)`; `elif arm == "MO": img = arms.blend_MO(w_geo, model.opacity_raw, depth, color, w_b, c_b)`.

**`spike/model.py`**: base form has **no new learnable scalar** (m is a fixed hyperparameter). If you
add the optional learnable depth-warp endpoints, register them under `lr["depth"]`.

**`spike/m05_spike.py`**: add `"MO"` to `CANDIDATE_ARMS`.

**`spike/tests/test_toy_occlusion.py`**: the 2-gaussian probe must occlude (front red dominates).
With m≥2 the moments separate the two depths; `T_far = exp(−a_near) ≈ 0.1`:
```python
def test_moment_occludes():
    s = _setup(); C = arms.blend_MO(s["w_geo"], s["opacity_raw"], s["depth"], s["color"], s["w_b"], s["c_b"], m=4)[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.80 and C[2].item() < 0.20, C       # moment recon is approximate; looser bound

def test_moment_depth_order():
    s = _setup(); C = arms.blend_MO(s["w_geo"], s["opacity_raw"], _f([5.0,2.0]), s["color"], s["w_b"], s["c_b"], m=4)[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C

def test_moment_zgrad():
    s = _setup(); z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_MO(s["w_geo"], s["opacity_raw"], z, s["color"], s["w_b"], s["c_b"], m=4)
    (g,) = torch.autograd.grad(C.sum(), z); assert g.abs().sum() > 0
```

## Validation oracle (replaces the β/τ-style backward oracle)

Moment OIT's correctness risk is the **reconstruction**, not a hand-derived backward (autograd covers
the backward). So the oracle is **forward fidelity vs the exact sorted transmittance**:
`tools/moment_oracle.py` — on random `(z, α)` per pixel, compute the EXACT front-to-back transmittance
`T_i = Π_{j in front}(1−α_j)` (sorted, brute force, this is the ground truth) and compare to
`blend_MO`'s reconstructed `T`. Report mean/max abs error vs `m ∈ {2,4,6}` and assert the m=4 error is
below a stated bound (e.g. mean < 0.05). This quantifies the moment approximation and picks `m`.

## Device implementation (per `02-device-map.md`)

**Per-pixel** ⇒ modify `rend_fwd`/`rend_bwd` (NOT the `keo` fold). After the existing gather:

**Forward (in `render_fwd_cache`)**: gather `z_t[T,K,1]` (`ttnn.embedding(idx_u, mcz)`), build the
per-tile power basis `zp_t[T,K,m+1]`, and `a_t[T,K,1] = −log(1−oc_t)` (absorbance from the gathered
`keo`/α). Then the moments are a **per-pixel GEMM**:
```python
# w[T,256,K] is the poly-splat per-(pixel,gaussian) weight; a_pg = w · a_t broadcast
b = ttnn.matmul(M(w, a_t_bcast), zp_t, core_grid=CG)   # [T,256,m+1] per-pixel moments  (the GEMM)
```
Then the **reconstruction** `Â(z_g)` per (pixel, gaussian): the `(m/2+1)²` Cholesky solve per pixel on
SFPU/RISCV (m=4 ⇒ 3×3 solve), evaluate at each `z_g`, `T = exp(−b₀·Â)`, `W = α·T`, and composite
`num = matmul(W, col_t)`, `C = num + exp(−b₀)·bias`. This is the novel kernel — **the moments are
matrix-engine-bound (good), the reconstruction is the SFPU piece.**

**Backward (in `render_bwd`)**: autograd-equivalent grads through the composite, moments GEMM (transpose
matmul), reconstruction (the Cholesky solve's backward — differentiate the small linear solve), and
`a_t`→`o`/`z`. Provide a device-vs-host grad check on small `T,K`. The reconstruction backward is the
hardest part — consider a fixed-`m` closed form for the 3×3 case.

**Re-capture**: `rfid`/`rbid` must be re-captured with `zp_t`, `b`, and the reconstruction buffers
pre-allocated. No `keo` change, no host-Adam scalar (base form).

## Sizing / cost
- Moments: `[T,256,K]·[T,K,m+1]` GEMM = `(m+1)/3 ×` a color-channel GEMM — cheap, matrix-engine bound.
- Reconstruction: `O(256·(m/2)³)` per tile on SFPU — the real cost; `m=4` keeps it a 3×3 solve.
- Memory: `+[T,256,m+1]` moments + `[T,K,m+1]` basis. Bounded by `m` (4–6).

## Acceptance
- Host: `test_toy_occlusion` (+3) passes; `moment_oracle.py` reports m=4 forward error below the stated
  bound vs exact sorted transmittance; arm A bit-identical; `m05_spike --arms A,B,MO,D` includes `MO`.
- Device: small-`T,K` device fwd+grad match host; hardware smoke (`sweep_resident.py --arm-moment
  --G 1000 --iters 30 --res 64`) loss decreases.
- Result note: MO vs B vs SM/BP holdout PSNR on lego; does modelling the *profile* (not just a front
  bias) help on multi-layer occluders; chosen `m`.

## Risks / open
- **Reconstruction is the research piece** — power-moment bounding can over/under-shoot; the Fourier
  (#7) variant is the documented fallback. Budget time here; validate against the sorted oracle early.
- Power-basis conditioning ⇒ the `_depth_warp` to `[0,1]` is mandatory; report ringing if it appears.
- Reconstruction backward on device (differentiating the Cholesky solve) — prefer the fixed-`m` closed
  form; flag if autograd-on-device is infeasible and a custom adjoint is needed.
