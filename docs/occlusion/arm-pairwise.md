# Hand-off — arm `PW` (moment-free per-pixel pairwise soft-occlusion) + GPU bake-off

> **One Sonnet agent.** Instructions only — implement in torch + enable a GPU comparison; do NOT do the
> tt-nn port (that follows, on the winner). Branch: `feat/arm-pairwise` off `feat/mo-sortfree` (it has
> the sort-free MO reconstruction + the extended `moment_oracle`).

## Why (the insight from the MO sort-free result)

The GPU re-validation showed the only sort-free reconstruction that holds the lego gap is the
**pairwise soft-compare** `A_frac = w @ Sᵀ` (variant A / softcmp: lego 18.32 ≈ the sorted 18.45), while
the cheap closed-form MBOIT (variant B) degrades and is unstable. But MO-softcmp still runs a **moment
solve** to recover `w ≈ a/b0` before the compare — and that recovery is a **redundant, lossy** step:

```
MO-softcmp:  b = a@zp → solve w≈a/b0 → A_frac = w@Sᵀ → T = exp(-b0·A_frac) ≈ exp(-(a@Sᵀ))
```

i.e. the working core is just `T ≈ exp(-(a @ Sᵀ))` on the **true** per-(p,g) absorbance `a`. Dropping
the moment solve gives the pure **pairwise soft-occlusion** — the working, per-pixel realization of the
design-notes **arm B′ (#5)**, *without* BP's `G=2000` `rho≈0` saturation (BP summed occluders over all
`G` with per-gaussian `o`; here it's per-tile `K` occluders with true per-(p,g) `a`, which is small
where `w_geo` is small). Expectation: **≥ MO-softcmp quality at ≤ cost, and closer to the D ceiling**
(no moment-recovery loss). This bake-off decides the tt-nn port target so we don't bake a redundant
moment solve into a Blackhole kernel.

## Implement arm `PW` (torch)

Simplest: add a `recon="pairwise"` mode to `blend_MO` (skip the moment solve; use `a` directly), and
register a new arm letter `PW` that calls it. Keep MO's OIT-over composite (NOT `_wsr`).

```python
# in blend_MO (or a thin blend_PW): moment-free pairwise branch
o = torch.sigmoid(opacity_raw)
alpha = (o[None, :] * w_geo).clamp(eps, 1.0 - 1e-4)          # [P,G]
a = -torch.log1p(-alpha)                                      # [P,G] true absorbance
zw = _depth_warp(depth)                                       # [G] in [0,1], ATTACHED (C3)
S = torch.sigmoid((zw[:, None] - zw[None, :]) / tau)          # [G,G]  S[g,h]=σ((z_g-z_h)/τ) ("h in front of g")
S = S * (1.0 - torch.eye(G, ...))                            # exclusive
logT = -(a @ S.transpose(-1, -2))                            # [P,G]  = -Σ_h a[p,h]·S[g,h]
T = torch.exp(logT.clamp(min=-30.0))                         # [P,G] transmittance
W = alpha * T                                                 # [P,G]
num = W @ color                                              # [P,3]
T_bg = torch.exp(-a.sum(dim=1, keepdim=True).clamp(min=0.0)) # [P,1] residual transmittance past all
return num + T_bg * c_b[None, :]
```

- Reuse the same `tau` MO-softcmp uses (sharp, e.g. `0.01`; expose as the same knob / learnable scalar
  under `lr["depth"]`). Start with the same value softcmp used so the comparison is apples-to-apples.
- `z` stays attached through `S` (C3). No `argsort`/`sort`/`cumsum`.
- Register `"PW"` in `spike/render.py` `ARMS` + dispatch, and in `spike/m05_spike.py` `CANDIDATE_ARMS`.

## Validate (agent, host/CPU only)

1. Extend `tools/moment_oracle.py` (or add `tools/pairwise_oracle.py`) to check `PW`'s forward vs the
   exact sorted transmittance: at sharp `τ` it should match to ~0 (it's the true absorbance cumulative,
   soft-compared) — report mean/max abs error vs `τ ∈ {0.1, 0.03, 0.01}`.
2. **No-sort proof**: grep clean for `argsort`/`.sort(`/`cumsum` in the `PW`/reconstruction path.
3. `spike/tests/test_toy_occlusion.py`: add `PW` occlusion + depth-order + z-grad tests (front red
   dominates; `dL/dz` nonzero on the far gaussian).
4. arm A bit-identical; `python -m py_compile spike/arms.py spike/render.py spike/m05_spike.py`.
You CANNOT run the GPU sweep — that's the user's step.

## GPU bake-off (the user runs on the GPU box) — decides the port target

```bash
python -m spike.m05_spike --scene data/nerf_synthetic/lego  --arms A,B,MO,PW,D \
    --G 2000 --res 64 --iters 600 --seeds 3 --device cuda --out outputs/m05_pw_lego
python -m spike.m05_spike --scene data/nerf_synthetic/ficus --arms A,B,MO,PW,D ... --out outputs/m05_pw_ficus
```
**Compare** PW vs MO-softcmp (lego 18.32 / ficus 24.66-ish for softcmp; old sorted MO 18.45 / 25.40) and
vs D (lego 19.78 / ficus 25.40). Decision:
- **PW ≥ MO-softcmp** (expected) → **PW is the tt-nn port target** (simpler kernel: per-tile `[K,K]`
  soft-compare on gathered absorbance, no moment solve; reuse SM's z-force wiring). If PW closes more of
  the gap to D, even better.
- PW < MO-softcmp (unexpected) → the moment solve was doing something useful; port MO-softcmp instead.
Report a 1-table summary: A / B / MO(softcmp) / PW / D holdout PSNR on lego+ficus + seed std.

## Device implication (for the follow-on port, not now)
PW's device form = per-tile `[T,K,K]` soft-compare GEMM on gathered `a_t` (matrix-engine-bound, K=128
bounded — same sizing as BP but no saturation), then `T_pg = exp(logT)`, OIT-over composite in
`rend_fwd`. Backward reuses the SM z-force pattern (`grad_mcz → means`) plus the `[K,K]` GEMM adjoint.

## Commit + report
Commit on `feat/arm-pairwise` (co-author + session trailer). Report: sha; the PW forward code; the
no-sort proof; oracle fidelity vs τ; toy + arm-A confirmation; py_compile; and the exact GPU bake-off
commands + baselines (MO-softcmp 18.32 / D 19.78 on lego) to make the port-target decision.
