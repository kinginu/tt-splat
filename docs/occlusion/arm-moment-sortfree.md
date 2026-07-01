# Hand-off — make arm `MO` genuinely sort-free, then GPU-revalidate (candidate #3)

> **One Sonnet agent owns this.** Instructions only — implement in torch + enable a GPU re-validation;
> do NOT do the tt-nn device port yet (that is a separate follow-up, gated on the re-validation below).
> Branch: `feat/mo-sortfree` off `feat/arm-integration`.

## Why this task exists (the finding that triggered it)

The GPU triage picked `MO` as the only arm that closes the lego occlusion gap (ficus 25.40 = the D
ceiling exactly; lego 18.45, +2.27 vs A). **But the current `spike/arms.py::_moment_reconstruct` is NOT
sort-free** — lines 80-86 do:
```python
order = zw.argsort().detach()                 # a depth SORT
w_sorted = w[:, order]
cum_excl = cat([0, w_sorted[:, :-1].cumsum(1)])   # prefix-sum in depth order
A_hat[:, order] = cum_excl
```
That argsort + depth-ordered cumsum is exactly the workload §3 bans on Blackhole (data-dependent sort;
matrix engine idles). MO's win is essentially **sorted alpha-compositing (arm D) with the sort hidden
in the reconstruction** — which is why MO ≈ D. The moment solve (lines 60-78, recovering `w ≈ a/b0`) is
already sort-free; **only the cumulative step uses a sort.** Goal: replace that step with a genuinely
sort-free reconstruction and re-confirm on GPU that the gap still closes. If it does NOT, the honest
conclusion is "MO's win was sort-dependent" — report that.

## The math context (unchanged parts you keep)

`blend_MO` (keep its interface + the OIT-over composite): per-(p,g) `alpha = o·w_geo`, absorbance
`a = -log(1-alpha)`, warped depth `zw = _depth_warp(z) ∈ [0,1]`, power moments `b = a @ zp` (`zp[g,n]=zw_g^n`),
recovered per-pixel weights `w[P,G]` from the moment solve (lines 70-78, sort-free, keep it), then
`A_frac = reconstruct(...)`, `T = exp(-b0·A_frac)`, composite `num = (alpha·T) @ color + exp(-b0)·c_b`.
`z` must stay **attached** (C3). Only `_moment_reconstruct`'s ordering step changes.

## Implement BOTH sort-free reconstructions and compare

### (A) Soft-compare GEMM reconstruction — the quality-preserving baseline (do first)
Replace the argsort+cumsum with a pairwise soft-compare (no sort):
```python
# A_frac[p,g] = Σ_h w[p,h] · [z_h in front of z_g]  ->  soft, sort-free, z-differentiable:
S = torch.sigmoid((zw[:, None] - zw[None, :]) / tau)     # [G,G]  S[g,h] = σ((z_g - z_h)/τ)  ("h in front of g")
S = S * (1.0 - torch.eye(G, ...))                        # exclusive (no self)
A_frac = (w @ S.transpose(-1, -2)).clamp(0.0, 1.0)       # [P,G] @ [G,G] -> [P,G]
```
- **At the hard limit (τ→0) this is bit-identical to the current sorted reconstruction** — i.e. it
  *proves* the gap is closable sort-free (same numbers), just at `O(P·G²)` instead of `O(P·G + G log G)`.
- Finite `τ` is what gives the C3 z-force (the soft step is differentiable in z). Add `tau` (start as a
  fixed hyperparameter ≈ a few % of the warped depth range, e.g. 0.05; optionally make it a learnable
  scalar registered like `bp_tau`, under `lr["depth"]`).
- This is structurally **per-pixel BP** (the same `[G,G]` soft-compare). Note that for the device the
  feasible form is the **per-tile `[K,K]`** (K=128) version — same sizing resolution as BP.

### (B) True power-moment MBOIT closed-form — the efficient thesis-ideal (do second)
Reconstruct `A_frac(z_g)` **directly from the m+1 moments**, `O(P·m)`, no pairwise term, no sort —
the real Münstermann et al. 2018 power-moment OIT reconstruction (cite it in code): normalize moments
by `b0`, build the Hankel system, Cholesky-solve the small `(m/2+1)²` per-pixel system, evaluate the
**bounded** CDF reconstruction at each `z_g`, clamp `[0,1]`. If the power-moment bounding is too
fiddly, the **trigonometric-moment / Fourier variant (candidate #7)** is the documented, often-more-
robust fallback — implement that instead and say so. This is the form that keeps the matrix engine busy
(moments GEMM) with a tiny per-pixel solve, and is what a real tt-nn port would use.

Keep `(A)` as the reference (it pins the achievable quality); `(B)` is the one that must hold the gap
at low cost. Expose the reconstruction choice via an arg/kwarg (e.g. `recon="softcmp"|"mboit"`).

## Validation YOU (the agent) do — host/CPU only

1. `python tools/moment_oracle.py` — extend it to test BOTH reconstructions' forward fidelity vs the
   exact sorted transmittance (it already does m∈{2,4,6}); assert (A) at small τ matches to ~0 and (B)
   stays under the existing m=4 bound (mean < 0.05). Report the numbers.
2. **No-sort assertion**: add a check (and state it in your report) that neither reconstruction calls
   `argsort`/`sort`/`cumsum`-in-depth-order. `grep -n 'argsort\|\.sort(\|cumsum' spike/arms.py` over the
   reconstruction must be clean (the old path is removed).
3. `spike/tests/test_toy_occlusion.py` MO tests still pass (front red dominates; z-grad nonzero).
4. **arm A unchanged**: `blend_A` path bit-identical (you only touch `_moment_reconstruct`/`blend_MO`).
5. `python -m py_compile spike/arms.py tools/moment_oracle.py`.

You CANNOT run the GPU quality sweep (no GPU here) — that is the user's step below. Do not run it.

## GPU re-validation — the DECISION GATE (the user runs this on the GPU box)

The user runs, on the GPU box, with the sort-free MO:
```bash
python -m spike.m05_spike --scene data/nerf_synthetic/lego  --arms A,B,MO,D \
    --G 2000 --res 64 --iters 600 --seeds 3 --device cuda --out outputs/m05_mo_sortfree_lego
python -m spike.m05_spike --scene data/nerf_synthetic/ficus --arms A,B,MO,D ... --out outputs/m05_mo_sortfree_ficus
```
**Compare to the old (sorted) MO:** ficus 25.40, lego 18.45. Decision:
- If sort-free MO (variant B, the efficient one) **still closes the lego gap** (≈ within seed-std of the
  old MO / clearly > A) → **GO**: proceed to the tt-nn device port (separate hand-off; per-tile `[K,K]`
  for (A) or the moments-GEMM + per-pixel solve for (B), reusing the SM z-force wiring).
- If only variant (A) holds it (and (B) degrades) → the gap needs the `O(G²)`/`O(K²)` pairwise form;
  note the device cost reality (per-tile `K²`, same sizing question as BP).
- If neither holds it without the sort → **MO's win was sort-dependent**; report it plainly (the honest
  outcome: no genuinely sort-free arm closes the gap yet).

Provide a 1-table summary: A / B / MO(softcmp) / MO(mboit) / D holdout PSNR on lego+ficus.

## Scope boundary
- **In scope:** the two sort-free reconstructions in torch + the extended `moment_oracle` + the no-sort
  proof + the exact GPU re-validation commands. **Out of scope:** the tt-nn device port (gated on the
  GPU result above) and touching any other arm.
- Leave a short note in `docs/occlusion/arm-moment.md` pointing here (the §Device there assumed a
  sort-free reconstruction that did not exist; this file builds it).

## Commit
On `feat/mo-sortfree`, commit with a clear message + the `Co-Authored-By: Claude Opus 4.8` /
`Claude-Session:` trailer.

## Final report (data for the orchestrator)
Branch + sha; what changed in `_moment_reconstruct`/`blend_MO`; the no-sort proof (the clean grep);
`moment_oracle` fidelity numbers for (A) and (B) at the chosen τ/m; toy-test + arm-A-unchanged
confirmation; py_compile; and the exact GPU re-validation commands + what numbers to compare against
(ficus 25.40 / lego 18.45) to make the GO / sort-dependent decision.
