# Sort-free occlusion approximations for route-B on Blackhole — design notes

> **STATUS: exploratory, NOT adopted, NOT implemented.** A design discussion captured for later.
> WSR depth-free (arm A) remains the **locked** default (§3); any occlusion/opacity/depth extension
> needs human sign-off (§3 open thesis risk). This file is the *menu of options* to reach for **if**
> the empirical lego occlusion gap (`docs/benchmark-ficus-lego.md`, route-B −3 dB on the solid
> occluder) is not closed by the cheapest lever (arm B, per-gaussian depth weight). Nothing here
> changes the `Q=Φθ` GEMM hot path; every candidate is **sort-free** and **multiplies `o` only**.

## The problem, precisely

Occlusion = per-ray **transmittance** `Tᵢ = Πⱼ in front (1−αⱼ)`, then `C = Σᵢ cᵢ αᵢ Tᵢ`. The two
operations that make this expensive are the **prefix-product** (needs ordering) and the **sort**
that produces the order. Standard 3DGS pays both.

§3 bans the depth sort; the Blackhole ISA confirms *why* it is the wrong workload (see the ISA
read in the project history): the SFPU has **no hardware gather / scatter / data-dependent
indexing**; cross-lane movement is limited to fixed patterns (`SFPTRANSP`, `SFPSHFT2`); `SFPSWAP`
is lane-pairwise only and the docs call sorting networks on it "impractical"; and during any sort
the **matrix engine — the entire 332 TFLOPS — sits idle**. So the design rule is: *re-express
occlusion as something the matrix engine or the min-unit can do.*

Constraints for any candidate lever: **(1) sort-free, (2) multiply `o` only** (leave the
`Q=Φθ` GEMM untouched), **(3) differentiable in `z`** (occlusion learning *is* a depth-gradient —
gaussians must feel a force to move front/back; a purely hard operator gives a.e. zero `z`-grad).

## The 3 fast BH primitives (the axis the ideas are organized on)

- **(a) reductions / moments** — `Σ wᵢ zᵢⁿ` etc.; the power of `z` is lane-wise, the sum is a reduction.
- **(b) dense pairwise GEMM `[G,G]`** — the matrix engine's native food.
- **(c) min / argmin / top-k** — `SFPSWAP` is the *one* cross-element op BH does fast, and its
  argmin mode (`ENABLE_DEST_INDEX`: swap `LReg[0..3]` while carrying the `LReg[4..7]` payload along)
  reduces to the nearest gaussian **and carries its color with it — no gather needed**.

### Key connection — arm B's rank is already a GEMM

The "number of gaussians in front of `i`" (the rank that a sort would produce) has a sort-free soft
form:

```
Qrankᵢ = Σⱼ sigmoid((zᵢ − zⱼ)/τ)  =  (S · 1)ᵢ ,   S = [G,G] soft-comparison matrix
```

= a row-sum of a `[G,G]` matrix = a GEMM. So **arm B is not an isolated trick — it is the entry
point of a whole family** (the `[G,G]` soft-occlusion forms below).

## Catalog of candidates

### Tier 1 — realistic (established graphics, ≈ pure GEMM/reduction)

1. **Weighted Blended OIT** (McGuire–Bavoil 2013). The real-time technique built precisely to avoid
   sorting: accumulate `Σ wᵢcᵢαᵢ` and `Σ wᵢαᵢ` with a depth-decreasing weight `w(z,α)`. **arm B is
   essentially a learnable WBOIT.** Reuse the paper's tuned weight function to *initialize* `g`.
2. **Depth-softmin / Boltzmann weight.** `wᵢ = oᵢ·softmax(−zᵢ/τ)`. `τ→0` = front-takes-all (hard
   occlusion), `τ→∞` = WSR. **One scalar `τ` continuously connects occlusion ↔ WSR**, so the limits
   are interpretable. `exp` is lane-wise, normalization is a reduction, `τ` learnable. Advantage over
   B's monotone `g`: the *limit* is provably correct occlusion.
3. **Moment-based OIT.** Represent `T(z)` by low-order power moments `Σ wᵢzⁿ` (n=0..3/4); reconstruct
   each gaussian's `Tᵢ` from the moments. Moments are pure GEMM; reconstruction is a tiny per-pixel
   4×4 Cholesky (SFPU/RISCV). Captures *where in depth the occluders sit* — strictly more expressive
   than a global monotone `g`. A plausible **superset of arm B**.

### Tier 2 — matched to BH's terrain (uses min + GEMM aggressively)

4. **argmin anchor (front-relative depth) + soft depth-peeling.** BH does argmin fast: take the
   nearest surface exactly via `SFPSWAP`, re-reference depth to the front (`zᵢ − z_min`), weight
   `∝ exp(−(zᵢ−z_min)/τ)`. Kills B's weakness of depending on absolute scene depth. Extension: peel
   top-k via k `SFPSWAP` passes, composite the front k layers with correct soft transmittance and
   WSR the rest — "full sort impossible, but selection is cheap."
5. **Pairwise soft-occlusion GEMM (attention form).** One step past the rank GEMM above:
   ```
   log Tᵢ ≈ − Σⱼ sigmoid((zᵢ − zⱼ)/τ)·βⱼ ,   βⱼ = −log(1−αⱼ)
   wᵢ = oᵢ αᵢ exp(log Tᵢ)
   ```
   Sort → soft compare-matrix `S=[G,G]`; prefix-product → `exp(−sum)`. Recovers *true cumulative
   occlusion* (how much is packed in front), and the shape is a **dense matmul = straight at the
   matrix engine.** Cost `O(G²)` vs B's `O(G)`, but it lives on BH's fastest resource. Cheaper than
   differentiable-sort (NeuralSort/SoftSort) because it only builds the prefix-occlusion, not a full
   permutation matrix. **This is the sort-free realization of arm B's `Qrank` → call it arm B′.**
6. **Scatter-free depth-binning.** Fixed `B` depth bins (data-independent boundaries ⇒ no scatter);
   soft membership `[G,B] = softmax(−(zᵢ−bin)²)`; per-bin opacity-weighted sums via GEMM; front→back
   composite of the small fixed `B` bins is a fixed unroll (not a sort). = **learnable fixed-count
   depth-peeling**; fidelity dialed by `B`. Turns the forbidden scatter into a dense membership GEMM.

### Tier 3 — exotic (but coherent)

7. **Fourier opacity mapping** (Jansen–Bavoil 2010). `T(z)` as a truncated Fourier series; coeffs
   `Σ cos/sin(ωzᵢ)·αᵢ` = pure reduction; evaluate the series at each `zᵢ`. Frequency-domain cousin of
   moment OIT. Ringing is the downside; fully sort-free and GEMM.
8. **Stochastic transparency** (Enderton 2010) — *see deep dive below.* Keep each gaussian with prob
   `αᵢ` (independent coin); the nearest *kept* one wins via argmin. Sample-mean gives **unbiased**
   alpha compositing with **no sort at all** — only argmin, BH's hard-fast primitive.
9. **Neural occlusion head.** Feed sort-free summary stats (a few moments, `z_min`, `Σα`, a few order
   statistics — all reductions) to a small MLP that predicts per-gaussian `Tᵢ` or the final weight.
   Pre-process = reduction, MLP = GEMM. "Learn the occlusion operator from moments."

### Boundary case
**Depth-limited partial bitonic** (selection network for the front k only): a *fixed* op sequence,
so bounded cost — "use the slow path, but only minimally." Sits between #4 and #5.

## Deep dive — #8 stochastic transparency / Gumbel-softmin (candidate **arm E**)

Why it's attractive: the two hard operations vanish *together*.
- prefix-product `Πⱼ(1−αⱼ)` emerges for free from the **independence of independent keep-coins**;
- ordering collapses to a **single argmin**.
With keep-prob `αᵢ`, the expected compositite is exactly
`E[C] = Σᵢ cᵢ·αᵢ·Πⱼ in front(1−αⱼ)` — **unbiased** true alpha compositing. BH's two slow ops
(product chain, sort) become BH's two fast ops (lane-wise multiply, argmin).

`SFPSWAP`'s argmin-with-payload mode is almost purpose-built: it reduces to min-depth **while
carrying the winner's color**, so no post-argmin `color[idx]` gather (BH's forbidden op) is needed.
Lay the per-ray gaussians along the register depth axis and the reduction axis aligns with `SFPSWAP`.

Big upside: **hard occlusion edges**. The softmin (#2) is angularly smooth and cannot produce a
sharp visible↔hidden transition (the same weakness that sank arm C); a depth test solves order
exactly, so occlusion edges stand up.

The catch — **not differentiable as written** (Bernoulli + argmin both block gradients), and 3DGS
needs grads on `o, c, z`. Two routes:
1. **Straight-through:** forward = stochastic-hard (correct occlusion, argmin on BH), backward via the
   softmin (#2). Light to implement; biased but STE-style tricks (VQ-VAE etc.) usually train.
2. **Gumbel-softmin (the rigorous version, the publishable shape):** per-ray categorical
   (prob `∝ oᵢ·softmin(zᵢ/τ)`) reparameterized with Gumbel noise; **anneal `τ→0`** during training to
   converge to true occlusion. Gumbel `= −log(−log U)` is a lane-wise add — cheap on BH.
**Non-negotiable:** the `z`-gradient needs the soft backward — pure-hard argmin is a.e. zero in `z`,
so gaussians get no force to move front/back, and occlusion learning *is* `z`-gradient.

Honest costs: **variance ⇒ S≈8–16 samples/pixel** for a meaningful per-image loss (the earlier claim
that "SGD noise covers the averaging" was overstated — corrected). argmin is cheap so the S passes are
realistic; **stratified keep** (a fixed S-bit coverage mask set by `αᵢ`, randomize only the offset)
cuts variance and the mask is structurally fixed = scatter-free. Plus STE forward/backward bias and
`S`-layer memory.

**Unifying view:** #8 is the **Monte-Carlo sparse sample of #2**. #2 (softmin) is biased-by-`τ` but
low-variance / dense / deterministic; #8 is unbiased but high-variance / sparse / argmin-based. The
choice axis is **bias(`τ`) vs variance(sampling)**. BH bonus: #8 leans on `SFPSWAP`, not the matrix
engine, so it does **not** contend for hardware occupancy with the GEMM-path ideas (#5/#6).

## Ranking & how they'd enter the harness

- Practical first picks: **2 (softmin) → 3 (moment) → 5 (pairwise GEMM)**. #5 = the sort-free
  realization of arm B (**arm B′**) and worth integrating/measuring directly. #3/#5/#6 keep the
  matrix engine busy (best fit to the perf thesis).
- Proposed M0.5-style arms when/if pursued: **arm B′** (sort-free `Qrank` GEMM), **arm E**
  (Gumbel-softmin). Measure vs softmin: (i) sharper occlusion edge on held-out, (ii) `S` vs
  variance/quality, (iii) `τ`-anneal effect. Lined up with arm B in `docs/benchmark-ficus-lego.md`
  this fills the triplet: **soft single (B) / soft limit (softmin) / stochastic-hard (Gumbel)**.

## Open sizing questions (deferred)

- #5 `O(G²)` `[G,G]` GEMM at res128/G2000 — is the `[2000,2000]` per-tile compare-matrix realistic in
  L1, or does it need the same binning cap `K` as the forward? (`O(K²)` per tile is fine.)
- #8 `O(S·G)` — `S` vs the existing dense `[P,G]` cost; cheaper than #5 but adds the sample axis.
- For both: where the depth `z` comes from on-device (project gives depth host-side per §3 today).

## Bottom line

A menu, not a decision. arm A (depth-free WSR) stays locked. The cheap empirical test (arm B on
lego, currently running) comes first; **these notes are what to try if B alone does not close the
−3 dB**, in roughly the order 2 → 3 → 5, with #8 (arm E) as the high-risk/high-reward
"stochastic-hard" option that uniquely produces crisp occlusion edges and uniquely avoids the GEMM
units. Any adoption is §3-gated.
