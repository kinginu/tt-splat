# Hand-off — arm `E` (stochastic transparency / Gumbel-softmin) · candidate #8

> **One Sonnet agent owns this file.** Read `01-arm-authoring-contract.md` + `02-device-map.md` first.
> §3-gated; ADD `E` alongside arm A. Branch: `feat/arm-gumbel`.
>
> **Device difficulty: HIGH (research).** arm E is **argmin-based, NOT GEMM** — it does not fold into
> `keo` and does not reuse the `rend_fwd` matmul WSR. Its device form needs a **new `SFPSWAP`
> argmin render path** outside the existing traced pipeline. Land the **host** layer fully and
> measured; treat the device kernel as a **prototyped sketch** (benchmark standalone) — full device
> integration may remain gated until the argmin kernel is proven. Unique payoff: **crisp occlusion
> edges** (a depth test, not a smooth gate) and it **avoids the matrix engine** (no contention with
> B′/MO).

## What it is

Keep each gaussian with probability `αᵢ` (independent coin); the nearest **kept** one wins via argmin;
average over `S` samples. The sample-mean is **unbiased** true alpha compositing:

```
E[C] = Σᵢ cᵢ · αᵢ · Π_{j in front}(1−αⱼ)
```

The two BH-slow ops vanish *together*: the prefix-product `Π(1−αⱼ)` emerges for free from the
**independence of keep-coins**, and ordering collapses to a **single argmin**. BH's two fast ops
(lane-wise multiply, `SFPSWAP` argmin) replace them. `SFPSWAP`'s argmin-with-payload (`ENABLE_DEST_INDEX`)
reduces to min-depth **while carrying the winner's color** — no post-argmin gather.

**The catch (C3):** Bernoulli + argmin are non-differentiable; pure-hard argmin gives a.e.-zero
`z`-grad, but occlusion learning *is* a `z`-force. So the forward is hard (correct occlusion) and the
**backward goes through a soft surrogate** (the softmin, candidate #2). Two routes:

1. **Straight-through (STE)** — forward = stochastic-hard, backward = softmin grad. Light; biased but
   trains (VQ-VAE-style). **Ship this first.**
2. **Gumbel-softmin (rigorous)** — per-ray categorical, logits `∝ log αᵢ − zᵢ/τ`, reparameterised with
   Gumbel noise `g = −log(−log U)` (a lane-wise add), **anneal `τ→0`** during training to converge to
   true occlusion. The publishable shape.

**Unifying view:** arm E is the **Monte-Carlo sparse sample of softmin (`SM`)** — softmin is
biased-by-`τ`/low-variance/dense; arm E is unbiased/high-variance/argmin. Bench them as a pair.

## Host implementation (per `01-arm-authoring-contract.md`)

> Per-pixel, sampled, **OIT-style composite** (not `_wsr`). Pass a `torch.Generator` for determinism;
> default `S=8` samples. Keep `z` attached through the **soft** path only.

```python
def blend_E(w_geo, opacity_raw, depth, e_tau, color, w_b, c_b, S=8, gen=None, route="ste"):
    """Stochastic transparency / Gumbel-softmin (candidate #8, 'arm E'). Forward = stochastic-hard
    (Bernoulli keep prob alpha_pg, nearest-kept argmin, mean of S samples) = unbiased alpha
    compositing with crisp occlusion edges. Backward via the softmin surrogate (STE) or Gumbel-softmin
    (route='gumbel', anneal tau). z attached through the soft path (C3). argmin-based -> avoids GEMM."""
    o = torch.sigmoid(opacity_raw)
    alpha = (o[None, :] * w_geo).clamp(1e-6, 1.0 - 1e-4)          # [P,G] keep prob
    hard = _stochastic_hard(alpha, depth, color, c_b, S, gen)    # [P,3] no-grad, unbiased
    soft = _soft_surrogate(alpha, depth, color, c_b, e_tau, route, gen)  # [P,3] differentiable, z attached
    return soft + (hard - soft).detach()                         # STE: value=hard, grad=soft
```

- **`_stochastic_hard(alpha, z, color, c_b, S, gen)`**: for each of `S` samples draw `keep ~
  Bernoulli(alpha)` `[P,G]`; among kept, pick `argmin z` per pixel (mask non-kept to `+inf` depth);
  gather winner color (background where none kept); mean over `S`. No grad (wrap in `no_grad`). Use
  **stratified keep** for variance: a fixed `S`-bit coverage pattern set by `alpha` with only the
  offset randomized (also makes the device mask structurally fixed = scatter-free).
- **`_soft_surrogate`**: `route="ste"` → softmin composite (reuse arm `SM`'s `ρ=exp(−(z−z_ref)/τ)`,
  composite OIT-over). `route="gumbel"` → per-pixel Gumbel-softmax over logits `log α − z/τ` giving a
  soft one-hot `[P,G]`, composite `Σ soft·c`. Both differentiable with `z` attached.

**`spike/model.py`**: `self.e_tau = nn.Parameter(torch.tensor(1.0, …))`; register under `lr["depth"]`.
For Gumbel, **anneal `τ`** via a `fit(..., tau_anneal=...)` schedule (see contract §4) — do not hard-code.

**`spike/render.py`**: `ARMS += ("E",)`; `elif arm == "E": img = arms.blend_E(w_geo, model.opacity_raw, depth, model.e_tau, color, w_b, c_b, S=args... )`. Thread `S`/`gen` from the caller; default `S=8`.

**`spike/m05_spike.py`**: add `"E"` to `CANDIDATE_ARMS`. (Expect higher variance run-to-run; report it.)

**`spike/tests/test_toy_occlusion.py`** (seed the generator; with `o=0.9` the near gaussian is kept
~90% of samples and wins argmin ⇒ red dominates in expectation):
```python
def test_gumbel_occludes():
    s = _setup(); g = torch.Generator().manual_seed(0)
    C = arms.blend_E(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.5), s["color"], s["w_b"], s["c_b"], S=256, gen=g)[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C      # front dominates in expectation
    assert C[0].item() > 0.75, C                                  # high-variance ⇒ looser, big S

def test_gumbel_depth_order():
    s = _setup(); g = torch.Generator().manual_seed(0)
    C = arms.blend_E(s["w_geo"], s["opacity_raw"], _f([5.0,2.0]), _f(0.5), s["color"], s["w_b"], s["c_b"], S=256, gen=g)[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C

def test_gumbel_zgrad():    # grad flows through the SOFT surrogate
    s = _setup(); z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_E(s["w_geo"], s["opacity_raw"], z, _f(0.8), s["color"], s["w_b"], s["c_b"], S=8)
    (g,) = torch.autograd.grad(C.sum(), z); assert g.abs().sum() > 0
```

## Validation oracle

`tools/gumbel_oracle.py` — **unbiasedness**: for random `(z, α)`, the stochastic-hard estimator mean
over large `S` must converge to the exact sorted `Σ cᵢ αᵢ Π_{j in front}(1−αⱼ)`; report
`|mean_S − exact|` vs `S ∈ {1,8,16,64,256}` (shows the `S`≈8–16 sweet spot and the variance law). Also
check the STE gradient sign matches the softmin gradient on the toy (the surrogate must push the far
gaussian back).

## Device implementation (per `02-device-map.md`) — SKETCH, prototype standalone

arm E does **not** reuse the GEMM WSR render. It needs a new **argmin render**:
- Lay each tile's `K` candidate gaussians along the register depth axis; per sample, set a keep mask
  from `α` (stratified: fixed `S`-bit pattern, randomized offset — scatter-free), set non-kept depth to
  `+inf`, and `SFPSWAP` argmin-with-payload (`ENABLE_DEST_INDEX`, swap `LReg[0..3]` carrying the
  `LReg[4..7]` color payload) → nearest kept's color, **no gather**. Accumulate the mean over `S`.
- Backward = the **soft surrogate on device** (softmin `ρ` fold, the `SM` device path) via STE — i.e.,
  reuse arm `SM`'s `geom_fwd`/`geom_bwd` ops for the gradient, and the argmin kernel only for the
  forward value. So the **z-force/τ-grad device wiring is shared with `arm-softmin.md`**.
- Gumbel: add `g = −log(−log U)` (lane-wise) to the logits before the argmin/softmax — cheap.
- **It avoids the matrix engine** (pure SFPU/SFPSWAP), so it does not contend with B′/MO for GEMM
  occupancy — a scheduling upside.

**Realistic plan:** prototype the `SFPSWAP` argmin-with-payload reduction as a **standalone device
kernel + microbench** (argmin over `K`, `S` samples, vs the GEMM render cost) BEFORE wiring it into a
trainer. The existing traced pipeline is GEMM-shaped; integrating an argmin forward + STE-soft backward
is the riskiest device task here — keep it isolated, and gate full integration on the microbench +
the host result justifying it. Host-only arm E is a legitimate stopping point if the kernel doesn't pay.

## Sizing / cost
- Host/device forward `O(S·G)` (or `O(S·K)` per tile) — `S≈8–16`. argmin cheap; memory `+S` mask layers.
- Stratified keep cuts variance and fixes the mask structure (scatter-free). STE backward = the softmin
  cost (negligible). Higher run-to-run variance than SM/BP/MO — report it.

## Acceptance
- Host: `test_toy_occlusion` (+3) passes; `gumbel_oracle.py` shows convergence to exact sorted compositing
  and the variance-vs-`S` curve; arm A bit-identical; `m05_spike --arms A,B,SM,E,D` includes `E`.
- Device: a standalone `SFPSWAP` argmin-with-payload microbench (correctness vs host + cost vs GEMM
  render). Full `sweep_resident.py --arm-gumbel` integration is **stretch / gated** on the microbench.
- Result note: **crisp-edge** comparison — does arm E produce sharper visible↔hidden transitions than
  softmin on lego held-out (the unique selling point); `S` vs quality; `τ`-anneal effect; the
  bias(`τ`, softmin) vs variance(`S`, E) trade.

## Risks / open
- **Highest-risk device port** — argmin render is outside the GEMM harness; may stay host-only.
- STE bias (forward-hard / backward-soft mismatch) — monitor training stability; Gumbel route is the
  rigorous fallback if STE stalls.
- Variance: `S` too small ⇒ noisy loss; too large ⇒ memory/compute. The oracle's `S`-curve sets it.
- Shares the softmin device z-force/τ wiring — coordinate with `arm-softmin.md` (land SM first).
