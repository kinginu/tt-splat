# Arm-authoring contract (host reference) — shared by every occlusion hand-off

Every per-arm hand-off in this folder (`arm-softmin.md`, `arm-moment.md`, `arm-bprime.md`,
`arm-gumbel.md`) implements **two layers**:

1. **Host reference** (this contract) — a torch arm in `spike/`, the oracle + measurable baseline.
2. **Device (traced) implementation** — the Blackhole port (each hand-off's own §Device section,
   keyed off the device map in `02-device-map.md`).

This file is the **host-side checklist**, identical for all four arms. Read it once; each hand-off
then only gives you the arm-specific math, the new params, and the device port.

---

## Non-negotiable constraints (from §3 / `00-design-notes.md`)

A new arm is **only** allowed to:

- **C1 — sort-free.** No `argsort`/`sort`/data-dependent gather on the hot path. (`render_D` is the
  *only* sorted arm; it is the reference ceiling, never a building block.)
- **C2 — multiply `o` only.** Build a per-`(pixel,gaussian)` or per-gaussian weight and fold it into
  the WSR weight column `W = o · w_geo · (…)`. **Do NOT touch the `Q = Φθ` geometry GEMM**
  (`spike/forward.py`), the poly-splat `w_geo`, or `geometry.project_ewa`.
- **C3 — differentiable in `z`.** Occlusion learning *is* a depth gradient: gaussians must feel a
  force to move front/back. A purely hard operator (a.e. zero `z`-grad) is rejected. Your toy test
  must prove a non-zero, correctly-signed `dL/dz`.

arm A (`blend_A`, depth-free WSR) stays the **locked default**. Your arm is added *alongside* it for
measurement; **adoption is human-sign-off-gated** — do not change any default, do not remove arm A,
do not edit `render_D`.

---

## The host render contract (what every arm shares)

`spike/render.py::render(model, cam, arm, …)` computes, for the chosen arm:

```
mu2d, conic_abc, depth, keep = geometry.project_ewa(…)   # depth[G] = per-gaussian camera z
Q     = forward.quad_form(px, py, mu2d, conic_abc)        # [P,G]  -- DO NOT TOUCH
w_geo = forward.poly_splat_wgeo(Q, k, keep)               # [P,G]  -- DO NOT TOUCH
color = forward.color_from_dc(model.color_dc)             # [G,3]
w_b   = softplus(model.w_b_raw);  c_b = model.c_b
img   = arms.blend_<ARM>(w_geo, model.opacity_raw, depth, <arm params>, color, w_b, c_b)
return img.reshape(cam.H, cam.W, 3)
```

The shared WSR primitive every arm composites through (in `spike/arms.py`):

```python
def _wsr(W, color, w_b, c_b):
    """Weighted Sum Rendering with learnable background. W[P,G], color[G,3] -> [P,3]."""
    num = W @ color + w_b * c_b[None, :]
    den = W.sum(dim=1, keepdim=True) + w_b
    return num / den
```

Your arm's job is to build `W[P,G]` (or an equivalent composite, like `blend_RV`) using a sort-free,
`z`-differentiable occlusion factor, and call `_wsr` (or composite explicitly). Reference the two
already-shipped sort-free occlusion arms as templates:

- **`blend_B`** — per-gaussian global front-bias `rho = σ(β(τ−z))`, `W = o·w_geo·rho`. (per-gaussian)
- **`blend_SZ`** — per-pixel `zstar = Σwz/Σw`, gate `h = σ(−β(z−zstar))`, `W = o·w_geo·h`. (per-pixel)
- **`blend_RV`** — revealage `R = exp(Σ log(1−a))`, composite `C_avg·(1−R) + c_b·R`. (custom composite)

---

## Host checklist (do all 6)

### 1. `spike/arms.py` — add `blend_<ARM>(…)`
Match the existing docstring style (state the math, the per-pixel/per-gaussian nature, why it is
sort-free and `z`-differentiable, and the BH-primitive it maps to). Keep `depth` **detached only if**
your arm uses `z` purely as a visibility gate AND you still provide a `z`-gradient path elsewhere —
**most occlusion arms must keep `z` attached** (C3). `blend_SZ` detaches `z` because its gate is a
steering signal, but it is the exception; default to **attached `z`**.

### 2. `spike/model.py` — add learnable scalar param(s) + register for Adam
Add the arm's scalar(s) next to the existing ones (`depth_beta`, `depth_tau`, `softz_beta`):

```python
self.<arm>_tau = nn.Parameter(torch.tensor(<init>, dtype=dtype, device=device))
```

and register in `param_groups` under the occlusion LR key (`lr["depth"]`, currently `1e-2`):

```python
{"params": [self.<arm>_tau], "lr": lr["depth"]},
```

Init values matter — pick so the arm starts near the WSR/arm-A behaviour (so it is not advantaged or
destabilised at init); each hand-off gives its recommended init and the limit it anneals toward.

### 3. `spike/render.py` — wire the dispatch
- Add the arm letter to `ARMS = ("A", "B", "C0", "C", "SZ", "RV", "D")`.
- Add an `elif arm == "<ARM>":` branch passing the model fields your `blend_<ARM>` needs (mirror the
  `B`/`SZ` branches). `depth` is already in scope.

### 4. `spike/train.py` — nothing usually needed
`fit()` and `eval_psnr()` are arm-agnostic (they take `arm` and call `render`). `DEFAULT_LR["depth"]`
already exists. Only add an LR key if your arm needs a *separate* LR (state it in the hand-off).
If your arm anneals a temperature `τ` over training (Gumbel), add the anneal schedule **as an
argument to `fit`** (e.g. `tau_anneal=None`) rather than hard-coding it, so arm A is unaffected.

### 5. `spike/m05_spike.py` — register as a measured candidate
Add the arm letter to `CANDIDATE_ARMS` (currently `("A", "B", "C", "SZ", "RV")`) so it appears in the
overfit sweep, and confirm it is reachable via `--arms`. This is the **benchmark harness**:

```
python -m spike.m05_spike --res 64 --G 2000 --iters 600 --arms A,B,<ARM>,D --seeds 3
```

reports per-arm train/holdout PSNR and picks the best CANDIDATE_ARM. The **acceptance signal** is:
your arm closes (some of) the **lego occlusion −3 dB gap** relative to arm A, without an arm-A
regression, and beats or matches arm B on the solid-occluder views.

### 6. `spike/tests/test_toy_occlusion.py` — the load-bearing correctness probe
The two-gaussian probe (near=red @ z=2, far=blue @ z=5, same screen pos, equal footprint) is the
occlusion oracle. arm A averages → purple (fail-to-occlude); arm D/SZ → red (occludes). Add:

```python
def test_<arm>_occludes():
    s = _setup()
    C = arms.blend_<ARM>(s["w_geo"], s["opacity_raw"], s["depth"], <params>, s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C     # front dominates
    assert C[0].item() > 0.85 and C[2].item() < 0.15, C          # front-dominated like a sorted blend

def test_<arm>_depth_order():
    s = _setup(); depth_swapped = _f([5.0, 2.0])                 # blue now in front
    C = arms.blend_<ARM>(s["w_geo"], s["opacity_raw"], depth_swapped, <params>, s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C

def test_<arm>_zgrad():   # C3: occlusion must produce a z-force
    s = _setup(); z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_<ARM>(s["w_geo"], s["opacity_raw"], z, <params>, s["color"], s["w_b"], s["c_b"])
    (g,) = torch.autograd.grad(C.sum(), z)
    assert g.abs().sum() > 0, g                                  # non-zero, and sign-checked per hand-off
```

(`DT = torch.float64` in this test; keep everything float64 there.)

---

## Acceptance / "done" for the host layer

- `python -m spike.tests.test_toy_occlusion` passes incl. your three new tests (run via the repo's
  test bootstrap; these tests import `_bootstrap`).
- `python -m spike.m05_spike --arms A,B,<ARM>,D --G 2000 --res 64 --iters 600 --seeds 3` runs and
  emits a per-arm table including `<ARM>`.
- No change to arm A numbers (run A alone before/after — must be bit-identical; you only *added* code
  paths).
- A one-paragraph result note: does `<ARM>` move holdout PSNR toward arm D on the occlusion probe /
  lego solid-occluder views, and how does it compare to arm B?

---

## Device layer (per-hand-off §Device)

Each hand-off then ports the validated host arm to the Blackhole traced path. The shared device facts
(render fwd/bwd interfaces, the `keo = keep·o·rho` fold-in that arm B uses, where depth `z` lives,
the traced-pipeline insertion point, host-Adam scalar update) are in **`02-device-map.md`**. The
host arm is the **bit-level oracle** for the device port — every device kernel must match the host
arm to a tight `rel` tolerance via a `tools/<arm>_oracle.py` (mirror `tools/m8_depthweight_oracle.py`,
which validates arm B's analytic `β/τ` backward against autograd through `blend_B`).

---

## Shared-file / parallelism note (for whoever spawns the agents)

All four arms edit the **same five shared files** (`arms.py`, `model.py`, `render.py`, `m05_spike.py`,
`test_toy_occlusion.py`) with **purely additive** changes (new function, new `elif`, new param, new
`CANDIDATE_ARMS` entry, new test fns). Run the four agents on **separate branches**, then integrate
**sequentially** (rebase/merge one arm at a time) — the additions are non-overlapping in intent but
land at the same file tails, so expect trivial append-conflicts that resolve by keeping both. Device
work lands in **new per-arm files** (`tools/<arm>_oracle.py`, and the device kernel in the arm's own
module or a clearly-fenced block of `tools/resident_traced.py` / `tools/sweep_resident.py`) — keep
device edits to the resident/sweep files fenced and minimal to limit cross-arm collisions.
