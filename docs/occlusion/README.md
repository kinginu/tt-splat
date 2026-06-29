# Sort-free occlusion arms — hand-off pack

Implementation hand-offs for the **sort-free occlusion** levers (route-B on Blackhole). Each per-arm
file is a **self-contained brief for one Sonnet agent**: it implements the arm's **host reference**
*and* its **device (traced) port**. Authored by Opus from the design notes + a read of the live render
path; the agents implement from these.

> **§3 gating (read first).** arm A (depth-free WSR) is the **locked default**. Every arm here is
> **added alongside** A for measurement — no default changes, nothing removed, the `Q=Φθ` GEMM
> untouched. **Adoption (and any device landing) is human-sign-off-gated.** These hand-offs produce
> the *evidence* (host arm + benchmark) to decide with; see `00-design-notes.md` for the full menu and
> the locked-arm rationale.

## Read order

| File | What |
|---|---|
| `00-design-notes.md` | The exploratory menu + §3 rationale (committed reference). |
| `01-arm-authoring-contract.md` | **Shared host scaffolding** — the 6-step "add an arm" checklist (`arms.py`, `model.py`, `render.py`, `m05_spike.py`, toy tests), constraints C1–C3, acceptance. Every hand-off references it. |
| `02-device-map.md` | **Device interfaces** — render fwd/bwd, the `keo`/`gkeo` fold-in, where depth `z` lives, the traced-pipeline insertion points, host-Adam scalar pattern. Every device port references it. |

## The four arms (already-shipped SZ/RV excluded)

| Hand-off | Arm | Cand. | Device difficulty | One-line |
|---|---|---|---|---|
| `arm-softmin.md` | `SM` | #2 | **LOW** (arm-B clone, folds into `keo`) | Boltzmann front-weight `exp(−(z−z_ref)/τ)`; τ→0 = front-takes-all, τ→∞ = arm A. |
| `arm-bprime.md` | `BP` | #5 | **MEDIUM** (`[G,G]`/`[K,K]` GEMM, folds into `keo`) | Pairwise soft-occlusion `ρ=exp(−Sβ)`; sort-free realization of arm B; **best perf-thesis fit**. |
| `arm-moment.md` | `MO` | #3 | **MED-HIGH** (per-pixel, modifies `rend_fwd`) | Moment-based OIT; reconstruct `T(z)` from power moments; models the whole occluder profile. |
| `arm-gumbel.md` | `E` | #8 | **HIGH / research** (SFPSWAP argmin, new render path) | Stochastic transparency; unbiased, **crisp occlusion edges**; avoids the matrix engine. |

## Recommended order (per `00-design-notes.md`: 2 → 3 → 5, E as high-risk)

1. **`SM` first** — lowest risk; it establishes the shared device wiring (per-gaussian `keo` fold,
   the `gkeo` backward hook, and the **z-force → means** fix that softmin/B/B′ all need). BP and E reuse it.
2. **`BP`** — reuses SM's fold-in; adds the GEMM. The strongest perf-thesis candidate.
3. **`MO`** — per-pixel; the most expressive; independent of the `keo` path.
4. **`E`** — host fully, device as a **gated standalone prototype** (argmin microbench first).

`MO` and `E` are independent of `SM`/`BP` on the device side and can proceed in parallel once `SM`
lands the shared host pattern.

## How these get executed (for whoever spawns the agents)

- **One Sonnet agent per hand-off file**, each on its **own branch** (`feat/arm-softmin`,
  `feat/arm-bprime`, `feat/arm-moment`, `feat/arm-gumbel`).
- **Shared-file collisions:** all four make *additive* edits to the same five host files (`arms.py`,
  `model.py`, `render.py`, `m05_spike.py`, `test_toy_occlusion.py`). **Integrate sequentially**
  (land `SM`, then rebase/merge `BP`, then `MO`, then `E`) — the additions don't overlap in intent but
  land at the same file tails, so expect trivial append-conflicts that resolve by *keeping both*. New
  device code lands in **new per-arm files** (`tools/<arm>_oracle.py`) or **fenced minimal blocks** of
  `resident_traced.py`/`sweep_resident.py` to limit cross-arm device collisions.
- **Order matters:** land `SM` first so the shared z-force/`keo` device wiring exists before `BP`/`E`
  build on it.

## Acceptance signal (all arms)

- **Correctness:** `spike/tests/test_toy_occlusion.py` (the 2-gaussian near-red/far-blue probe) — the
  arm must occlude (front dominates) and be `z`-differentiable; plus the per-arm oracle
  (`tools/<arm>_oracle.py`).
- **Quality:** `python -m spike.m05_spike --arms A,B,<arm>,D --G 2000 --res 64 --iters 600 --seeds 3` —
  does `<arm>` close (some of) the **lego solid-occluder −3 dB gap** vs arm A, without an arm-A
  regression, and how does it compare to arm B? That delta is the go/no-go evidence (§3 sign-off).
- **No arm-A regression:** arm A must stay bit-identical (you only add code paths).

## Status

Hand-offs only — **nothing implemented yet**. Branch `docs/occlusion-handoffs`. Spawn the four agents
when ready (recommend `SM` first, then the rest).
