# tt-nn port playbook — taking the torch occlusion arms to Blackhole

> **Workflow rule (the answer to "how do we proceed?"):** quality is decided on **GPU**, feasibility +
> perf on **BH**. The torch arms run unchanged on a CUDA box; the tt-nn port is expensive — so
> **GPU-triage first, port only the winners.** CPU is not a dev surface for any scale run.

## Where we are

- **Done** (`feat/arm-integration`, 17/17 toy-occlusion tests): host torch arms `blend_SM / BP / MO / E`
  in `spike/arms.py`, wired into `render.py` + `m05_spike.py`, each with an analytic/forward oracle
  (`tools/{softmin,bprime,moment,gumbel}_oracle.py`). **Math is proven; quality-at-scale is NOT.**
- **Not done**: the device (ttnn) kernels in the traced path (`sweep_resident.py` / `resident_traced.py`).
- **Key fact**: `m05_spike.py` is pure torch (`--device cuda` auto-uses a GPU); it does **not** touch the
  Blackhole path. So GPU answers *does the arm help*, BH answers *does the kernel fit + how fast*.

## The two-track funnel

### Track 1 — GPU quality triage (now, zero new code)
Push `feat/arm-integration` to the GPU box and run the host arms at the spike scale:
```bash
python -m spike.m05_spike --scene data/nerf_synthetic/ficus \
    --arms A,B,SM,BP,MO,E,D --G 2000 --res 64 --iters 600 --seeds 3 --device cuda \
    --out outputs/m05_occlusion_ficus
# repeat on the OCCLUDER scene (the -3 dB gap lives here):
python -m spike.m05_spike --scene data/nerf_synthetic/lego --arms A,B,SM,BP,MO,E,D ... \
    --out outputs/m05_occlusion_lego
```
**Output = the go/no-go:** per-arm holdout PSNR vs A (and vs B). The arms that close the lego
solid-occluder gap without an arm-A regression are the **port candidates**. `D` is the sorted ceiling;
how close each arm gets to `D` on lego is the headline. (res 64 first for a fast signal; res 128 to
confirm.)

### Track 2 — tt-nn port (BH, winners only)
Port **only** the Track-1 winners into the device traced path. The device kernel is validated against
the **torch oracle that already exists** (no new ground truth needed).

**Port SM now, in parallel with triage** — it is low-risk *and* foundational: it lands the shared
per-gaussian `keo` fold-in, the `gkeo` backward hook, and the **z-force → means** wiring that BP and E
both reuse (`02-device-map.md` §2/§4). So SM pays off regardless of which arm wins.

## Universal device-port loop (per arm)

1. **Gate**: arm is a Track-1 winner, or is SM (foundational).
2. **Implement** the device kernel at the insertion point from `02-device-map.md` and the arm's
   `§Device` (in `arm-<x>.md`): per-gaussian arms (SM/BP) fold into `keo` in `geom_fwd` + back-prop off
   `gkeo` in `geom_bwd`; per-pixel arms (MO) add buffers+GEMMs in `rend_fwd`/`rend_bwd` (re-capture);
   E is a separate `SFPSWAP` argmin path (prototype standalone first).
3. **Match the oracle**: device output (fwd + grads) must match `tools/<arm>_oracle.py` on small G
   (`G ≤ 256`) on BH to a tight `rel`. The oracle is the contract.
4. **Match GPU quality**: a short device train must reach the same holdout PSNR the GPU host arm got
   (same arm, same scene) within tolerance — proves the kernel is faithful, not just gradient-correct.
5. **Measure BH**: it/s, and answer the arm's **feasibility question** (below). Report L1 fit / crossover.
6. **Wire a flag**: add `--arm-<x>` to `sweep_resident.py` mirroring `--depth-weight`; default off.

## Per-arm port sheet

| Arm | Host oracle | Device insertion (`02-device-map.md`) | BH feasibility question |
|---|---|---|---|
| **SM** | `softmin_oracle.py` | `geom_fwd`: `rho=exp(-(z-zref)/tau)` → `keo`; `geom_bwd` off `gkeo`; host-Adam `tau`; **z-force→means** | none — cheapest; do first. |
| **BP** | `bprime_oracle.py` | (A) global `[G,G]` in `geom_fwd` → `keo`; (B) per-tile `[K,K]` in `rend_fwd` | **does `[G,G]` fit L1 at G=2000?** if not, ship per-tile (B). |
| **MO** | `moment_oracle.py` | per-pixel moments GEMM + `(m/2+1)²` Cholesky in `rend_fwd`/`rend_bwd` (re-capture) | is the per-pixel reconstruction viable on SFPU? recon backward closed-form? |
| **E** | `gumbel_oracle.py` | **new** `SFPSWAP` argmin-with-payload render; STE backward reuses **SM**'s soft path | prototype argmin kernel standalone; does it beat GEMM-render cost? (may stay host-only) |

## Why this order / this split

- **SM → BP → MO → E** by device difficulty (`README.md`). SM lands the shared wiring; BP reuses the
  `keo` fold + adds the matrix-engine GEMM (best perf-thesis fit); MO is per-pixel (independent path);
  E is the research kernel (argmin, no GEMM) — host-validate, then gate device on a microbench.
- **GPU decides quality, BH decides feasibility.** Never port an arm the GPU triage shows doesn't help.
  Never conclude "feasible on BH" from a GPU run — the L1/SFPU questions are BH-only.

## Going forward (the standing dev rule)

Every quality lever from here: **prototype + validate in torch on GPU** (it's the oracle and the fast
quality signal), **then port the winner to tt-nn** for BH feasibility/perf, validated against the torch
oracle. CPU is for unit tests only. §3 adoption sign-off still gates any default change.
