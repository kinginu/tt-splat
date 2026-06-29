#!/usr/bin/env bash
# Axis-1 (quality) benchmark on the GPU box: standard 3DGS (gsplat) vs route-B (poly+WSR, arm A)
# on ficus + lego, matched conditions (res128/G2000/3000it/seeds3).
# Sequential (avoids 3090 OOM); each job writes its own JSON + the per-seed wall time is in the JSON.
set -u
cd "$(dirname "$0")/.."

RES=${RES:-128}; G=${G:-2000}; ITERS=${ITERS:-3000}; SEEDS=${SEEDS:-3}
LOG=outputs/bench_axis1.log
mkdir -p outputs
echo "=== axis-1 bench start $(date -u +%FT%TZ) | res=$RES G=$G iters=$ITERS seeds=$SEEDS ===" | tee "$LOG"

run() {  # $1=label $2=svc $3...=cmd
  local label=$1; shift; local svc=$1; shift
  echo "--- [$label] start $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  local t0=$SECONDS
  docker compose run --rm "$svc" "$@" 2>&1 | tee -a "$LOG"
  echo "--- [$label] done $(date -u +%FT%TZ) wall=$((SECONDS-t0))s ---" | tee -a "$LOG"
}

for scene in ficus lego; do
  run "baseline_$scene" baseline python tools/baseline_gsplat.py \
    --scene data/nerf_synthetic/$scene --res $RES --G $G --iters $ITERS --seeds $SEEDS \
    --out outputs/bench_baseline_$scene
  run "routeB_$scene" cuda python -m spike.m05_spike \
    --scene data/nerf_synthetic/$scene --res $RES --G $G --iters $ITERS --seeds $SEEDS --arms A \
    --out outputs/bench_routeB_$scene
done

echo "=== axis-1 bench end $(date -u +%FT%TZ) ===" | tee -a "$LOG"
