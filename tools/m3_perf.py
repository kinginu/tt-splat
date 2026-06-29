"""Perf BASELINE on real Blackhole silicon -- the first real timing for the project
(ttsim was functional-only, so this number could not exist until silicon).

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m3_perf.py

Honest async timing: warmup (JIT-compile + program-cache fill) -> synchronize_device ->
N reps -> synchronize_device. Measures:
  (1) square bf16 GEMM TFLOPS sweep      -> matrix-engine utilization vs the 332 TFLOPS peak;
  (2) the fat-shallow Phi.theta GEMM     -> the K=6 tile-padding loss the hardware predicts;
  (3) the full (B) forward hot path      -> Phi.theta -> poly -> WSR latency + effective GEMM TFLOPS.

This is a NAIVE baseline: un-fused (separate ttnn ops), un-sharded (whole-tensor, ttnn picks the
grid), host-dispatched, DENSE (un-binned P x G). It is the number the optimized path (fused kernels,
120-core sharding, binning) must beat. It is NOT a perf/$ claim yet.

ttnn API per the earlier tools: open_device, from_torch(TILE_LAYOUT,bfloat16), matmul,
mul/rsub/relu/square/add/div, synchronize_device, to_torch, close_device.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ttnn

PEAK_BF16 = 332.0  # TFLOPS, p150a datasheet
K = 4.0


def up(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def _dealloc(t):
    try:
        t.deallocate()
    except Exception:  # noqa: BLE001
        try:
            ttnn.deallocate(t)
        except Exception:  # noqa: BLE001
            pass


def timed(fn, dev, warmup=3, reps=30):
    """Seconds per call. fn() returns its device output tensor; we free it each iter."""
    for _ in range(warmup):
        _dealloc(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        _dealloc(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps


def gemm_row(dev, M, Kd, N, reps=30):
    a, b = up(torch.randn(M, Kd), dev), up(torch.randn(Kd, N), dev)
    spc = timed(lambda: ttnn.matmul(a, b), dev, reps=reps)
    _dealloc(a)
    _dealloc(b)
    return spc, 2.0 * M * Kd * N / spc / 1e12


def forward_row(dev, P, G, reps=20):
    Phi, thT = up(torch.randn(P, 6), dev), up(torch.randn(6, G), dev)
    color_o, o_col = up(torch.randn(G, 3), dev), up(torch.randn(G, 1), dev)
    bias = up(torch.randn(P, 3), dev)

    def fwd():
        Q = ttnn.matmul(Phi, thT)                       # (ii) Phi.theta  [P,G]
        w = ttnn.square(ttnn.relu(ttnn.rsub(ttnn.mul(Q, 1.0 / K), 1.0)))  # (iii) poly
        num = ttnn.add(ttnn.matmul(w, color_o), bias)   # (iv) Sum w c  [P,3]
        den = ttnn.add(ttnn.matmul(w, o_col), 0.05)     #      Sum w     [P,1]
        return ttnn.div(num, den)                       # WSR normalize [P,3]

    spc = timed(fwd, dev, reps=reps)
    tf = 2.0 * P * G * (6 + 3 + 1) / spc / 1e12         # real GEMM FLOPs only
    print(f"  P={P:>6} G={G:>5} (P*G={P*G/1e6:5.1f}M): {spc*1e3:8.3f} ms/frame | "
          f"GEMM-eff {tf:7.1f} TFLOPS ({100*tf/PEAK_BF16:4.1f}% peak) | {1.0/spc:8.1f} frame/s")
    for t in (Phi, thT, color_o, o_col, bias):
        _dealloc(t)


def main():
    print(f"KMD/firmware in device-open log above. bf16 peak ref = {PEAK_BF16} TFLOPS (p150a)\n")
    dev = ttnn.open_device(device_id=0)
    try:
        print("== (1) square bf16 GEMM (matrix-engine utilization) ==")
        print(f"{'M=K=N':>8} | {'ms/call':>9} | {'TFLOPS':>8} | {'%peak':>6}")
        for s in [512, 1024, 2048, 4096, 8192]:
            spc, tf = gemm_row(dev, s, s, s)
            print(f"{s:>8} | {spc*1e3:9.3f} | {tf:8.1f} | {100*tf/PEAK_BF16:5.1f}")

        print("\n== (2) fat-shallow Phi.theta GEMM (P x 6 x G) -- K=6 padding loss expected ==")
        print(f"{'P':>8} {'G':>6} | {'ms/call':>9} | {'TFLOPS':>8} | {'%peak':>6}")
        for (P, G) in [(4096, 2048), (16384, 2048), (16384, 4096), (32768, 4096)]:
            spc, tf = gemm_row(dev, P, 6, G)
            print(f"{P:>8} {G:>6} | {spc*1e3:9.3f} | {tf:8.1f} | {100*tf/PEAK_BF16:5.1f}")

        print("\n== (3) full (B) forward hot path (dense, un-binned, un-fused) ==")
        for (P, G) in [(4096, 2048), (16384, 2048), (16384, 4096)]:
            forward_row(dev, P, G)

        print("\n== (4) forward stage breakdown @ P=16384 G=4096 (where the time goes) ==")
        breakdown_row(dev, 16384, 4096)
    finally:
        ttnn.close_device(dev)


def breakdown_row(dev, P, G, reps=20):
    """Time each forward stage in isolation: which lever (fuse poly? bin to kill P*G?) matters."""
    Phi, thT = up(torch.randn(P, 6), dev), up(torch.randn(6, G), dev)
    color_o, o_col = up(torch.randn(G, 3), dev), up(torch.randn(G, 1), dev)
    bias = up(torch.randn(P, 3), dev)
    Q0 = ttnn.matmul(Phi, thT)
    w0 = ttnn.square(ttnn.relu(ttnn.rsub(ttnn.mul(Q0, 1.0 / K), 1.0)))

    t_pt = timed(lambda: ttnn.matmul(Phi, thT), dev, reps=reps)
    t_poly = timed(lambda: ttnn.square(ttnn.relu(ttnn.rsub(ttnn.mul(Q0, 1.0 / K), 1.0))), dev, reps=reps)

    def wsr():
        num = ttnn.add(ttnn.matmul(w0, color_o), bias)
        den = ttnn.add(ttnn.matmul(w0, o_col), 0.05)
        return ttnn.div(num, den)

    t_wsr = timed(wsr, dev, reps=reps)
    tot = t_pt + t_poly + t_wsr
    for name, t in [("Phi.theta GEMM", t_pt), ("poly (SFPU: mul/rsub/relu/sq)", t_poly), ("WSR (2 GEMMs + div)", t_wsr)]:
        print(f"  {name:<32}: {t*1e3:7.3f} ms ({100*t/tot:4.1f}%)")
    print(f"  {'sum of isolated stages':<32}: {tot*1e3:7.3f} ms  (note: the [P,G]={P*G/1e6:.0f}M "
          f"intermediate is written+reread each stage -> fusion target)")
    for t in (Phi, thT, color_o, o_col, bias, Q0, w0):
        _dealloc(t)


if __name__ == "__main__":
    main()
