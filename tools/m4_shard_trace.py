"""SHARDING (use all 110 cores) -- the fix for the occupancy bound that the binned forward revealed.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m4_shard_trace.py

The binned forward was flat in K and unhelped by trace => device-bound but under-occupied.
Per-op profiling showed the BATCHED matmuls' ttnn-auto core grid is the culprit: batched Phi.theta
0.205 ms on auto vs 0.041 ms when handed the explicit 11x10 compute grid (5x). The fix is to pass
`core_grid` (= all 110 Tensix cores) to every matmul in the binned forward. Trace is retested on top
(it was a no-op on the dispatch path; confirm it stays so once compute-bound).

Variants @ K=128, same binned forward as tools/m4_binned_forward.py (math already oracle-verified):
  V_auto : ttnn-auto core grid (the binned forward)
  V_grid : core_grid=11x10 on every matmul                 <- the sharding win
  V_grid+trace : capture+replay V_grid

NOTE: this shards the matmul COMPUTE grid; tensors are still DRAM-interleaved between ops. Full
L1-resident sharded tensors (create_sharded_memory_config, no DRAM round-trip between ops) is the
deeper follow-up toward the "scene fits L1, DRAM paid once" target.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from m3_perf import PEAK_BF16, _dealloc, up

RS = None
CG = None


def make_inputs(dev, T, Kb):
    return dict(
        Phi=up(torch.randn(T, 256, 6), dev), thU=up(torch.randn(T, 6, Kb), dev),
        col=up(torch.rand(T, Kb, 3), dev), ocol=up(torch.rand(T, Kb, 1) * 0.8 + 0.1, dev),
        bias=up(torch.full((T, 256, 3), 0.05), dev))


def fwd_fn(io, grid):
    mm = (lambda a, b: ttnn.matmul(a, b, core_grid=grid)) if grid else ttnn.matmul

    def fwd():
        w = ttnn.unary_chain(mm(io["Phi"], io["thU"]), RS)
        num = ttnn.add(mm(w, io["col"]), io["bias"])
        den = ttnn.add(mm(w, io["ocol"]), 0.05)
        return ttnn.div(num, den)

    return fwd


def timed(dev, fwd, reps=40):
    for _ in range(3):
        _dealloc(fwd())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        _dealloc(fwd())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps


def timed_trace(dev, fwd, reps=100):
    _dealloc(fwd())
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fwd()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t = (time.perf_counter() - t0) / reps
    try:
        ttnn.release_trace(dev, tid)
    except Exception:  # noqa: BLE001
        pass
    return t


def check(dev, T=4, Kb=32):
    io = make_inputs(dev, T, Kb)
    Phi, thU = ttnn.to_torch(io["Phi"]).float(), ttnn.to_torch(io["thU"]).float()
    col, ocol, bias = ttnn.to_torch(io["col"]).float(), ttnn.to_torch(io["ocol"]).float(), ttnn.to_torch(io["bias"]).float()
    w = torch.relu(Phi @ thU) ** 2
    C_ref = (w @ col + bias) / (w @ ocol + 0.05)
    C = ttnn.to_torch(fwd_fn(io, CG)()).float()
    rel = ((C - C_ref).norm() / C_ref.norm()).item()
    print(f"  sharded (core_grid) binned forward vs torch: rel-err {rel:.4f}  {'PASS' if rel < 0.05 else 'FAIL'}")
    for b in io.values():
        _dealloc(b)


def check_trace(dev, T=4, Kb=32):
    """Verify the REPLAYED (execute_trace) output -- not just the host-dispatch path -- vs torch."""
    io = make_inputs(dev, T, Kb)
    Phi, thU = ttnn.to_torch(io["Phi"]).float(), ttnn.to_torch(io["thU"]).float()
    col, ocol, bias = ttnn.to_torch(io["col"]).float(), ttnn.to_torch(io["ocol"]).float(), ttnn.to_torch(io["bias"]).float()
    C_ref = (torch.relu(Phi @ thU) ** 2 @ col + bias) / (torch.relu(Phi @ thU) ** 2 @ ocol + 0.05)
    fwd = fwd_fn(io, CG)
    _dealloc(fwd())
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fwd()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)   # replay, then read the replayed buffer
    C = ttnn.to_torch(out).float()
    rel = ((C - C_ref).norm() / C_ref.norm()).item()
    print(f"  REPLAYED (execute_trace) output vs torch:    rel-err {rel:.4f}  {'PASS' if rel < 0.05 else 'FAIL'}")
    try:
        ttnn.release_trace(dev, tid)
    except Exception:  # noqa: BLE001
        pass
    for b in io.values():
        _dealloc(b)


def main():
    global RS, CG
    dev = ttnn.open_device(device_id=0, trace_region_size=128 * 1024 * 1024)
    try:
        RS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
        CG = ttnn.CoreGrid(x=11, y=10)
        print(f"compute grid {dev.compute_with_storage_grid_size()} -> core_grid {CG}, K=128\n")

        print("== correctness ==")
        check(dev)
        check_trace(dev)

        print("\n== perf: auto-grid vs sharded(core_grid) vs +trace ==")
        print(f"{'res':>5} {'tiles':>6} | {'V_auto':>9} | {'V_grid':>9} | {'grid x':>7} | {'V_grid+trace':>13} | {'best fps':>9}")
        for res in (128, 256):
            T = (res // 16) ** 2
            io = make_inputs(dev, T, 128)
            ta = timed(dev, fwd_fn(io, None))
            tg = timed(dev, fwd_fn(io, CG))
            try:
                tgt = timed_trace(dev, fwd_fn(io, CG))
                tgt_s = f"{tgt*1e3:10.3f} ms"
            except Exception as e:  # noqa: BLE001
                tgt, tgt_s = tg, f"ERR {type(e).__name__}"
            best = min(tg, tgt)
            print(f"{res:>5} {T:>6} | {ta*1e3:6.3f} ms | {tg*1e3:6.3f} ms | {ta/tg:5.2f}x | {tgt_s:>13} | {1.0/best:8.1f}")
            for b in io.values():
                _dealloc(b)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
