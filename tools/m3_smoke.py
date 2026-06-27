"""Smoke-test the tt-metal toolchain on REAL Blackhole silicon.

Run inside the hw container (real device, fast dispatch):
    podman-compose --profile hw run --rm hw python3 tools/m3_smoke.py

Confirms (a) ttnn opens the real Blackhole and (b) a basic bf16 GEMM on the matrix
engine matches torch to the bf16 floor. This is the entry gate:
"Start with the simplest kernel ... to confirm the toolchain works on real silicon."
This is CORRECTNESS only — perf is a separate timed run (tools/m3_perf.py).

ttnn API read from the installed source / the ttsim tools: open_device,
from_torch(layout=TILE_LAYOUT, dtype=bfloat16, device=), matmul, to_torch, close_device.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ttnn


def main():
    if os.environ.get("TT_METAL_SIMULATOR"):
        print("WARNING: TT_METAL_SIMULATOR is set -> this is the ttsim, NOT real silicon.")
    print("TT_METAL_SLOW_DISPATCH_MODE =", os.environ.get("TT_METAL_SLOW_DISPATCH_MODE", "(unset -> fast dispatch)"))

    try:
        print("num available devices:", ttnn.GetNumAvailableDevices())
    except Exception as e:  # noqa: BLE001 - probe only
        print("(GetNumAvailableDevices probe failed:", e, ")")

    ok = False
    dev = ttnn.open_device(device_id=0)
    print("device opened:", dev)
    try:
        torch.manual_seed(0)
        M, K, N = 256, 256, 256
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        ref = A @ B

        at = ttnn.from_torch(A, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        bt = ttnn.from_torch(B, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.to_torch(ttnn.matmul(at, bt)).float()

        rel = ((out - ref).norm() / ref.norm()).item()
        ref_bf16 = (A.bfloat16() @ B.bfloat16()).float()
        rel_floor = ((ref_bf16 - ref).norm() / ref.norm()).item()
        print(f"GEMM {M}x{K}x{N}: rel-err device vs fp32 = {rel:.4f}  (bf16 CPU floor {rel_floor:.4f})")
        ok = rel < 0.05
        print("smoke (real-silicon GEMM):", "PASS" if ok else "FAIL")
    finally:
        ttnn.close_device(dev)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
