"""The BINNED forward on real silicon -- the [P,G] -> [T,256,K] structural win.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m4_binned_forward.py

The locked binning scheme (quality proven lossless at R=1,K=128): the image is 16x16
tiles; each tile renders its 256 pixels from only its <=K=128 gaussians. The dense forward computes
every pixel x every gaussian ([P,G]); the binned forward computes T tiles x [256 pixels, K gaussians]
= a BATCHED tiled GEMM. For res128/G4096 that is 64x[256,128]=2.1M vs 16384x4096=67M -> ~32x less
compute & DRAM, and each tile's working set ([256,K] bf16 ~64 KB) is L1-resident.

Layout (the batched form -- maps 1 tile = 1 batch elem, the natural unit for the next sharding step):
  Q = matmul(Phi[T,256,6], theta_u[T,6,K])      -> [T,256,K]   (Phi tile-local, replicated; cheap)
  w = unary_chain[relu,square](Q)               -> [T,256,K]   (poly fusion)
  C = (w@color_o[T,K,3] + bias) / (w@o_col[T,K,1] + w_b)        -> [T,256,3]

The per-tile gaussian lists come from host-side bucketing (the locked CPU split); the on-device
NoC-atomic scatter is a SEPARATE change and is NOT
done here. For this PERF measurement the binned layout is synthesized (perf depends on shape/layout,
not values); correctness of the batched wiring is checked vs torch at the bf16 floor.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from m3_perf import PEAK_BF16, _dealloc, timed, up

K = 4.0  # poly k
RELU_SQ = None  # built after device-open (needs ttnn enums)


def dense_fused_v3(dev, P, G, reps=20):
    """The best non-binned forward (V3): affine-fold + unary_chain poly."""
    Phi, thU = up(torch.randn(P, 6), dev), up(torch.randn(6, G), dev)
    col, ocol, bias = up(torch.randn(G, 3), dev), up(torch.randn(G, 1), dev), up(torch.randn(P, 3), dev)

    def fwd():
        w = ttnn.unary_chain(ttnn.matmul(Phi, thU), RELU_SQ)
        num = ttnn.add(ttnn.matmul(w, col), bias)
        den = ttnn.add(ttnn.matmul(w, ocol), 0.05)
        return ttnn.div(num, den)

    t = timed(fwd, dev, reps=reps)
    for x in (Phi, thU, col, ocol, bias):
        _dealloc(x)
    return t


def binned_forward(dev, T, Kb, reps=20):
    """Batched binned forward: T tiles, each [256 px, Kb gaussians]."""
    Phi = up(torch.randn(T, 256, 6), dev)
    thU = up(torch.randn(T, 6, Kb), dev)
    col, ocol, bias = up(torch.randn(T, Kb, 3), dev), up(torch.randn(T, Kb, 1), dev), up(torch.randn(T, 256, 3), dev)

    def fwd():
        w = ttnn.unary_chain(ttnn.matmul(Phi, thU), RELU_SQ)     # [T,256,Kb]
        num = ttnn.add(ttnn.matmul(w, col), bias)                # [T,256,3]
        den = ttnn.add(ttnn.matmul(w, ocol), 0.05)              # [T,256,1]
        return ttnn.div(num, den)

    t = timed(fwd, dev, reps=reps)
    for x in (Phi, thU, col, ocol, bias):
        _dealloc(x)
    return t


def check_correctness(dev, T=4, Kb=32):
    # Realistic WSR ranges (match m2_forward.make_case): color_o = o*color >= 0, opacity o in
    # [0.1,0.9] > 0, bias = w_b*c_b > 0  =>  den = sum(w*o) + w_b >= w_b > 0 (well-conditioned).
    # (randn opacities would make den ~ 0 / negative -> div singularities, a bad test not a real bug.)
    torch.manual_seed(0)
    Phi, thU = torch.randn(T, 256, 6), torch.randn(T, 6, Kb)
    col = torch.rand(T, Kb, 3)
    ocol = torch.rand(T, Kb, 1) * 0.8 + 0.1
    bias = torch.full((T, 256, 3), 0.05)
    Q = Phi @ thU
    w = torch.relu(Q) ** 2
    C_ref = (w @ col + bias) / (w @ ocol + 0.05)

    Phi_t, thU_t = up(Phi, dev), up(thU, dev)
    w_t = ttnn.unary_chain(ttnn.matmul(Phi_t, thU_t), RELU_SQ)
    num = ttnn.add(ttnn.matmul(w_t, up(col, dev)), up(bias, dev))
    den = ttnn.add(ttnn.matmul(w_t, up(ocol, dev)), 0.05)
    C = ttnn.to_torch(ttnn.div(num, den)).float()
    rel = ((C - C_ref).norm() / C_ref.norm()).item()
    print(f"  batched binned forward vs torch (bf16): rel-err {rel:.4f}  {'PASS' if rel < 0.05 else 'FAIL'}")


def main():
    global RELU_SQ
    dev = ttnn.open_device(device_id=0)
    try:
        RELU_SQ = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]

        print("== correctness (batched binned wiring vs torch) ==")
        check_correctness(dev)

        print("\n== perf: binned vs dense-fused, same image (G_dense=4096, K=128) ==")
        print(f"{'res':>5} {'tiles T':>8} | {'dense-fused':>12} | {'binned':>10} | {'speedup':>8} | {'binned fps':>10}")
        for res, G in [(128, 4096), (256, 4096)]:
            T = (res // 16) ** 2
            P = res * res
            td = dense_fused_v3(dev, P, G)
            tb = binned_forward(dev, T, 128)
            print(f"{res:>5} {T:>8} | {td*1e3:9.3f} ms | {tb*1e3:7.3f} ms | {td/tb:6.2f}x | {1.0/tb:8.1f}")

        print("\n== binned K sweep @ res128 (T=64) ==")
        for Kb in (32, 64, 128, 256):
            tb = binned_forward(dev, 64, Kb)
            print(f"  K={Kb:>4}: {tb*1e3:7.3f} ms | {1.0/tb:8.1f} fps")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
