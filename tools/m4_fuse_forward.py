"""Poly fusion (the cheap fusion path): fold the affine + fuse the poly's RELU into the Phi.theta
matmul epilogue, killing 3 of the 4 SFPU passes the naive baseline spent 60% of the forward in.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m4_fuse_forward.py

The poly is  w_geo = relu(1 - Q/k)^2  with  Q = Phi.theta_Q^T.  Two free rewrites:
  (a) AFFINE FOLD: Phi's 6th monomial is the constant 1, so `1 - Q/k` folds into a 6-wide theta_u
      (forward.theta_from_conic returns it).  matmul(Phi, theta_u^T) = 1 - Q/k  exactly, with NO
      separate mul/rsub ops.  -> kills 2 of the 4 elementwise [P,G] passes.
  (b) EPILOGUE RELU: ttnn matmul applies one fused_activation in the matmul epilogue (UnaryOpType
      has RELU), so relu(1 - Q/k) comes straight out of the GEMM with no extra [P,G] round-trip.
      -> kills the 3rd pass.  Only the final `square` remains as a standalone elementwise op.

Three variants are timed at the same shape so each fusion's contribution is visible:
  V0 baseline       : matmul(theta_Q) -> mul -> rsub -> relu -> square      (the naive baseline poly)
  V1 affine-folded  : matmul(theta_u) -> relu -> square                     (fold saves mul+rsub)
  V2 relu-epilogue  : matmul(theta_u, act=relu) -> square                   (also fuses the relu)
Correctness: V1 and V2 are diffed against the exact arm-A CPU oracle (m2_forward_ttsim.cpu_reference).
This is the documented-ttnn cheap path; a hand-written Metalium kernel that ALSO fuses square + the
WSR GEMMs (so [P,G] never touches DRAM) is the next, larger step.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root -> `spike`
sys.path.insert(0, HERE)                     # tools/   -> sibling modules

import torch
import ttnn

import m2_forward_ttsim as m2f               # the exact arm-A oracle + make_case
from m3_perf import PEAK_BF16, _dealloc, timed, up
from spike import forward

K = m2f.K

# unary_chain (relu+square in one SFPU pass) availability is build-dependent; probed at startup.
_UNARY_CHAIN_OK = False


def detect_unary_chain(dev):
    a = up(torch.randn(64, 64), dev)
    try:
        chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
        out = ttnn.unary_chain(a, chain)
        ref = torch.relu(ttnn.to_torch(a).float()) ** 2
        ok = ((ttnn.to_torch(out).float() - ref).norm() / ref.norm().clamp(min=1e-9)).item() < 0.05
    except Exception as e:  # noqa: BLE001
        print("  (unary_chain unavailable:", type(e).__name__, str(e)[:80], ")")
        ok = False
    finally:
        _dealloc(a)
    return ok


def detect_fused_relu(dev):
    """Does ttnn.matmul accept a fused `activation` epilogue on this build?"""
    a = up(torch.randn(64, 64), dev)
    b = up(torch.randn(64, 64), dev)
    try:
        out = ttnn.matmul(a, b, activation="relu")
        ref = torch.relu(ttnn.to_torch(a).float() @ ttnn.to_torch(b).float())
        rel = ((ttnn.to_torch(out).float() - ref).norm() / ref.norm()).item()
        ok = rel < 0.05
    except Exception as e:  # noqa: BLE001
        print("  (activation='relu' kwarg not accepted:", type(e).__name__, str(e)[:80], ")")
        ok = False
    finally:
        _dealloc(a)
        _dealloc(b)
    return ok


# ---------- correctness: V1 / V2 vs the arm-A CPU oracle ----------
def fused_forward_cpu_inputs(dev, case, fuse_relu):
    origin, conic, mu, color, o, w_b, c_b, lx, ly = case
    P = lx.shape[0]
    Phi = forward.phi(lx, ly)
    _, theta_u = forward.theta_from_conic(conic, mu - origin, k=K)     # folds 1 - Q/k
    color_o = o[:, None] * color
    o_col = o[:, None]
    bias = (w_b * c_b)[None, :].expand(P, 3).contiguous()

    Phi_t = up(Phi, dev)
    thu_t = up(theta_u.transpose(0, 1).contiguous(), dev)
    if fuse_relu:
        u_t = ttnn.matmul(Phi_t, thu_t, activation="relu")            # V2: relu in epilogue
    else:
        u_t = ttnn.relu(ttnn.matmul(Phi_t, thu_t))                    # V1: separate relu
    w_t = ttnn.square(u_t)
    num = ttnn.add(ttnn.matmul(w_t, up(color_o, dev)), up(bias, dev))
    den = ttnn.add(ttnn.matmul(w_t, up(o_col, dev)), float(w_b))
    return ttnn.to_torch(ttnn.div(num, den)).float()


def check_correctness(dev, fused_relu_ok):
    case = m2f.make_case()
    C_ref = m2f.cpu_reference(*case)
    for name, fuse in [("V1 affine-folded", False), ("V2 relu-epilogue", fused_relu_ok)]:
        C = fused_forward_cpu_inputs(dev, case, fuse)
        rel = ((C - C_ref).norm() / C_ref.norm()).item()
        tag = "PASS" if rel < 0.05 else "FAIL"
        note = "" if (fuse or name.startswith("V1")) else "  (relu kwarg absent -> ran as V1)"
        print(f"  {name}: rel-err vs arm-A oracle = {rel:.4f}  {tag}{note}")


# ---------- perf: V0 / V1 / V2 at one realistic shape ----------
def bench(dev, P, G, fused_relu_ok, reps=20):
    Phi = up(torch.randn(P, 6), dev)
    thQ = up(torch.randn(6, G), dev)
    thU = up(torch.randn(6, G), dev)
    color_o, o_col, bias = up(torch.randn(G, 3), dev), up(torch.randn(G, 1), dev), up(torch.randn(P, 3), dev)

    def wsr(w):
        num = ttnn.add(ttnn.matmul(w, color_o), bias)
        den = ttnn.add(ttnn.matmul(w, o_col), 0.05)
        return ttnn.div(num, den)

    def v0():  # baseline
        Q = ttnn.matmul(Phi, thQ)
        w = ttnn.square(ttnn.relu(ttnn.rsub(ttnn.mul(Q, 1.0 / K), 1.0)))
        return wsr(w)

    def v1():  # affine-folded, separate relu
        w = ttnn.square(ttnn.relu(ttnn.matmul(Phi, thU)))
        return wsr(w)

    def v2():  # relu fused into epilogue
        w = ttnn.square(ttnn.matmul(Phi, thU, activation="relu"))
        return wsr(w)

    relu_sq = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]

    def v3():  # affine-fold + relu+square in ONE sfpu pass via unary_chain
        w = ttnn.unary_chain(ttnn.matmul(Phi, thU), relu_sq)
        return wsr(w)

    variants = [("V0 baseline (unfused)", v0), ("V1 affine-folded", v1)]
    if fused_relu_ok:
        variants.append(("V2 relu-epilogue-fused", v2))
    if _UNARY_CHAIN_OK:
        variants.append(("V3 unary_chain[relu,sq]", v3))
    t0 = None
    print(f"\n  P={P} G={G} (P*G={P*G/1e6:.0f}M), {reps} reps:")
    for name, fn in variants:
        t = timed(fn, dev, reps=reps)
        if t0 is None:
            t0 = t
        print(f"    {name:<26}: {t*1e3:7.3f} ms/frame | {1.0/t:7.1f} fps | {t0/t:4.2f}x vs baseline")
    for x in (Phi, thQ, thU, color_o, o_col, bias):
        _dealloc(x)


def main():
    global _UNARY_CHAIN_OK
    dev = ttnn.open_device(device_id=0)
    try:
        print("== detecting matmul epilogue-activation support ==")
        fused_relu_ok = detect_fused_relu(dev)
        _UNARY_CHAIN_OK = detect_unary_chain(dev)
        print(f"  fused matmul activation='relu' supported: {fused_relu_ok}")
        print(f"  unary_chain[relu,square] supported:       {_UNARY_CHAIN_OK}")

        print("\n== correctness (V1/V2 vs arm-A oracle) ==")
        check_correctness(dev, fused_relu_ok)

        print("\n== perf: poly-fusion variants (peak ref %.0f TFLOPS) ==" % PEAK_BF16)
        for (P, G) in [(16384, 2048), (16384, 4096)]:
            bench(dev, P, G, fused_relu_ok)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
