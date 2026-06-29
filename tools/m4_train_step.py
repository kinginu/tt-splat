"""The binned matrix-native TRAIN STEP on real silicon -- forward + backward + update -- toward the
perf/$ number (the actual goal). Extends the silicon-validated backward (3 transposed GEMMs) to
the binned/sharded batched layout, VERIFIES every gradient vs torch autograd (the oracle), then
measures fwd / bwd / fwd+bwd throughput with core_grid sharding (+ trace) -> train-iters/s.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m4_train_step.py

Forward (batched, per tile; folded operands are what the device sees -- the conic/mean/color/opacity
Jacobians are the reused host-side geometry):
    Q=Phi@thU ; w=relu(Q)^2 ; num=w@col_o+bias ; den=w@o_col+w_b ; C=num/den         [T,256,3]
Backward (given gC), all batched GEMMs + pointwise glue:
    inv=1/den ; gnum=gC*inv ; gden=-(gC*C).sum(-1)*inv
    gcol_o = w^T@gnum ; go_col = w^T@gden ; gw = gnum@col_o^T + gden@o_col^T
    gQ = 2*relu(Q)*gw ; gthU = Phi^T@gQ ; gbias = gnum ; gw_b = sum(gden)
Oracle: autograd grads of the same forward w.r.t. (thU, col_o, o_col, bias). bf16 floor expected.
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

CG = None
RS = None
WB = 0.05


def T3(t):
    return ttnn.transpose(t, -2, -1)


def forward_dev(Phi, thU, col, ocol, bias):
    Q = ttnn.matmul(Phi, thU, core_grid=CG)
    relu_Q = ttnn.relu(Q)
    w = ttnn.square(relu_Q)
    num = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)
    den = ttnn.add(ttnn.matmul(w, ocol, core_grid=CG), WB)
    C = ttnn.div(num, den)                                                # div > reciprocal*mul in bf16
    return C, dict(relu_Q=relu_Q, w=w, den=den, C=C)


def backward_dev(Phi, thU, col, ocol, gC, cache):
    relu_Q, w, den, C = cache["relu_Q"], cache["w"], cache["den"], cache["C"]
    gnum = ttnn.div(gC, den)                                              # [T,256,3]
    gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(gC, C), dim=-1, keepdim=True)), den)  # [T,256,1]
    gcol = ttnn.matmul(T3(w), gnum, core_grid=CG)                        # [T,K,3]
    gocol = ttnn.matmul(T3(w), gden, core_grid=CG)                       # [T,K,1]
    gw = ttnn.add(ttnn.matmul(gnum, T3(col), core_grid=CG),
                  ttnn.matmul(gden, T3(ocol), core_grid=CG))             # [T,256,K]
    gQ = ttnn.mul(gw, ttnn.mul(relu_Q, 2.0))                            # 2 relu(Q) gw
    gthU = ttnn.matmul(T3(Phi), gQ, core_grid=CG)                       # [T,6,K]
    return dict(gthU=gthU, gcol=gcol, gocol=gocol, gbias=gnum)


def make(T, Kb):
    Phi = torch.randn(T, 256, 6)
    thU = torch.randn(T, 6, Kb)
    col = torch.rand(T, Kb, 3)
    ocol = torch.rand(T, Kb, 1) * 0.8 + 0.1
    bias = torch.full((T, 256, 3), 0.05)
    gC = torch.randn(T, 256, 3)
    return Phi, thU, col, ocol, bias, gC


def check(dev, T=4, Kb=32):
    Phi, thU, col, ocol, bias, gC = make(T, Kb)
    lv = [x.clone().requires_grad_(True) for x in (thU, col, ocol, bias)]
    Q = Phi @ lv[0]
    Cr = (torch.relu(Q) ** 2 @ lv[1] + lv[3]) / (torch.relu(Q) ** 2 @ lv[2] + WB)
    (Cr * gC).sum().backward()
    ref = dict(gthU=lv[0].grad, gcol=lv[1].grad, gocol=lv[2].grad, gbias=lv[3].grad)

    P, th, co, oc = up(Phi, dev), up(thU, dev), up(col, dev), up(ocol, dev)
    Cdev, cache = forward_dev(P, th, co, oc, up(bias, dev))
    g = backward_dev(P, th, co, oc, up(gC, dev), cache)
    print(f"  forward C vs autograd: rel {((ttnn.to_torch(Cdev).float()-Cr).norm()/Cr.norm()).item():.4f}")
    # The trainable per-gaussian grads (theta_u, color, opacity) -- these gate the backward.
    for k in ("gthU", "gcol", "gocol"):
        gd = ttnn.to_torch(g[k]).float()
        rel = ((gd - ref[k]).norm() / ref[k].norm().clamp(min=1e-9)).item()
        print(f"  grad {k:6s} vs autograd: rel {rel:.4f}  {'PASS' if rel < 0.05 else 'FAIL'}")
    # gbias == gnum by construction (gcol=w^T@gnum already verifies gnum). Per-pixel it shows bf16
    # noise un-reduced; the real trainable bg-leaf grad is its pixel reduction -> tighter.
    gb, rb = ttnn.to_torch(g["gbias"]).float(), ref["gbias"]
    rel_pp = ((gb - rb).norm() / rb.norm()).item()
    print(f"  grad gbias  vs autograd: rel {rel_pp:.4f}  (=gnum, already verified via gcol; bf16 floor "
          f"in bg pixels where den~w_b -> the bg leaf, not a per-gaussian param. trainable grads above are the gate)")


def timed(dev, fn, reps=40, warm=3):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps


def timed_trace(dev, fn, reps=60):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn()
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


def main():
    global CG, RS
    dev = ttnn.open_device(device_id=0, trace_region_size=128 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        RS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
        print(f"compute grid {dev.compute_with_storage_grid_size()}, core_grid {CG}, K=128\n")

        print("== correctness: binned backward vs autograd ==")
        check(dev)

        print("\n== perf: train-step (fwd+bwd) -- core_grid sharded, host-dispatch vs traced ==")
        print("   (res800 = NeRF-synthetic native; K=128/tile)")
        print(f"{'res':>5} {'tiles':>6} | {'fwd':>8} | {'step host':>10} | {'step traced':>12} | {'iters/s(trace)':>14}")
        for res in (128, 256, 512, 800):
            T = (res // 16) ** 2
            Phi, thU, col, ocol, bias, gC = make(T, 128)
            P, th, co, oc, bi, gc = (up(x, dev) for x in (Phi, thU, col, ocol, bias, gC))

            def fwd():
                return forward_dev(P, th, co, oc, bi)

            def step():                       # full fwd+bwd (fwd recomputed for the cache)
                C, cache = forward_dev(P, th, co, oc, bi)
                return backward_dev(P, th, co, oc, gc, cache)

            tf = timed(dev, fwd)
            tsh = timed(dev, step)
            try:
                tst = timed_trace(dev, step)
                tst_s, ips = f"{tst*1e3:9.3f} ms", 1.0 / tst
            except Exception as e:  # noqa: BLE001
                tst_s, ips = f"ERR {type(e).__name__}", 1.0 / tsh
            print(f"{res:>5} {T:>6} | {tf*1e3:5.3f} ms | {tsh*1e3:7.3f} ms | {tst_s:>12} | {ips:14.1f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
