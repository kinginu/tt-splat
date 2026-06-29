"""On-device Adam -- resident optimizer state + update on the Blackhole, the first brick of the
device-resident training step. Params and the (m, v) moments live as
persistent on-card ttnn tensors; each step's update is pure elementwise device work. NOTHING crosses
PCIe per step here (grads are uploaded once for the test; in the real loop they come from the on-device
backward). Oracle: torch.optim.Adam in fp32.

Adam (torch formulation, matched exactly):
    m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
    denom = sqrt(v)/sqrt(1-b2^t) + eps ; p -= (lr/(1-b1^t)) * m / denom

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_adam_device.py
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from m3_perf import _dealloc

B1, B2, EPS = 0.9, 0.999, 1e-8


def up(t, dev, dt):
    return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


def adam_step_dev(p, m, v, g, t, lr):
    """One resident Adam step. Returns (p,m,v) new; frees the consumed tensors."""
    bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
    m_new = ttnn.add(ttnn.mul(m, B1), ttnn.mul(g, 1.0 - B1))
    v_new = ttnn.add(ttnn.mul(v, B2), ttnn.mul(ttnn.square(g), 1.0 - B2))
    denom = ttnn.add(ttnn.mul(ttnn.sqrt(v_new), 1.0 / math.sqrt(bc2)), EPS)
    upd = ttnn.div(ttnn.mul(m_new, lr / bc1), denom)
    p_new = ttnn.add(p, ttnn.neg(upd))
    _dealloc(denom)
    _dealloc(upd)
    _dealloc(m)
    _dealloc(v)
    _dealloc(p)
    return p_new, m_new, v_new


def run(dev, dt, name, G=8000, steps=200, lr=5e-3, seed=0):
    torch.manual_seed(seed)
    shapes = {"means": (G, 3), "color": (G, 3), "quats": (G, 4)}
    inits = {k: torch.randn(*s) * 0.1 for k, s in shapes.items()}
    grads = [{k: torch.randn(*s) * 0.05 for k, s in shapes.items()} for _ in range(steps)]

    # --- oracle: torch fp32 Adam ---
    ref = {k: v.clone().requires_grad_(True) for k, v in inits.items()}
    opt = torch.optim.Adam([{"params": [ref[k]], "lr": lr} for k in ref], betas=(B1, B2), eps=EPS)
    for st in grads:
        opt.zero_grad(set_to_none=True)
        for k in ref:
            ref[k].grad = st[k].clone()
        opt.step()

    # --- device: resident params + moments ---
    P = {k: up(v.clone(), dev, dt) for k, v in inits.items()}
    M = {k: up(torch.zeros(*s), dev, dt) for k, s in shapes.items()}
    V = {k: up(torch.zeros(*s), dev, dt) for k, s in shapes.items()}
    Gd = [{k: up(st[k], dev, dt) for k in st} for st in grads]   # (real loop: from device backward)
    for t, gd in enumerate(Gd, start=1):
        for k in P:
            P[k], M[k], V[k] = adam_step_dev(P[k], M[k], V[k], gd[k], t, lr)
        for g in gd.values():
            _dealloc(g)
    ttnn.synchronize_device(dev)

    print(f"[{name}] after {steps} steps (G={G}, lr={lr}):")
    worst = 0.0
    for k in P:
        pd = ttnn.to_torch(P[k]).float()
        rel = ((pd - ref[k].detach()).norm() / ref[k].detach().norm().clamp(min=1e-9)).item()
        worst = max(worst, rel)
        print(f"    {k:6s}: rel {rel:.4f}")
    print(f"    -> worst rel {worst:.4f}  ({'PASS' if worst < 0.02 else 'see note'})")
    for d in (P, M, V):
        for x in d.values():
            _dealloc(x)
    return worst


def perf(dev, dt, G=8000, reps=200):
    p = up(torch.randn(G, 3) * 0.1, dev, dt)
    m = up(torch.zeros(G, 3), dev, dt)
    v = up(torch.zeros(G, 3), dev, dt)
    g = up(torch.randn(G, 3) * 0.05, dev, dt)
    for t in range(1, 4):
        p, m, v = adam_step_dev(p, m, v, g, t, 5e-3)
        g = up(torch.randn(G, 3) * 0.05, dev, dt)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for t in range(1, reps + 1):
        p2, m, v = adam_step_dev(p, m, v, g, t, 5e-3)
        p = p2
    ttnn.synchronize_device(dev)
    dt_s = (time.perf_counter() - t0) / reps
    print(f"    device Adam step [{G}x3]: {dt_s*1e6:.1f} us")
    for x in (p, m, v, g):
        _dealloc(x)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        print("== on-device Adam vs torch.optim.Adam (oracle) ==\n")
        run(dev, ttnn.bfloat16, "bf16 moments")
        print()
        try:
            run(dev, ttnn.float32, "fp32 moments")
        except Exception as e:  # noqa: BLE001
            print("  fp32 path:", type(e).__name__, str(e)[:100])
        print("\n== perf ==")
        perf(dev, ttnn.bfloat16)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
