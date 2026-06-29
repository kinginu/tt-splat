"""On-device backward of the theta_u build -- the one missing device-backward link. The render
backward (gC -> gthU, gcol, gocol) is already silicon-verified (tools/m4_train_step.py); the gather
scatter-add and the EWA/quaternion geometry Jacobians stay host (branchy, O(G)); Adam is a separate device unit. What
remained on device is theta_u_gathered's backward: gthU[T,6,K] -> gconic_t[T,K,3], gmu_t[T,K,2]. With
this, the device backward chain closes from the loss down to the per-gaussian geometry outputs.

theta_u (per tile) = ((-1/k) tQ + bump).transpose x valid, with
    tQ = [a, c, 2b, -2(a mux + b muy), -2(b mux + c muy), a mux^2 + 2b mux muy + c muy^2]
(a,b,c)=conic_t cols, (mux,muy)=mu_t-origin. Backward is the elementwise Jacobian of tQ.
Oracle: torch autograd of theta_u_gathered.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_theta_bwd.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from m4_train_binned import theta_u_gathered, K_POLY


def Mu(a, b):
    return ttnn.mul(a, b)


def Ad(a, b):
    return ttnn.add(a, b)


def theta_build_bwd_dev(dev, dt, conic_t, mu_t, origins, valid, gtheta):
    """gtheta[T,6,K] -> gconic_t[T,K,3], gmu_t[T,K,2] (device)."""
    T, K = conic_t.shape[0], conic_t.shape[1]

    def u(t):
        return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    a, b, c = u(conic_t[:, :, 0:1]), u(conic_t[:, :, 1:2]), u(conic_t[:, :, 2:3])
    mu = mu_t - origins[:, None, :]
    mux, muy = u(mu[:, :, 0:1]), u(mu[:, :, 1:2])
    valid6 = u(valid[:, None, :].expand(T, 6, K).float())
    gth = Mu(u(gtheta), valid6)                                 # mask
    gpoly = ttnn.transpose(gth, 1, 2)                           # [T,K,6]
    gtQ = Mu(gpoly, -1.0 / K_POLY)                              # bump const -> no grad
    g0, g1, g2 = gtQ[:, :, 0:1], gtQ[:, :, 1:2], gtQ[:, :, 2:3]
    g3, g4, g5 = gtQ[:, :, 3:4], gtQ[:, :, 4:5], gtQ[:, :, 5:6]
    mux2, muy2, muxy = Mu(mux, mux), Mu(muy, muy), Mu(mux, muy)

    ga = Ad(Ad(g0, Mu(g3, Mu(mux, -2.0))), Mu(g5, mux2))
    gb = Ad(Ad(Mu(g2, 2.0), Mu(g3, Mu(muy, -2.0))), Ad(Mu(g4, Mu(mux, -2.0)), Mu(g5, Mu(muxy, 2.0))))
    gc = Ad(Ad(g1, Mu(g4, Mu(muy, -2.0))), Mu(g5, muy2))
    gmux = Ad(Ad(Mu(g3, Mu(a, -2.0)), Mu(g4, Mu(b, -2.0))),
              Mu(g5, Ad(Mu(a, Mu(mux, 2.0)), Mu(b, Mu(muy, 2.0)))))
    gmuy = Ad(Ad(Mu(g3, Mu(b, -2.0)), Mu(g4, Mu(c, -2.0))),
              Mu(g5, Ad(Mu(b, Mu(mux, 2.0)), Mu(c, Mu(muy, 2.0)))))
    gconic_t = ttnn.concat([ga, gb, gc], dim=-1)
    gmu_t = ttnn.concat([gmux, gmuy], dim=-1)
    return ttnn.to_torch(gconic_t).float(), ttnn.to_torch(gmu_t).float()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def run(dev, dt, name, T=64, K=256, seed=0):
    torch.manual_seed(seed)
    conic_t = torch.randn(T, K, 3) * 0.3
    mu_t = torch.randn(T, K, 2) * 8.0
    origins = torch.rand(T, 2) * 100.0
    valid = (torch.rand(T, K) > 0.2)
    gtheta = torch.randn(T, 6, K)

    ct = conic_t.clone().requires_grad_(True)
    mt = mu_t.clone().requires_grad_(True)
    theta = theta_u_gathered(ct, mt, origins, K_POLY) * valid[:, None, :].float()
    (theta * gtheta).sum().backward()

    gct, gmt = theta_build_bwd_dev(dev, dt, conic_t, mu_t, origins, valid, gtheta)
    print(f"[{name}] T={T} K={K}:")
    print(f"    gconic_t rel {rel(gct, ct.grad):.4f}")
    print(f"    gmu_t    rel {rel(gmt, mt.grad):.4f}")
    worst = max(rel(gct, ct.grad), rel(gmt, mt.grad))
    print(f"    -> worst rel {worst:.4f}  ({'PASS' if worst < 0.03 else 'see note'})")
    return worst


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        print("== on-device theta_u-build backward vs autograd (oracle) ==\n")
        run(dev, ttnn.bfloat16, "bf16")
        print()
        try:
            run(dev, ttnn.float32, "fp32")
        except Exception as e:  # noqa: BLE001
            print("  fp32 path:", type(e).__name__, str(e)[:100])
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
