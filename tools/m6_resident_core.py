"""full-resident CORE: resident params on device + device geometry fwd/bwd + device Adam,
NO per-iter param upload / grad download. Verifies the residency mechanic: run N steps with fixed
(gconic, gmu2d) and check the resident device params evolve identically to a host reference (host geometry
+ torch Adam). Single view (P=G) to isolate the resident loop from tiling/reduction. Once correct, this is
the core the full-resident trainer is built on (render/loss/binning wired around it).

Oracle: host geometry (m6_geom_bwd) + torch.optim.Adam, same fixed grads. Host-vs-device param drift.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m6_resident_core.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike.model import GaussianModel
import m6_geom_bwd as gbh
from m6_geom_device import device_fwd_core, device_bwd_core, A, M, S

B1, B2, EPS = 0.9, 0.999, 1e-8


def adam_dev(p, m, v, g, t, lr):
    """One device Adam step (fp32). Returns (p,m,v) new device tensors."""
    bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
    m2 = A(M(m, B1), M(g, 1.0 - B1))
    v2 = A(M(v, B2), M(M(g, g), 1.0 - B2))
    denom = A(M(ttnn.sqrt(v2), 1.0 / math.sqrt(bc2)), EPS)
    p2 = A(p, ttnn.neg(M(ttnn.div(M(m2, lr / bc1), denom), 1.0)))
    return p2, m2, v2


def main():
    dev = ttnn.open_device(device_id=0)
    dt = ttnn.float32
    try:
        G, STEPS = 8000, 6
        torch.manual_seed(0)
        gm = GaussianModel(G, extent=1.5, seed=0)
        means0, quat0, logs0 = gm.means3d.detach().clone(), gm.quats.detach().clone(), gm.log_scales.detach().clone()
        Rv_t = torch.tensor([[0.9, 0.1, -0.05], [-0.08, 0.95, 0.2], [0.05, -0.18, 0.98]])
        tv_t = torch.tensor([0.1, -0.2, 4.0])
        fx = fy = 128 * 1.2
        cx = cy = 64.0
        Rv = [[float(Rv_t[i, j]) for j in range(3)] for i in range(3)]
        tv = [float(tv_t[i]) for i in range(3)]
        LR = {"means": 5e-3, "quats": 1e-3, "logs": 5e-3}
        gconic = torch.randn(G, 3) * 0.5
        gmu = torch.randn(G, 2) * 0.5

        # ---------- host reference ----------
        hm, hq, hl = means0.clone().requires_grad_(True), quat0.clone().requires_grad_(True), logs0.clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [hm], "lr": LR["means"]}, {"params": [hq], "lr": LR["quats"]},
                                {"params": [hl], "lr": LR["logs"]}], betas=(B1, B2), eps=EPS)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            co, mo, _ = gbh.fwd(hm, hq, torch.exp(hl), Rv, tv, fx, fy, cx, cy)
            (co * gconic).sum().add_((mo * gmu).sum()).backward()
            opt.step()

        # ---------- device resident ----------
        def u(t):
            return ttnn.from_torch(t.reshape(-1, 1).contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        # resident params (as [G,1] cols) + Adam moments
        P = {}
        for k, src in (("mx", means0[:, 0]), ("my", means0[:, 1]), ("mz", means0[:, 2]),
                       ("qw", quat0[:, 0]), ("qx", quat0[:, 1]), ("qy", quat0[:, 2]), ("qz", quat0[:, 3]),
                       ("lx", logs0[:, 0]), ("ly", logs0[:, 1]), ("lz", logs0[:, 2])):
            P[k] = u(src)
        Mn = {k: u(torch.zeros(G)) for k in P}
        Vn = {k: u(torch.zeros(G)) for k in P}
        lr_of = {"mx": LR["means"], "my": LR["means"], "mz": LR["means"],
                 "qw": LR["quats"], "qx": LR["quats"], "qy": LR["quats"], "qz": LR["quats"],
                 "lx": LR["logs"], "ly": LR["logs"], "lz": LR["logs"]}
        gca, gcb, gcc = u(gconic[:, 0]), u(gconic[:, 1]), u(gconic[:, 2])
        gma, gmb = u(gmu[:, 0]), u(gmu[:, 1])

        for t in range(1, STEPS + 1):
            sx, sy, sz = ttnn.exp(P["lx"]), ttnn.exp(P["ly"]), ttnn.exp(P["lz"])
            cols = (P["mx"], P["my"], P["mz"], P["qw"], P["qx"], P["qy"], P["qz"], sx, sy, sz)
            conic, mu2d, cache = device_fwd_core(cols, Rv, tv, fx, fy, cx, cy)
            g = device_bwd_core(cache, gca, gcb, gcc, gma, gmb)
            # geometry returns gmx..gsz (grad wrt means/quats/scale); glogs = gscale*scale
            grads = {"mx": g["gmx"], "my": g["gmy"], "mz": g["gmz"],
                     "qw": g["gqw"], "qx": g["gqx"], "qy": g["gqy"], "qz": g["gqz"],
                     "lx": M(g["gsx"], sx), "ly": M(g["gsy"], sy), "lz": M(g["gsz"], sz)}
            for k in P:
                P[k], Mn[k], Vn[k] = adam_dev(P[k], Mn[k], Vn[k], grads[k], t, lr_of[k])

        def d(k):
            return ttnn.to_torch(P[k]).float().reshape(-1)
        dmeans = torch.stack([d("mx"), d("my"), d("mz")], -1)
        dquat = torch.stack([d("qw"), d("qx"), d("qy"), d("qz")], -1)
        dlogs = torch.stack([d("lx"), d("ly"), d("lz")], -1)

        def rel(a, b):
            return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()
        print(f"== full-resident CORE: device resident params+geom+Adam vs host, {STEPS} steps, G={G} ==")
        print(f"   means rel {rel(dmeans, hm.detach()):.3e}")
        print(f"   quats rel {rel(dquat, hq.detach()):.3e}")
        print(f"   logs  rel {rel(dlogs, hl.detach()):.3e}")
        worst = max(rel(dmeans, hm.detach()), rel(dquat, hq.detach()), rel(dlogs, hl.detach()))
        print(f"   -> worst param-drift rel {worst:.3e}  ({'PASS' if worst < 1e-3 else 'see note'})")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
