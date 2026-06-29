"""The (B) backward hot path on ttsim — its three transposed GEMMs run on simulated Blackhole,
diffed vs the autograd reference. Completes "forward + backward match the CPU reference on ttsim".

Run INSIDE the sim container:
    docker compose run --rm sim python3 tools/m2_backward_ttsim.py

The hand-derived backward (spike/backward.py, gradcheck-verified) is, for the hot path, THREE
transposed GEMMs plus cheap pointwise/reduction glue:
    gw     = gnum @ color^T              [P,3]@[3,G] -> [P,G]
    gcolor = w^T  @ gnum                 [G,P]@[P,3] -> [G,3]
    gtheta = gu^T @ Phi                  [G,P]@[P,6] -> [G,6]
We run those three GEMMs on ttsim (bf16) and assemble the glue on the host (fp32; it is the
gradcheck-verified, hardware-uninteresting part), then diff all four leaf grads against autograd.
This isolates the one ttsim question: are the BACKWARD GEMMs correct on sim-Blackhole? (functional.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ttnn

from spike import backward, forward

K = 4.0


def make_case(G=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    origin = torch.tensor([400.0, 400.0])
    std = 1.0 + torch.rand(G, 2, generator=g) * 2.0
    conic = torch.stack([1.0 / std[:, 0] ** 2, torch.zeros(G), 1.0 / std[:, 1] ** 2], dim=-1)
    mu = origin + torch.rand(G, 2, generator=g) * 16.0
    color = torch.rand(G, 3, generator=g)
    o = torch.rand(G, generator=g) * 0.8 + 0.1
    w_b = torch.tensor(0.05)
    c_b = torch.ones(3)
    ly, lx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    Phi = forward.phi(lx.reshape(-1), ly.reshape(-1))                  # [P,6]
    _, theta_u = forward.theta_from_conic(conic, mu - origin, k=K)     # [G,6]
    return theta_u, o, color, w_b, Phi, c_b


def mm(dev, A, B):
    """A@B on ttsim in bf16 (operands pre-transposed on host)."""
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return ttnn.to_torch(ttnn.matmul(up(A), up(B))).float()


def main():
    theta_u, o, color, w_b, Phi, c_b = make_case()
    torch.manual_seed(1)
    gC = torch.randn(Phi.shape[0], 3)                                 # upstream grad dL/dC

    # --- autograd reference grads (the truth) ---
    leaves = [t.clone().requires_grad_(True) for t in (theta_u, o, color, w_b)]
    C = backward.wsr_polysplat_ref(leaves[0], leaves[1], leaves[2], leaves[3], Phi, c_b)
    (C * gC).sum().backward()
    gtheta_ref, go_ref, gcolor_ref, gwb_ref = (l.grad for l in leaves)

    # --- manual backward, with the THREE GEMMs on ttsim (host fp32 for the gradcheck'd glue) ---
    u = Phi @ theta_u.t()
    relu_u = u.clamp(min=0.0)
    w_geo = relu_u * relu_u
    w = o[None, :] * w_geo
    den = w.sum(1, keepdim=True) + w_b
    Cf = (w @ color + w_b * c_b[None, :]) / den
    inv = 1.0 / den
    gnum = gC * inv
    gden = -(gC * Cf).sum(1, keepdim=True) * inv

    dev = ttnn.open_device(device_id=0)
    try:
        gw = mm(dev, gnum, color.t()) + gden                         # GEMM #1 (ttsim) + glue
        gw_geo = gw * o[None, :]
        gu = gw_geo * (2.0 * relu_u)
        gtheta = mm(dev, gu.t(), Phi)                                # GEMM #2 (ttsim)
        gcolor = mm(dev, w.t(), gnum)                                # GEMM #3 (ttsim)
    finally:
        ttnn.close_device(dev)
    go = (gw * w_geo).sum(0)
    gw_b = (gnum * c_b[None, :]).sum() + gden.sum()

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()

    rels = {"gtheta": rel(gtheta, gtheta_ref), "go": rel(go, go_ref),
            "gcolor": rel(gcolor, gcolor_ref), "gw_b": rel(gw_b, gwb_ref)}
    for k, v in rels.items():
        print(f"  rel-err {k:7s} ttsim-backward vs autograd : {v:.4f}")
    ok = all(v < 0.05 for v in rels.values())
    print("backward hot-path (3 transposed GEMMs) on ttsim:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
