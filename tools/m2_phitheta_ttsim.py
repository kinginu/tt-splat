"""The tile-local Phi.theta GEMM on virtual Blackhole (ttsim), diffed vs the CPU oracle.

Run INSIDE the sim container:
    docker compose run --rm sim python3 tools/m2_phitheta_ttsim.py

Computes  Q = Phi(lx,ly)[P,6] @ theta_Q^T[6,G]  for a 16x16 tile under ttsim in bf16, and diffs it
against forward.quad_form_tilelocal — the CPU reference proven in spike/tests/test_tile_local.py.
This is the matmul-centric heart of the (B) forward ("Q = Phi . theta^T is an EXACT
GEMM"), now executed on simulated Blackhole. Tile-local coords keep px^2 <= 225 so bf16 is safe.

ttnn API read from the installed source: open_device(device_id=), from_torch(layout=TILE_LAYOUT,
dtype=bfloat16, device=), matmul, to_torch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ttnn

from spike import forward


def make_case(G=64, seed=0):
    """One tile worth of gaussians far from the origin (where global coords would overflow bf16)."""
    g = torch.Generator().manual_seed(seed)
    origin = torch.tensor([400.0, 400.0])
    L = torch.randn(G, 2, 2, generator=g)
    M = L @ L.transpose(-1, -2) + 0.1 * torch.eye(2)              # SPD conics
    conic = torch.stack([M[:, 0, 0], M[:, 0, 1], M[:, 1, 1]], dim=-1)
    mu = origin + torch.rand(G, 2, generator=g) * 16.0           # means inside the tile
    ly, lx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    return conic, mu, origin, lx.reshape(-1), ly.reshape(-1)     # lx,ly: [256]


def main():
    conic, mu, origin, lx, ly = make_case()

    # CPU fp64-ish reference (the oracle the kernel must match)
    Q_ref = forward.quad_form_tilelocal(lx, ly, conic, mu, origin)     # [256, G]

    # The two GEMM operands, tile-local (origin folded into the mean inside theta_from_conic)
    Phi = forward.phi(lx, ly)                                          # A = [256, 6]
    theta_Q, _ = forward.theta_from_conic(conic, mu - origin, k=1.0)   # [G, 6]
    B = theta_Q.transpose(0, 1).contiguous()                          # [6, G]

    dev = ttnn.open_device(device_id=0)
    try:
        at = ttnn.from_torch(Phi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        bt = ttnn.from_torch(B, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        Q_sim = ttnn.to_torch(ttnn.matmul(at, bt)).float()           # [256, G]
    finally:
        ttnn.close_device(dev)

    # Separate "ttsim/kernel error" from the unavoidable "bf16 rounding" by also doing the GEMM
    # in bf16 on CPU: ttsim should land at ~the bf16 floor, not worse.
    Q_bf16_cpu = (Phi.bfloat16() @ B.bfloat16()).float()
    rel_sim = ((Q_sim - Q_ref).norm() / Q_ref.norm()).item()
    rel_bf16 = ((Q_bf16_cpu - Q_ref).norm() / Q_ref.norm()).item()
    maxabs = (Q_sim - Q_ref).abs().max().item()

    print(f"Q range [{Q_ref.min():.2f}, {Q_ref.max():.2f}] | shape {tuple(Q_sim.shape)}")
    print(f"rel-err  ttsim   vs fp64 oracle : {rel_sim:.4f}")
    print(f"rel-err  bf16CPU vs fp64 oracle : {rel_bf16:.4f}   (the bf16 floor; ttsim should ~match)")
    print(f"max|abs| ttsim   vs fp64 oracle : {maxabs:.4f}")
    ok = rel_sim < 0.05 and abs(rel_sim - rel_bf16) < 0.03
    print("tile-local Phi.theta on ttsim:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
