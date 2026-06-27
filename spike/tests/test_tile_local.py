"""Tile-local Phi.theta GEMM.

Two things the existing test_transform does NOT cover, both required before the Metalium kernel:
  1. the tile-local form (origin folded into theta) is EXACTLY equal to the global quad_form, and
  2. it is the form that SURVIVES bf16 — global pixel coords overflow/cancel in the mantissa and
     the kernel "dies"; tile-local (px in 0..15, px^2 <= 225) stays accurate. This test makes the
     "mandatory for bf16" claim concrete with numbers, so the kernel has a verified oracle.
"""
import _bootstrap  # noqa: F401
import torch

from spike import forward

DT = torch.float64


def _tile_grid(n=16, dtype=DT):
    """Local pixel coords for an n x n tile, flattened -> lx[P], ly[P] (P = n*n)."""
    ly, lx = torch.meshgrid(torch.arange(n, dtype=dtype), torch.arange(n, dtype=dtype), indexing="ij")
    return lx.reshape(-1), ly.reshape(-1)


def test_tilelocal_equals_global_fp64():
    """Q via the tile-local GEMM == raw global quad_form to ~1e-9, over a 16x16 tile, many gaussians."""
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        origin = torch.randint(0, 64, (2,)).to(DT) * 16.0          # a real tile origin (multiple of 16)
        G = 5
        L = torch.randn(G, 2, 2, generator=g, dtype=DT)
        M = L @ L.transpose(-1, -2) + 0.1 * torch.eye(2, dtype=DT)  # SPD conics
        conic = torch.stack([M[:, 0, 0], M[:, 0, 1], M[:, 1, 1]], dim=-1)
        mu = origin + torch.rand(G, 2, generator=g, dtype=DT) * 16.0  # means inside the tile

        lx, ly = _tile_grid()
        px, py = origin[0] + lx, origin[1] + ly                    # the same pixels, global coords

        Q_local = forward.quad_form_tilelocal(lx, ly, conic, mu, origin)
        Q_global = forward.quad_form(px, py, mu, conic)
        assert torch.allclose(Q_local, Q_global, atol=1e-9), (Q_local - Q_global).abs().max().item()


def _q_global_gemm_bf16(px, py, conic, mu):
    """The NAIVE form a kernel would use without tiling: Phi(global) @ theta, evaluated in bf16."""
    theta_Q, _ = forward.theta_from_conic(conic, mu, k=1.0)
    Phi = forward.phi(px, py)
    return (Phi.bfloat16() @ theta_Q.bfloat16().transpose(-1, -2)).to(DT)


def test_bf16_global_dies_tilelocal_survives():
    """The crux: in bf16 the global GEMM loses Q entirely; tile-local stays within splat tolerance."""
    # a tile far from the origin (where global coords are large) — typical for an 800px render
    origin = torch.tensor([400.0, 400.0], dtype=DT)
    conic = torch.tensor([[0.08, 0.02, 0.05]], dtype=DT)
    mu = origin + torch.tensor([[7.3, 4.6]], dtype=DT)             # gaussian centered inside the tile
    lx, ly = _tile_grid()
    px, py = origin[0] + lx, origin[1] + ly

    Q_ref = forward.quad_form(px, py, mu, conic)                   # fp64 truth

    # tile-local GEMM in bf16
    mu_local = mu - origin
    theta_Q, _ = forward.theta_from_conic(conic, mu_local, k=1.0)
    Q_local_bf16 = (forward.phi(lx, ly).bfloat16() @ theta_Q.bfloat16().transpose(-1, -2)).to(DT)

    # naive global GEMM in bf16
    Q_global_bf16 = _q_global_gemm_bf16(px, py, conic, mu)

    err_local = (Q_local_bf16 - Q_ref).abs().max().item()
    err_global = (Q_global_bf16 - Q_ref).abs().max().item()

    # tile-local keeps Q usable (well under k=4 so the poly-splat weight is meaningful);
    # global is destroyed by mantissa cancellation (error >> the signal it is trying to represent).
    assert err_local < 0.1, f"tile-local bf16 error too high: {err_local}"
    assert err_global > 1.0, f"expected global bf16 to blow up, got {err_global}"
    assert err_global > 20 * err_local, f"global={err_global} not >> local={err_local}"


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
