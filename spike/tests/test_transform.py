"""Test 1: lock the conic->theta monomial identity Q == Phi.theta_Q and
(1 - Q/k) == Phi.theta_u to ~1e-9 in fp64. Kills the factor-of-2 / sign / fold traps."""
import _bootstrap  # noqa: F401
import torch

from spike import forward

DT = torch.float64


def _f(x):
    return torch.tensor(x, dtype=DT)


def test_worked_example():
    conic = _f([[0.08, 0.02, 0.05]])
    mu2d = _f([[3.5, 4.5]])
    k = 4.0
    theta_Q, theta_u = forward.theta_from_conic(conic, mu2d, k)
    expected_Q = _f([0.08, 0.05, 0.04, -0.74, -0.59, 2.6225])
    assert torch.allclose(theta_Q[0], expected_Q, atol=1e-12), theta_Q[0]

    px, py = _f([5.0]), _f([6.0])
    Q = forward.quad_form(px, py, mu2d, conic)
    assert abs(Q.item() - 0.3825) < 1e-9, Q.item()

    Phi = forward.phi(px, py)
    assert abs((Phi @ theta_Q[0]).item() - Q.item()) < 1e-9
    u = 1.0 - Q / k
    assert abs((Phi @ theta_u[0]).item() - u.item()) < 1e-9
    assert abs(u.item() - 0.904375) < 1e-9


def test_random_spd():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        L = torch.randn(2, 2, generator=g, dtype=DT)
        M = L @ L.T + 0.1 * torch.eye(2, dtype=DT)            # SPD conic
        conic = torch.stack([M[0, 0], M[0, 1], M[1, 1]]).reshape(1, 3)
        mu2d = torch.randn(1, 2, generator=g, dtype=DT) * 5
        k = 3.7
        px = torch.randn(8, generator=g, dtype=DT) * 6
        py = torch.randn(8, generator=g, dtype=DT) * 6

        Q = forward.quad_form(px, py, mu2d, conic)[:, 0]      # [8]
        theta_Q, theta_u = forward.theta_from_conic(conic, mu2d, k)
        Phi = forward.phi(px, py)                             # [8,6]
        assert torch.allclose(Phi @ theta_Q[0], Q, atol=1e-9), (Phi @ theta_Q[0] - Q).abs().max()
        assert torch.allclose(Phi @ theta_u[0], 1.0 - Q / k, atol=1e-9)


def test_center_is_zero():
    conic = _f([[0.13, -0.02, 0.09]])
    mu2d = _f([[7.0, 3.0]])
    Q = forward.quad_form(mu2d[:, 0], mu2d[:, 1], mu2d, conic)
    assert abs(Q.item()) < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module(globals()))
