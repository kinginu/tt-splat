"""The full (B) forward HOT-PATH (Phi.theta -> poly-splat -> WSR) on ttsim,
diffed vs the CPU reference (arm A). One device session.

Run INSIDE the sim container:
    docker compose run --rm sim python3 tools/m2_forward_ttsim.py

The per-(pixel x gaussian) hot path of the (B) forward, all on simulated Blackhole:
  (ii)  Q     = Phi(lx,ly)[P,6] @ theta^T[6,G]                 (tile-local, exact GEMM)
  (iii) w_geo = relu(1 - Q/k)^2                                (poly-splat; ttnn elementwise)
  (iv)  WSR   = (w_geo @ (o*color) + w_b*c_b) / (w_geo @ o + w_b)
        folding o into the matmul operands keeps the numerator (Sum w c) and denominator (Sum w)
        as GEMMs — the thesis claim that WSR is matmul-shaped. -> C[P,3].

Geometry (project_ewa: quat->conic, EWA) stays on the host (G-setup, cheap); only the
P x G hot path runs on device. Oracle = the same spike functions on CPU (forward.* + arms.blend_A).
ttnn API read from the installed source.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ttnn

from spike import arms, forward

K = 4.0


def make_case(G=64, seed=0):
    """A 16x16 tile of pixels + G axis-aligned gaussians of a few-pixel radius inside it."""
    g = torch.Generator().manual_seed(seed)
    origin = torch.tensor([400.0, 400.0])
    std = 1.0 + torch.rand(G, 2, generator=g) * 2.0                  # 1..3 px radius
    conic = torch.stack([1.0 / std[:, 0] ** 2,                       # a = 1/sx^2
                         torch.zeros(G),                             # b = 0 (axis-aligned)
                         1.0 / std[:, 1] ** 2], dim=-1)              # c = 1/sy^2
    mu = origin + torch.rand(G, 2, generator=g) * 16.0
    color = torch.rand(G, 3, generator=g)                           # [0,1]
    o = torch.rand(G, generator=g) * 0.8 + 0.1                      # opacity in [0.1,0.9]
    w_b = 0.05
    c_b = torch.ones(3)
    ly, lx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    return origin, conic, mu, color, o, w_b, c_b, lx.reshape(-1), ly.reshape(-1)


def cpu_reference(origin, conic, mu, color, o, w_b, c_b, lx, ly):
    Q = forward.quad_form_tilelocal(lx, ly, conic, mu, origin)
    w_geo = forward.poly_splat_wgeo(Q, K)
    return arms.blend_A(w_geo, torch.logit(o), color, torch.tensor(w_b), c_b)   # [P,3]


def ttsim_forward(dev, origin, conic, mu, color, o, w_b, c_b, lx, ly):
    P = lx.shape[0]
    # operands built on host (G-setup): the GEMM matrices and the folded-o color/weight columns
    Phi = forward.phi(lx, ly)                                        # [P,6]
    theta_Q, _ = forward.theta_from_conic(conic, mu - origin, k=1.0) # [G,6]
    color_o = o[:, None] * color                                    # [G,3]  (o folded in)
    o_col = o[:, None]                                              # [G,1]
    bias_num = (w_b * c_b)[None, :].expand(P, 3).contiguous()       # [P,3]

    def up(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    Phi_t, thT_t = up(Phi), up(theta_Q.transpose(0, 1).contiguous())
    Q_t = ttnn.matmul(Phi_t, thT_t)                                 # (ii) Q [P,G]
    u_t = ttnn.rsub(ttnn.mul(Q_t, 1.0 / K), 1.0)                    # 1 - Q/k
    w_geo_t = ttnn.square(ttnn.relu(u_t))                          # (iii) w_geo [P,G]
    num_t = ttnn.add(ttnn.matmul(w_geo_t, up(color_o)), up(bias_num))   # (iv) Sum w c + w_b c_b -> [P,3]
    den_t = ttnn.add(ttnn.matmul(w_geo_t, up(o_col)), w_b)         #      Sum w   + w_b        -> [P,1]
    try:
        C = ttnn.to_torch(ttnn.div(num_t, den_t)).float()         # normalize on device (broadcast [P,1])
        where = "device (incl. normalize)"
    except Exception:                                              # broadcast div unsupported -> host normalize
        num = ttnn.to_torch(num_t).float()
        den = ttnn.to_torch(den_t).float()[:, :1]
        C, where = num / den, "device GEMMs+poly; host normalize"
    return C, where


def main():
    case = make_case()
    C_ref = cpu_reference(*case)
    dev = ttnn.open_device(device_id=0)
    try:
        C_sim, where = ttsim_forward(dev, *case)
    finally:
        ttnn.close_device(dev)

    rel = ((C_sim - C_ref).norm() / C_ref.norm()).item()
    maxabs = (C_sim - C_ref).abs().max().item()
    print(f"arm-A forward on ttsim ({where})")
    print(f"  C range [{C_ref.min():.3f}, {C_ref.max():.3f}] | shape {tuple(C_sim.shape)}")
    print(f"  rel-err ttsim vs CPU reference : {rel:.4f}")
    print(f"  max|abs|                       : {maxabs:.4f}")
    ok = rel < 0.05
    print("full (B) forward hot-path on ttsim:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
