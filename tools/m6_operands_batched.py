"""Vectorize _operands over views (host, no device). The finding was that the
training loop is host-bound by `_operands` (geometry/binning/gather/theta) called in a per-view python
loop -- and most of it (quat->R, cov3d, color, sigmoid, softplus) is VIEW-INDEPENDENT yet recomputed for
every view. This hoists the view-independent work to once/iter and batches project_ewa over all N views
in one einsum; binning + gather stay a per-view loop (now with hoisted geometry). Returns the SAME stacked
operands the per-view loop produces.

Oracle: m4_train_binned._operands looped per view (forward values + autograd grads). Host-only.

Run (no device):
    podman run --rm -v $PWD:/workspace -w /workspace tt-splat:dev python3 tools/m6_operands_batched.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from spike import data, forward, geometry
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins, theta_u_gathered, _operands, K_POLY


def project_ewa_batched(means3d, cov3d_w, R_v, t_v, fx, fy, cx, cy, blur_eps=0.3, near=0.2):
    """Batched over N views. R_v[N,3,3] t_v[N,3] fx/fy/cx/cy[N]. -> mu2d[N,G,2] conic[N,G,3] depth/keep[N,G]."""
    mu_cam = torch.einsum("gj,nij->ngi", means3d, R_v) + t_v[:, None, :]      # [N,G,3]
    depth = mu_cam[..., 2]
    keep = depth > near
    z = depth.clamp(min=near)
    fx, fy = fx[:, None], fy[:, None]
    cx, cy = cx[:, None], cy[:, None]
    mu2d = torch.stack([fx * mu_cam[..., 0] / z + cx, fy * mu_cam[..., 1] / z + cy], dim=-1)  # [N,G,2]
    Sigma_cam = torch.einsum("nij,gjk,nlk->ngil", R_v, cov3d_w, R_v)          # [N,G,3,3]
    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([fx / z, zero, -fx * mu_cam[..., 0] / (z * z)], dim=-1),
        torch.stack([zero, fy / z, -fy * mu_cam[..., 1] / (z * z)], dim=-1),
    ], dim=-2)                                                                # [N,G,2,3]
    Sigma2d = torch.einsum("ngij,ngjk,nglk->ngil", J, Sigma_cam, J)          # [N,G,2,2]
    eye2 = torch.eye(2, dtype=Sigma2d.dtype, device=Sigma2d.device)
    Sigma2d = Sigma2d + blur_eps * eye2
    s00, s01, s11 = Sigma2d[..., 0, 0], Sigma2d[..., 0, 1], Sigma2d[..., 1, 1]
    det = (s00 * s11 - s01 * s01).clamp(min=1e-12)
    conic = torch.stack([s11 / det, -s01 / det, s00 / det], dim=-1)          # [N,G,3]
    return mu2d, conic, depth, keep


def operands_batched(model, cams, tmap, R, K):
    """View-independent geom once + batched project_ewa + per-view binning/gather. Returns stacked
    (theta_u[B,6,K], color_o_t[B,K,3], o_col_t[B,K,1], w_b) with B=len(cams)*T, matching cat of _operands."""
    Rm = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), Rm)
    color = forward.color_from_dc(model.color_dc)
    o = torch.sigmoid(model.opacity_raw)
    w_b = torch.nn.functional.softplus(model.w_b_raw)
    R_v = torch.stack([c.R_v for c in cams])
    t_v = torch.stack([c.t_v for c in cams])
    fx = torch.tensor([float(c.fx) for c in cams]); fy = torch.tensor([float(c.fy) for c in cams])
    cx = torch.tensor([float(c.cx) for c in cams]); cy = torch.tensor([float(c.cy) for c in cams])
    mu2d, conic, depth, keep = project_ewa_batched(model.means3d, cov, R_v, t_v, fx, fy, cx, cy)
    keo = (keep.float() * o[None, :])[..., None]                              # [N,G,1]
    color_o, o_col = keo * color[None], keo                                   # [N,G,3],[N,G,1]
    thetas, cols, ocs = [], [], []
    for v in range(len(cams)):
        idx, valid = assign_bins(mu2d[v].detach(), keep[v].detach(), tmap, R, K)
        vf = valid[..., None].float()
        theta = theta_u_gathered(conic[v][idx], mu2d[v][idx], tmap.origins, K_POLY) * valid[:, None, :].float()
        thetas.append(theta); cols.append(color_o[v][idx] * vf); ocs.append(o_col[v][idx] * vf)
    return torch.cat(thetas), torch.cat(cols), torch.cat(ocs), w_b


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def main():
    res, G, K, R, N = 128, 8000, 256, 1, 6
    cams, _ = data.load_blender("data/nerf_synthetic/ficus", "train", res=res, n=N, stride=max(1, 100 // N))
    tmap = TileMap(res, res)
    torch.manual_seed(0)
    m = GaussianModel(G, extent=1.5, seed=0)
    for p in m.parameters():
        p.requires_grad_(True)

    # oracle: per-view loop
    def per_view():
        outs = [_operands(m, c, tmap, R, K) for c in cams]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]), torch.cat([o[2] for o in outs])
    to = per_view()
    tb = operands_batched(m, cams, tmap, R, K)[:3]
    print(f"== batched _operands vs per-view loop (res={res} G={G} K={K} N={N}) ==")
    print(f"   forward match: theta {rel(tb[0], to[0]):.2e}  color {rel(tb[1], to[1]):.2e}  ocol {rel(tb[2], to[2]):.2e}")

    def grads(fn):
        for p in m.parameters():
            p.grad = None
        a, b, c = fn()[:3]
        (a.sum() + b.sum() + c.sum()).backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    go = grads(per_view)
    gb = grads(lambda: operands_batched(m, cams, tmap, R, K))
    worst = max(rel(gb[n], go[n]) for n in go)
    print(f"   grad match (worst over params): {worst:.2e}  ({'PASS' if worst < 1e-3 else 'CHECK'})")

    def timed(fn, reps=20):
        for _ in range(3):
            for p in m.parameters():
                p.grad = None
            a, b, c = fn()[:3]; (a.sum() + b.sum() + c.sum()).backward()
        t0 = time.perf_counter()
        for _ in range(reps):
            for p in m.parameters():
                p.grad = None
            a, b, c = fn()[:3]; (a.sum() + b.sum() + c.sum()).backward()
        return (time.perf_counter() - t0) / reps * 1e3
    tpv = timed(per_view)
    tba = timed(lambda: operands_batched(m, cams, tmap, R, K))
    print(f"   fwd+bwd time ({N} views): per-view {tpv:.1f} ms  ->  batched {tba:.1f} ms  ({tpv / tba:.2f}x)")


if __name__ == "__main__":
    main()
