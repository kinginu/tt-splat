"""FULL manual (no-autograd) operands fwd+bwd on host, verified vs autograd. Removes the
torch-autograd cost that the breakdown showed caps the trainer. Chain: geometry (geom_bwd, verified)
+ color/opacity activations + binning + gather + theta_u build, with a hand-written backward composing
geometry-bwd + theta-build-bwd + gather scatter + activation grads. Once verified, plugs into a
no-autograd trainer with FastRender (device render) -- every gradient manual, no torch graph.

Oracle: torch autograd of the same forward (per-param grads). Host-only.

Run (no device):
    podman run --rm -v $PWD:/workspace -w /workspace tt-splat:dev python3 tools/operands_manual.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from spike import data, forward, geometry, sh
from spike.model import GaussianModel
from m4_train_binned import TileMap, assign_bins, theta_u_gathered, K_POLY
import geom_bwd as gb

C0 = sh.C0


def cams_scalars(cam):
    Rv = [[float(cam.R_v[i, j]) for j in range(3)] for i in range(3)]
    tv = [float(cam.t_v[i]) for i in range(3)]
    return Rv, tv, float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)


def fwd_manual(model, cams, tmap, R, K, bins=None):
    """Manual forward. Returns stacked (theta_u, color_o_t, o_col_t) + a cache for the manual backward.
    bins: optional list of (idx, valid) per view to REUSE (every-N-iters stale binning); else recompute."""
    scale = torch.exp(model.log_scales)
    color = (0.5 + C0 * model.color_dc).clamp(min=0.0)                 # [G,3]
    cmask = (0.5 + C0 * model.color_dc > 0).float()
    o = torch.sigmoid(model.opacity_raw)                              # [G]
    thetas, cols, ocs = [], [], []
    pv = []
    for vi, cam in enumerate(cams):
        Rv, tv, fx, fy, cx, cy = cams_scalars(cam)
        conic, mu2d, gcache = gb.fwd(model.means3d, model.quats, scale, Rv, tv, fx, fy, cx, cy)
        keep = gcache["zmask"]                                        # 1.0 where mcz>near
        keo = keep * o                                               # [G]
        color_o = keo[:, None] * color                              # [G,3]
        o_col = keo[:, None]                                        # [G,1]
        if bins is None:
            idx, valid = assign_bins(mu2d.detach(), (keep > 0.5), tmap, R, K)
        else:
            idx, valid = bins[vi]                                   # reuse stale bins
        vf = valid[..., None].float()
        conic_t, mu_t = conic[idx], mu2d[idx]
        theta = theta_u_gathered(conic_t, mu_t, tmap.origins, K_POLY) * valid[:, None, :].float()
        thetas.append(theta); cols.append(color_o[idx] * vf); ocs.append(o_col[idx] * vf)
        pv.append(dict(gcache=gcache, idx=idx, valid=valid, conic_t=conic_t, mu_t=mu_t,
                       keo=keo, keep=keep, color_o=color_o))
    cache = dict(pv=pv, color=color, cmask=cmask, o=o, scale=scale, origins=tmap.origins, K=K, G=model.G)
    return torch.cat(thetas), torch.cat(cols), torch.cat(ocs), cache


def _theta_bwd(gtheta, conic_t, mu_t, origins, valid):
    """gtheta[T,6,K] -> gconic_t[T,K,3], gmu_t[T,K,2] (host torch formulas)."""
    T, K = conic_t.shape[0], conic_t.shape[1]
    valid6 = valid[:, None, :].expand(T, 6, K).float()
    gpoly = (gtheta * valid6).transpose(1, 2)                        # [T,K,6]
    gtQ = gpoly * (-1.0 / K_POLY)
    g0, g1, g2, g3, g4, g5 = (gtQ[..., j] for j in range(6))
    a, b, c = conic_t[..., 0], conic_t[..., 1], conic_t[..., 2]
    mu = mu_t - origins[:, None, :]
    mux, muy = mu[..., 0], mu[..., 1]
    ga = g0 + g3 * (-2 * mux) + g5 * mux * mux
    gb_ = 2 * g2 + g3 * (-2 * muy) + g4 * (-2 * mux) + g5 * (2 * mux * muy)
    gc = g1 + g4 * (-2 * muy) + g5 * muy * muy
    gmux = g3 * (-2 * a) + g4 * (-2 * b) + g5 * (2 * a * mux + 2 * b * muy)
    gmuy = g3 * (-2 * b) + g4 * (-2 * c) + g5 * (2 * b * mux + 2 * c * muy)
    return torch.stack([ga, gb_, gc], -1), torch.stack([gmux, gmuy], -1)


def bwd_manual(cache, gtheta_all, gcol_all, goc_all):
    pv, color, cmask, o, scale = cache["pv"], cache["color"], cache["cmask"], cache["o"], cache["scale"]
    G, K = cache["G"], cache["K"]
    gmeans = torch.zeros(G, 3); gquat = torch.zeros(G, 4); gscale = torch.zeros(G, 3)
    gcolor = torch.zeros(G, 3); go = torch.zeros(G)
    T = pv[0]["valid"].shape[0]
    for v, p in enumerate(pv):
        sl = slice(v * T, (v + 1) * T)
        gtheta, gcol, goc = gtheta_all[sl], gcol_all[sl], goc_all[sl]
        idx, valid, vf = p["idx"], p["valid"], p["valid"][..., None].float()
        gconic_t, gmu_t = _theta_bwd(gtheta, p["conic_t"], p["mu_t"], cache["origins"], valid)
        flat = idx.reshape(-1)
        gconic = torch.zeros(G, 3).index_add_(0, flat, gconic_t.reshape(-1, 3))
        gmu2d = torch.zeros(G, 2).index_add_(0, flat, gmu_t.reshape(-1, 2))
        gcolor_o = torch.zeros(G, 3).index_add_(0, flat, (gcol * vf).reshape(-1, 3))
        go_col = torch.zeros(G, 1).index_add_(0, flat, (goc * vf).reshape(-1, 1))
        # geometry bwd (per view)
        gm, gq, gs = gb.bwd(p["gcache"], gconic[:, 0], gconic[:, 1], gconic[:, 2], gmu2d[:, 0], gmu2d[:, 1])
        gmeans += gm; gquat += gq; gscale += gs
        # color/opacity bwd
        keo = p["keo"]; keep = p["keep"]
        gkeo = (gcolor_o * color).sum(-1) + go_col[:, 0]
        gcolor += gcolor_o * keo[:, None]
        go += gkeo * keep
    glog_scales = gscale * scale                                      # scale = exp(log)
    gcolor_dc = gcolor * C0 * cmask
    gopacity_raw = go * o * (1 - o)
    return dict(means3d=gmeans, quats=gquat, log_scales=glog_scales,
                color_dc=gcolor_dc, opacity_raw=gopacity_raw)


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

    theta_all, col_all, oc_all, cache = fwd_manual(m, cams, tmap, R, K)
    gth = torch.randn_like(theta_all); gco = torch.randn_like(col_all); goc = torch.randn_like(oc_all)
    # autograd oracle
    for p in m.parameters():
        p.grad = None
    (theta_all * gth).sum().add_((col_all * gco).sum()).add_((oc_all * goc).sum()).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    # manual
    with torch.no_grad():
        theta_all2, col_all2, oc_all2, cache2 = fwd_manual(m, cams, tmap, R, K)
        gman = bwd_manual(cache2, gth, gco, goc)

    print(f"== full manual operands backward vs autograd (res={res} G={G} K={K} N={N}) ==")
    worst = 0.0
    for name in ("means3d", "quats", "log_scales", "color_dc", "opacity_raw"):
        r = rel(gman[name], ref[name]); worst = max(worst, r)
        print(f"   g{name:12s} rel {r:.3e}")
    print(f"   -> worst rel {worst:.3e}  ({'PASS' if worst < 1e-3 else 'FAIL'})")


if __name__ == "__main__":
    main()
