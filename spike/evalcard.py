"""Quantitative eval card — maps each known failure mode to ONE scalar, so an agent can diagnose
"why is quality low" from JSON instead of a human eyeballing the .ply in a viewer.

Renderer-agnostic: pass a `render_fn(model, cam, bg) -> [H,W,3]` (composited over constant gray `bg`,
0=black 1=white). Works for BOTH the NV gsplat baseline and the BH matrix-native renderer, so the eval card
is identical across sides (the unification point).

Channels (each validated to move in the human-judged direction on known cases before trust):
  psnr/ssim          magnitude only (kept, not diagnostic)
  hf_ratio           high-freq energy ratio render/GT (Laplacian). <1 = render blurrier = the
                     occlusion/WSR-blur ceiling signature ("green blob vs resolved leaves").
  empty_space_leak   mean model COVERAGE in GT-empty pixels (alpha<eps), via the two-bg trick
                     (coverage = 1 - |render_white - render_black|). >0 = haze in empty space
                     (the "white gaussian" bug). bg-invariant by construction.
  obj_bg_shift       mean |render_white - render_black| in GT-object pixels. >0 = object is
                     semi-transparent (under-converged coverage); cross-checks leak.
  scale_p99/_tail    giant-spike gaussians (degenerate scale).
  low_op_frac        fraction with opacity<0.05 (dead gaussians).
  haze_frac          fraction bright+desaturated+low-opacity (the haze population).
  floater_frac       fraction whose center is a spatial outlier (floaters).
  train_holdout_gap  overfit (passed in).
  lpips              perceptual distance (best single eye-proxy); null if lpips not installed.

Layout/naming convention (unified NV+BH): outputs/<method>/<scene>/G<count>_res<res>.{ply,eval.json}
"""
import datetime
import json
import os
import subprocess

import torch
import torch.nn.functional as F

from . import metrics, forward

_LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
_LPIPS = None  # lazy best-effort


def run_paths(method, scene, G, res, *, iters=None, K=None, seed=None, tag=None, root="outputs"):
    """Unified output paths for one trained model. method = rough algo+variant (gsplat / matrix_native /
    matrix_native_dw / matrix_native_bh / ...). The filename carries the numeric axes that distinguish runs so
    `ls` shows the sweep grid and runs differing in iters/K/seed don't collide; eval.json is the
    full source of truth. Layout: <root>/<method>/<scene>/G<G>_res<res>[_it<iters>][_K<K>][_s<seed>][_<tag>]."""
    parts = [f"G{G}", f"res{res}"]
    if iters is not None:
        parts.append(f"it{iters}")
    if K is not None:
        parts.append(f"K{K}")
    if seed:                                 # omit the default seed 0
        parts.append(f"s{seed}")
    if tag:
        parts.append(str(tag))
    d = os.path.join(root, method, scene)
    stem = os.path.join(d, "_".join(parts))
    return {"dir": d, "ply": stem + ".ply", "eval_json": stem + ".eval.json", "stem": stem}


def _git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _hf_energy(img):  # img [H,W,3] -> scalar mean Laplacian^2 energy
    x = img.permute(2, 0, 1).unsqueeze(1)                       # [3,1,H,W]
    lap = F.conv2d(x, _LAP.to(img), padding=1)
    return lap.pow(2).mean()


def _lpips(render, gt):
    global _LPIPS
    try:
        if _LPIPS is None:
            import lpips
            _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(render.device).eval()

        def chw(t):
            return (t.permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
        with torch.no_grad():
            return float(_LPIPS(chw(render), chw(gt)).item())
    except Exception:
        return None


@torch.no_grad()
def perceptual(model, cams, gts_rgba, render_fn, eps_alpha=0.05):
    """Per-view render metrics averaged over held-out cams. gts_rgba: list of [H,W,4] (straight rgb+alpha)."""
    acc = {k: 0.0 for k in ("psnr", "ssim", "hf_ratio", "empty_space_leak", "obj_bg_shift", "lpips")}
    nlp = 0
    for cam, rgba in zip(cams, gts_rgba):
        a = rgba[..., 3:4]
        gt = rgba[..., :3] * a + (1 - a)                       # composite over white
        rw = render_fn(model, cam, 1.0)
        rb = render_fn(model, cam, 0.0)
        cov = 1.0 - (rw - rb).mean(-1).clamp(0, 1)             # per-pixel coverage in [0,1]
        empty = (a[..., 0] < eps_alpha)
        obj = (a[..., 0] > 1 - eps_alpha)
        acc["psnr"] += float(metrics.psnr(rw, gt))
        acc["ssim"] += float(metrics.ssim(rw, gt))
        acc["hf_ratio"] += float(_hf_energy(rw) / _hf_energy(gt).clamp(min=1e-9))
        acc["empty_space_leak"] += float(cov[empty].mean()) if empty.any() else 0.0
        acc["obj_bg_shift"] += float((rw - rb).abs().mean(-1)[obj].mean()) if obj.any() else 0.0
        lp = _lpips(rw, gt)
        if lp is not None:
            acc["lpips"] += lp
            nlp += 1
    n = len(cams)
    out = {k: round(v / n, 5) for k, v in acc.items() if k != "lpips"}
    out["lpips"] = round(acc["lpips"] / nlp, 5) if nlp else None
    return out


@torch.no_grad()
def param_stats(model):
    """Render-free degeneracy stats straight from the gaussian params (also readable from a .ply)."""
    scales = torch.exp(model.log_scales)
    smax = scales.max(-1).values
    op = torch.sigmoid(model.opacity_raw)
    color = forward.color_from_dc(model.color_dc).clamp(0, 1)
    bright = color.mean(-1)
    sat = color.max(-1).values - color.min(-1).values
    med = model.means3d.median(0).values
    d = (model.means3d - med).norm(dim=-1)
    dmed = d.median().clamp(min=1e-9)
    return {
        "G": int(model.means3d.shape[0]),
        "scale_p99": round(float(torch.quantile(smax, 0.99)), 5),
        "scale_tail_frac": round(float((smax > 5 * smax.median()).float().mean()), 5),
        "low_op_frac": round(float((op < 0.05).float().mean()), 5),
        "haze_frac": round(float(((op < 0.15) & (bright > 0.6) & (sat < 0.15)).float().mean()), 5),
        "floater_frac": round(float((d > 3 * dmed).float().mean()), 5),
    }


def build(model, cams, gts_rgba, render_fn, *, method, scene, G, res, iters=None, seed=None,
          K=None, arm=None, tag=None, train_psnr=None, perf=None, eps_alpha=0.05):
    """Assemble the full self-describing eval card dict (config + provenance + metrics). Does not write."""
    card = {"method": method, "scene": scene, "G": int(G), "res": int(res), "iters": iters,
            "seed": seed, "K": K, "arm": arm, "tag": tag,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit()}
    p = perceptual(model, cams, gts_rgba, render_fn, eps_alpha)
    card["perceptual"] = p
    if train_psnr is not None:
        card["train_holdout_gap"] = round(float(train_psnr) - p["psnr"], 3)
    card["params"] = param_stats(model)
    if perf:
        card["perf"] = perf
    return card


def save(card, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(card, open(path, "w"), indent=2)
    return path
