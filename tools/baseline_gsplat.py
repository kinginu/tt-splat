"""Standard 3DGS baseline (gsplat) — the external comparison oracle for matrix-native 3DGS.

Runs INSIDE the `baseline` container:  docker compose run --rm baseline \
    python tools/baseline_gsplat.py --res 128 --G 2000 --iters 3000 --seeds 3

This deliberately reuses the spike's EXACT data loader, view split, GaussianModel init, Adam LRs,
photometric loss, and PSNR metric (imported from spike/), and swaps ONLY the renderer to gsplat's
standard sorted-alpha rasterization. So the single variable vs `m05_spike` arm A (poly-splat + WSR)
is the renderer math — an apples-to-apples held-out PSNR comparison at matched gaussian count.

The in-repo arm "D" is the PyTorch sorted-alpha *upper bound*; THIS is the real upstream
implementation, the "once-run gsplat baseline" the matrix-native arms gate against.

NOTE (never trust LLM memory for a fast-moving API): the gsplat.rasterization() call in
render_gsplat() is written against gsplat 1.4.x. On the FIRST GPU run, sanity-check it against the
installed gsplat (`python -c "import gsplat, inspect; print(inspect.signature(gsplat.rasterization))"`)
and a 1-view render before trusting the numbers. Everything else is shared, verified spike code.
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gsplat

from spike import data, forward, metrics
from spike.device import default_device
from spike.model import GaussianModel
from spike.train import DEFAULT_LR


def render_gsplat(model, cam, bg=1.0, sh_degree=0):
    """Standard 3DGS render of `model` from `cam` via gsplat -> [H,W,3] over a white background.

    Maps the shared GaussianModel parameterization to gsplat's expected inputs:
      means3d -> means | quats (gsplat normalizes internally) | exp(log_scales) -> scales
      sigmoid(opacity_raw) -> opacities | color_from_dc (the SAME DC-SH->RGB map arm A uses) -> colors
    Camera is OpenCV world->cam (R_v, t_v) — gsplat's native convention, so viewmat is direct.

    sh_degree=0 -> DC-only RGB (the original byte-identical path; colors precomputed [G,3]).
    sh_degree>0 -> view-dependent SH: pass RAW coefficients [G,(deg+1)^2,3] (DC + rest, inria/gsplat
      layout per spike.sh) and let gsplat eval the basis and apply its own (+0.5).clamp(0). This matches
      the BH device-SH path (spike.sh.eval_sh_color) for a matched view-dependent-capacity comparison.
    """
    dev, dt = model.means3d.device, model.means3d.dtype
    viewmat = torch.eye(4, dtype=dt, device=dev)
    viewmat[:3, :3] = cam.R_v
    viewmat[:3, 3] = cam.t_v
    K = torch.tensor([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]], dtype=dt, device=dev)
    if sh_degree > 0:
        # raw SH coeffs [G, (deg+1)^2, 3] = [DC | deg1(3) | deg2(5) | deg3(7)] (spike.model layout);
        # gsplat applies the SH basis + (color+0.5).clamp_min(0), matching spike.sh.eval_sh_color.
        colors = torch.cat([model.color_dc[:, None, :], model.color_rest], dim=1)   # [G,16,3] at deg3
        sh_kw = dict(colors=colors, sh_degree=sh_degree)
    else:
        colors = forward.color_from_dc(model.color_dc)               # [G,3] in [0,1], raw RGB
        sh_kw = dict(colors=colors)

    # --- gsplat API surface (validated against gsplat 1.5.3) ---
    # Render WITHOUT a background (premultiplied colors + accumulated alpha), then composite over
    # white ourselves — matches data.composite_over_white for the GT, and sidesteps the version-
    # specific `backgrounds` batched-shape requirement.
    render_colors, render_alphas, _meta = gsplat.rasterization(
        means=model.means3d,
        quats=model.quats,
        scales=torch.exp(model.log_scales),
        opacities=torch.sigmoid(model.opacity_raw),
        viewmats=viewmat[None],                                       # [1,4,4] world->cam
        Ks=K[None],                                                   # [1,3,3]
        width=cam.W, height=cam.H,
        render_mode="RGB",
        **sh_kw,
    )
    rgb = render_colors[0]                                            # [H,W,3], premultiplied
    alpha = render_alphas[0]                                          # [H,W,1], accumulated
    return rgb + (1.0 - alpha) * bg                                   # composite over white (NeRF-synthetic)


def fit_gsplat(model, cameras, gt_images, iters, lambda_ssim=0.2, lr=None, sh_degree=0):
    lr = {**DEFAULT_LR, **(lr or {})}
    opt = torch.optim.Adam(model.param_groups(lr))
    history = []
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for cam, gt in zip(cameras, gt_images):
            total = total + metrics.loss_fn(render_gsplat(model, cam, sh_degree=sh_degree), gt, lambda_ssim)
        total = total / len(cameras)
        total.backward()
        opt.step()
        history.append(float(total.detach()))
    return history


@torch.no_grad()
def eval_psnr(model, cameras, gt_images, sh_degree=0):
    vals = [float(metrics.psnr(render_gsplat(model, cam, sh_degree=sh_degree), gt))
            for cam, gt in zip(cameras, gt_images)]
    return sum(vals) / len(vals)


def _spread(n_total, n):
    return max(1, n_total // max(1, n))


def main():
    ap = argparse.ArgumentParser(description="Standard 3DGS (gsplat) baseline — comparison oracle for matrix-native 3DGS.")
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-holdout", type=int, default=2)
    ap.add_argument("--holdout-split", default="val", help="blender split for held-out eval (val|test). "
                    "Use 'test' to match the m6 device run (test/n=25/stride=8).")
    ap.add_argument("--holdout-stride", type=int, default=0,
                    help="held-out view stride; 0 = auto _spread(100,n). Pass 8 (with --holdout-split test "
                    "--n-holdout 25) to exactly mirror the BH multi-view held-out split.")
    ap.add_argument("--sh-degree", type=int, default=0,
                    help="view-dependent SH degree (0=DC-only, the original path; 3 matches the BH device-SH run).")
    ap.add_argument("--extent", type=float, default=1.5)
    ap.add_argument("--device", default=None, help="cuda|cpu; default auto (CUDA if available)")
    ap.add_argument("--out", default="outputs/baseline_gsplat")
    args = ap.parse_args()

    dev = default_device(args.device)
    if os.environ.get("EXPECT_CUDA") == "1" and dev.type != "cuda":
        raise SystemExit("EXPECT_CUDA=1 but CUDA is unavailable — GPU not passed through; refusing CPU run.")
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(0)

    # SAME views as m05_spike (matched train/holdout split) for a like-for-like comparison.
    tr_cams, tr_imgs = data.load_blender(args.scene, "train", res=args.res, device=dev,
                                         n=args.n_train, stride=_spread(100, args.n_train))
    ho_stride = args.holdout_stride or _spread(100, args.n_holdout)
    ho_cams, ho_imgs = data.load_blender(args.scene, args.holdout_split, res=args.res, device=dev,
                                         n=args.n_holdout, stride=ho_stride)
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"[gsplat baseline] {len(tr_cams)} train + {len(ho_cams)} holdout ({args.holdout_split}/stride{ho_stride}) "
          f"@ {args.res}px | sh_degree={args.sh_degree} | device={dev} [{dev_name}] | gsplat {gsplat.__version__} | "
          f"G={args.G} iters={args.iters} seeds={args.seeds}")

    rows = []
    for seed in range(args.seeds):
        model = GaussianModel(args.G, extent=args.extent, seed=seed, device=dev)
        t0 = time.time()
        hist = fit_gsplat(model, tr_cams, tr_imgs, args.iters, sh_degree=args.sh_degree)
        dt = time.time() - t0
        tr = eval_psnr(model, tr_cams, tr_imgs, sh_degree=args.sh_degree)
        ho = eval_psnr(model, ho_cams, ho_imgs, sh_degree=args.sh_degree)
        print(f"  [gsplat seed{seed}] train {tr:.2f}  holdout {ho:.2f}  loss {hist[-1]:.4f}  ({dt:.0f}s)")
        rows.append({"seed": seed, "train_psnr": tr, "holdout_psnr": ho, "final_loss": hist[-1], "secs": dt})

    tr_m = statistics.mean(r["train_psnr"] for r in rows)
    ho_m = statistics.mean(r["holdout_psnr"] for r in rows)
    print(f"\n=== gsplat baseline (standard 3DGS, G={args.G}) ===")
    print(f"  train {tr_m:.2f} dB | holdout {ho_m:.2f} dB  (mean over {args.seeds} seed(s))")
    print(f"  -> compare holdout vs m05.json arm A (poly-splat+WSR) at the same --res/--G.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"config": vars(args), "rows": rows,
                   "train_mean": tr_m, "holdout_mean": ho_m}, f, indent=2)
    print(f"saved {args.out}.json")


if __name__ == "__main__":
    main()
