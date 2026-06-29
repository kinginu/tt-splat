"""gsplat (standard 3DGS) native-res G-sweep on the GPU box — the NVIDIA reference point
the BH matrix-native perf will later be compared against.

Per (scene, G): train gsplat fixed-count at res 800, measure train it/s + render fps + peak VRAM +
held-out/train PSNR/SSIM, and SAVE the trained model as a standard .ply (these are gsplat-standard
models, so standard viewers DO reproduce them — unlike matrix-native plys). Fixed-count (no ADC) to match
matrix-native's MCMC fixed budget. Few-view (8 train) = matched to the spike bench; at high G this overfits,
so the headline here is PERF/scaling, with quality reported-but-caveated.

    docker compose run --rm baseline python tools/gsplat_native_sweep.py \
        --scenes ficus,lego --G 30000,100000,300000 --res 800 --iters 3000

Artifacts -> outputs/gsplat_sweep/ (gitignored; not pushed): <scene>_G<G>.ply + results.json + summary.txt
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spike import data, metrics, plyio, evalcard  # noqa: E402
from spike.model import GaussianModel  # noqa: E402
import tools.baseline_gsplat as bg  # noqa: E402


def fit_gsplat_stochastic(model, cams, imgs, iters, lr=None, lambda_ssim=0.2, seed=0):
    """Standard 3DGS training: ONE random view per iteration (scales to the full 100-view set,
    unlike baseline_gsplat.fit_gsplat which is full-batch). Returns loss history."""
    lr = {**bg.DEFAULT_LR, **(lr or {})}
    opt = torch.optim.Adam(model.param_groups(lr))
    rng = random.Random(seed)
    n = len(cams)
    hist = []
    for _ in range(iters):
        i = rng.randrange(n)
        opt.zero_grad(set_to_none=True)
        loss = metrics.loss_fn(bg.render_gsplat(model, cams[i]), imgs[i], lambda_ssim)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    return hist


@torch.no_grad()
def eval_split(model, cams, imgs):
    ps = [float(metrics.psnr(bg.render_gsplat(model, c), g)) for c, g in zip(cams, imgs)]
    ss = [float(metrics.ssim(bg.render_gsplat(model, c), g)) for c, g in zip(cams, imgs)]
    return sum(ps) / len(ps), sum(ss) / len(ss)


@torch.no_grad()
def render_fps(model, cam, n=50):
    bg.render_gsplat(model, cam)  # warmup
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        bg.render_gsplat(model, cam)
    torch.cuda.synchronize()
    return n / (time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="ficus,lego")
    ap.add_argument("--G", default="1000,5000,10000",
                    help="G values per scene (ficus meaningful range: 1k,5k,10k)")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--n-train", type=int, default=100, help="train views (standard NeRF-synth = 100)")
    ap.add_argument("--n-holdout", type=int, default=25, help="held-out eval views from --eval-split")
    ap.add_argument("--eval-split", default="test", help="held-out split (standard = test, 200 frames)")
    ap.add_argument("--extent", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="gsplat", help="rough algo label for the output path")
    ap.add_argument("--card-views", type=int, default=8, help="held-out views for the eval card")
    ap.add_argument("--out", default="outputs", help="root; layout = <out>/<method>/<scene>/G<G>_res<res>")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Gs = [int(x) for x in args.G.split(",") if x]
    rows = []
    rfn = lambda mdl, cam, b: bg.render_gsplat(mdl, cam, bg=b)

    for scene in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        root = os.path.join("data/nerf_synthetic", scene)
        tr_c, tr_i = data.load_blender(root, "train", res=args.res, n=args.n_train, device=dev)
        ho_c, ho_i = data.load_blender(root, args.eval_split, res=args.res, n=args.n_holdout, device=dev)
        cv_c, cv_rgba = data.load_blender(root, args.eval_split, res=args.res, n=args.card_views,
                                          device=dev, keep_alpha=True)   # rgba for the eval card
        print(f"[{scene}] {len(tr_c)} train + {len(ho_c)} held-out ({args.eval_split}) @ {args.res}px",
              flush=True)
        for G in Gs:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            m = GaussianModel(G, extent=args.extent, seed=args.seed, device=dev)
            t0 = time.time()
            fit_gsplat_stochastic(m, tr_c, tr_i, iters=args.iters, seed=args.seed)
            torch.cuda.synchronize()
            train_s = time.time() - t0
            fps = render_fps(m, ho_c[0])
            vram = torch.cuda.max_memory_allocated() / 1e9
            ho_p, ho_s = eval_split(m, ho_c, ho_i)
            tr_p, tr_s = eval_split(m, tr_c, tr_i)
            paths = evalcard.run_paths(args.method, scene, G, args.res, iters=args.iters,
                                       seed=args.seed, root=args.out)
            os.makedirs(paths["dir"], exist_ok=True)
            plyio.save_ply(paths["ply"], m)
            card = evalcard.build(m, cv_c, cv_rgba, rfn, method=args.method, scene=scene, G=G,
                                  res=args.res, iters=args.iters, seed=args.seed, train_psnr=tr_p,
                                  perf={"it_per_s": round(args.iters / train_s, 2),
                                        "render_fps": round(fps, 1), "peak_vram_gb": round(vram, 2),
                                        "train_s": round(train_s, 1)})
            card["perceptual_25"] = {"holdout_psnr": round(ho_p, 2), "holdout_ssim": round(ho_s, 4)}
            evalcard.save(card, paths["eval_json"])
            row = {"scene": scene, "G": G, "res": args.res, "holdout_psnr": round(ho_p, 2),
                   "hf_ratio": card["perceptual"]["hf_ratio"],
                   "empty_space_leak": card["perceptual"]["empty_space_leak"],
                   "it_per_s": round(args.iters / train_s, 2), "render_fps": round(fps, 1),
                   "ply": paths["ply"], "eval_json": paths["eval_json"]}
            rows.append(row)
            print(f"[{scene} G={G}] {row['it_per_s']} it/s | {fps:.0f} fps | holdout {ho_p:.2f}dB "
                  f"| hf {row['hf_ratio']:.3f} leak {row['empty_space_leak']:.4f} "
                  f"-> {paths['eval_json']}", flush=True)

    json.dump(rows, open(os.path.join(args.out, "gsplat_sweep_rollup.json"), "w"), indent=2)
    print(f"\n# {'scene':6} {'G':>7} {'it/s':>6} {'fps':>6} {'ho_psnr':>8} {'hf_ratio':>9} {'leak':>8}")
    for r in rows:
        print(f"  {r['scene']:6} {r['G']:>7} {r['it_per_s']:>6} {r['render_fps']:>6} "
              f"{r['holdout_psnr']:>8} {r['hf_ratio']:>9} {r['empty_space_leak']:>8}")
    print(f"\nsaved {len(rows)} (ply + eval.json) under {args.out}/{args.method}/<scene>/")


if __name__ == "__main__":
    main()
