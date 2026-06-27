"""gsplat (standard 3DGS) ACCURACY on a real COLMAP scene — trains to convergence + held-out PSNR/SSIM.

Fills the GPU-side real-scene accuracy gap (TODO F1): playroom-on-3090 had perf only. Loads a COLMAP
scene, splits train/test (every `--test-every`-th view = held-out, the 3DGS llffhold convention),
inits gaussians from the SfM points, trains gsplat stochastically, evaluates held-out PSNR/SSIM +
hf_ratio + param stats, and emits an eval-card (same schema as the BH matrix_native_bh cards; real scenes have
no alpha so empty_space_leak / obj_bg_shift are null).

    docker compose run --rm baseline python tools/gsplat_real_train.py \
        --root data/db/playroom --downscale 4 --G 100000,300000 --iters 30000
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spike import data, metrics, plyio, evalcard  # noqa: E402
import tools.baseline_gsplat as bg  # noqa: E402
from tools.gsplat_native_sweep import fit_gsplat_stochastic  # noqa: E402
from tools.gsplat_real_perf import init_from_points  # noqa: E402


@torch.no_grad()
def eval_real(model, cams, imgs):
    ps = [float(metrics.psnr(bg.render_gsplat(model, c), g)) for c, g in zip(cams, imgs)]
    ss = [float(metrics.ssim(bg.render_gsplat(model, c), g)) for c, g in zip(cams, imgs)]
    hf = [float(evalcard._hf_energy(bg.render_gsplat(model, c)) / evalcard._hf_energy(g).clamp(min=1e-9))
          for c, g in zip(cams, imgs)]
    return sum(ps) / len(ps), sum(ss) / len(ss), sum(hf) / len(hf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/db/playroom")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--G", default="100000,300000")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="gsplat")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cams, imgs = data.load_colmap(args.root, downscale=args.downscale, device=dev)
    xyz, _ = data.load_colmap_points(args.root)
    scene = os.path.basename(args.root.rstrip("/"))
    te = list(range(0, len(cams), args.test_every))
    tr = [i for i in range(len(cams)) if i not in set(te)]
    tr_c, tr_i = [cams[i] for i in tr], [imgs[i] for i in tr]
    te_c, te_i = [cams[i] for i in te], [imgs[i] for i in te]
    H, W = cams[0].H, cams[0].W
    print(f"[{scene}] {len(tr_c)} train / {len(te_c)} test @ {W}x{H} | {len(xyz)} SfM points", flush=True)

    for G in [int(x) for x in args.G.split(",") if x]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m, _ = init_from_points(G, xyz, dev)
        t0 = time.time()
        fit_gsplat_stochastic(m, tr_c, tr_i, iters=args.iters, seed=args.seed)
        torch.cuda.synchronize()
        train_s = time.time() - t0
        ho_p, ho_s, ho_hf = eval_real(m, te_c, te_i)
        tr_p, _, _ = eval_real(m, tr_c[:20], tr_i[:20])
        vram = torch.cuda.max_memory_allocated() / 1e9
        paths = evalcard.run_paths(args.method, scene, G, f"{W}x{H}", iters=args.iters,
                                   seed=args.seed, root=args.out)
        os.makedirs(paths["dir"], exist_ok=True)
        plyio.save_ply(paths["ply"], m)
        card = {"method": args.method, "scene": scene, "G": G, "res": f"{W}x{H}", "iters": args.iters,
                "seed": args.seed, "real_scene": True, "git_commit": evalcard._git_commit(),
                "perceptual": {"psnr": round(ho_p, 3), "ssim": round(ho_s, 4), "hf_ratio": round(ho_hf, 4),
                               "empty_space_leak": None, "obj_bg_shift": None, "lpips": None},
                "train_psnr": round(tr_p, 3), "train_holdout_gap": round(tr_p - ho_p, 3),
                "params": evalcard.param_stats(m),
                "perf": {"it_per_s": round(args.iters / train_s, 2), "train_s": round(train_s, 1),
                         "peak_vram_gb": round(vram, 2)}}
        evalcard.save(card, paths["eval_json"])
        print(f"[{scene} G={G}] held-out {ho_p:.2f} dB / {ho_s:.3f} ssim / hf {ho_hf:.3f} | "
              f"train {tr_p:.2f} (gap {tr_p-ho_p:.2f}) | {args.iters/train_s:.1f} it/s | "
              f"VRAM {vram:.2f} GB | {train_s/60:.1f} min -> {paths['eval_json']}", flush=True)


if __name__ == "__main__":
    main()
