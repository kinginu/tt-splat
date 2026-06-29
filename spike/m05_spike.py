"""Experiment driver: run the arms, the under-fit pre-flight, and emit the
go/extend/no-go verdict on the locked depth-free WSR.

  python -m spike.m05_spike --res 64 --G 2000 --iters 600 --arms A,B,C0,C,D --seeds 3

Primary metric = train-view reconstruction PSNR (held-out reported secondary). Ceiling = arm D
(apples-to-apples). Attribution: C vs C0 (view-dependence) and C0 vs A (capacity).
"""
import argparse
import json
import os
import statistics
import time

# Deterministic cuBLAS for reproducible matmuls under use_deterministic_algorithms on CUDA
# (must be set before torch initializes cuBLAS; harmless on CPU).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from . import data, train
from .device import default_device
from .model import GaussianModel
from .render import ARMS

# per-gaussian param count (SZ/RV add only global scalars, so same as A's 14)
PARAMS_PER_GAUSSIAN = {"A": 14, "B": 14, "C0": 22, "C": 22, "SZ": 14, "RV": 14, "D": 14}
CANDIDATE_ARMS = ("A", "B", "C", "SZ", "RV")   # real candidates (C0 = capacity control, D = ceiling)


def _spread(n_total, n):
    return max(1, n_total // max(1, n))


def run_arm(arm, tr_cams, tr_imgs, ho_cams, ho_imgs, G, iters, seed, extent, lr=None, device="cpu",
            sh_degree=0):
    model = GaussianModel(G, extent=extent, seed=seed, device=device)
    t0 = time.time()
    hist = train.fit(model, tr_cams, tr_imgs, arm, iters=iters, lr=lr, sh_degree=sh_degree)
    dt = time.time() - t0
    tr_psnr = train.eval_psnr(model, tr_cams, tr_imgs, arm, sh_degree=sh_degree)
    ho_psnr = train.eval_psnr(model, ho_cams, ho_imgs, arm, sh_degree=sh_degree) if ho_cams else float("nan")
    return {"arm": arm, "G": G, "seed": seed, "train_psnr": tr_psnr, "holdout_psnr": ho_psnr,
            "final_loss": hist[-1], "params_per_g": PARAMS_PER_GAUSSIAN[arm], "secs": dt}


def preflight(tr_cams, tr_imgs, Gs, iters, seed, extent, device="cpu"):
    rows = []
    for G in Gs:
        m = GaussianModel(G, extent=extent, seed=seed, device=device)
        train.fit(m, tr_cams, tr_imgs, "D", iters=iters)
        rows.append({"G": G, "D_train_psnr": train.eval_psnr(m, tr_cams, tr_imgs, "D")})
        print(f"  [preflight] D @ G={G:>6}  train PSNR {rows[-1]['D_train_psnr']:.2f} dB")
    return rows


def aggregate(rows):
    """Group per-arm rows across seeds -> mean/std of train+holdout PSNR."""
    out = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        tr = [r["train_psnr"] for r in rs]
        ho = [r["holdout_psnr"] for r in rs]
        out[arm] = {
            "train_mean": statistics.mean(tr), "train_std": statistics.pstdev(tr) if len(tr) > 1 else 0.0,
            "holdout_mean": statistics.mean(ho), "holdout_std": statistics.pstdev(ho) if len(ho) > 1 else 0.0,
            "params_per_g": rs[0]["params_per_g"], "secs": sum(r["secs"] for r in rs),
        }
    return out


def decide(agg, metric="holdout", overfit_db=8.0):
    """Gate on GENERALIZATION (held-out PSNR by default), since train PSNR is gameable by
    view-dependent opacity (pilot finding). Flags arms whose train-holdout gap > overfit_db."""
    mkey, skey = f"{metric}_mean", f"{metric}_std"
    D = agg["D"][mkey]
    cand = {a: agg[a][mkey] for a in CANDIDATE_ARMS if a in agg}
    best_arm = max(cand, key=cand.get)
    gap = D - cand[best_arm]
    sigma = max((agg[a][skey] for a in agg), default=0.0)
    if gap <= 0.5:
        v = "GO"
    elif gap <= 1.5:
        v = "EXTEND"
    else:
        v = "NO-GO"
    if sigma > 0 and abs(gap) < sigma:
        v += " (INCONCLUSIVE: |gap| < seed std)"
    overfit = {a: agg[a]["train_mean"] - agg[a]["holdout_mean"] for a in agg}
    flagged = [a for a, d in overfit.items() if d > overfit_db]
    return {"metric": metric, "ceiling_D": D, "best_arm": best_arm, "best_psnr": cand[best_arm],
            "gap": gap, "seed_std": sigma, "verdict": v, "overfit_db": overfit, "overfit_flagged": flagged}


def report(agg, dec):
    m = dec["metric"]
    print(f"\n=== results (gate metric = {m}; train shown for diagnosis) ===")
    print(f"{'arm':<5}{'train':>13}{'holdout':>13}{'tr-ho':>8}{'p/g':>5}{'secs':>7}")
    for arm in ARMS:
        if arm not in agg:
            continue
        a = agg[arm]
        of = a["train_mean"] - a["holdout_mean"]
        flag = "  <- OVERFIT" if arm in dec["overfit_flagged"] else ""
        print(f"{arm:<5}{a['train_mean']:>7.2f}±{a['train_std']:<4.2f}"
              f"{a['holdout_mean']:>7.2f}±{a['holdout_std']:<4.2f}{of:>8.1f}{a['params_per_g']:>5}{a['secs']:>7.0f}{flag}")
    if "C" in agg and "C0" in agg and "A" in agg:
        mk = f"{m}_mean"
        print(f"\nattribution ({m}):  C - C0 (view-dependence) = {agg['C'][mk]-agg['C0'][mk]:+.2f} dB"
              f"   C0 - A (capacity) = {agg['C0'][mk]-agg['A'][mk]:+.2f} dB")
    print(f"\ngate={m} | ceiling D = {dec['ceiling_D']:.2f} | best candidate = {dec['best_arm']} "
          f"{dec['best_psnr']:.2f} | gap = {dec['gap']:.2f} dB | seed std = {dec['seed_std']:.2f}")
    if dec["overfit_flagged"]:
        print(f"overfit-flagged (train-holdout > 8 dB): {dec['overfit_flagged']}  "
              f"-> their train PSNR is not trustworthy; gate uses held-out")
    print(f"VERDICT: {dec['verdict']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-holdout", type=int, default=2)
    ap.add_argument("--extent", type=float, default=1.5)
    ap.add_argument("--arms", default="A,B,C0,C,D")
    ap.add_argument("--sh-degree", type=int, default=0, help="view-dependent SH colour degree (0=DC)")
    ap.add_argument("--preflight", default="", help="comma G list for the arm-D floor check, e.g. 500,2000,8000")
    ap.add_argument("--device", default=None, help="cuda|cpu|cuda:0; default auto (CUDA if available)")
    ap.add_argument("--out", default="outputs/m05")
    args = ap.parse_args()

    dev = default_device(args.device)
    if os.environ.get("EXPECT_CUDA") == "1" and dev.type != "cuda":
        raise SystemExit("EXPECT_CUDA=1 but CUDA is unavailable — GPU not passed through "
                         "(check nvidia-container-toolkit + compose GPU config); refusing to run on CPU.")
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn_only: some CUDA ops lack det impls
    arms = [a for a in args.arms.split(",") if a in ARMS]
    tr_cams, tr_imgs = data.load_blender(args.scene, "train", res=args.res, device=dev,
                                         n=args.n_train, stride=_spread(100, args.n_train))
    ho_cams, ho_imgs = data.load_blender(args.scene, "val", res=args.res, device=dev,
                                         n=args.n_holdout, stride=_spread(100, args.n_holdout))
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else f"cpu ({torch.get_num_threads()} threads)"
    print(f"loaded {len(tr_cams)} train + {len(ho_cams)} holdout @ {args.res}px | device={dev} [{dev_name}] | "
          f"arms={arms} G={args.G} iters={args.iters} seeds={args.seeds}")

    pre = []
    if args.preflight:
        Gs = [int(x) for x in args.preflight.split(",")]
        pre = preflight(tr_cams, tr_imgs, Gs, args.iters, seed=0, extent=args.extent, device=dev)

    rows = []
    for seed in range(args.seeds):
        for arm in arms:
            r = run_arm(arm, tr_cams, tr_imgs, ho_cams, ho_imgs, args.G, args.iters, seed, args.extent, device=dev, sh_degree=args.sh_degree)
            print(f"  [{arm:>2} seed{seed}] train {r['train_psnr']:.2f}  holdout {r['holdout_psnr']:.2f}  "
                  f"loss {r['final_loss']:.4f}  ({r['secs']:.0f}s)")
            rows.append(r)

    agg = aggregate(rows)
    dec = decide(agg) if "D" in agg and any(a in agg for a in ("A", "B", "C")) else None
    if dec:
        report(agg, dec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {"config": vars(args), "preflight": pre, "rows": rows, "agg": agg, "decision": dec}
    with open(args.out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {args.out}.json")


if __name__ == "__main__":
    main()
