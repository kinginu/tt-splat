"""gsplat (route-A) + 3DGS-MCMC fixed-count density control — HIGH-QUALITY ficus/lego plys.

Uses gsplat's canonical MCMCStrategy (cap_max=G, relocate dead->live + noise + prune, fixed count) with
SH degree-3 colour, the standard path to crisp NeRF-synthetic (~34-35 dB). This is the route-A quality
ceiling at a FIXED gaussian budget (matched to route-B's BH-feasible MCMC). Saves a standard 3DGS .ply
(with SH) + held-out PSNR/SSIM + representative GT|render panels.

    docker compose run --rm baseline python tools/gsplat_mcmc_train.py \
        --scenes ficus,lego --res 800 --cap 250000 --iters 30000 --sh-degree 3
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gsplat
from gsplat.strategy import MCMCStrategy
from spike import data, metrics  # noqa: E402


def cams_to_gsplat(cams, dev):
    vm, Ks = [], []
    for c in cams:
        v = torch.eye(4, device=dev); v[:3, :3] = c.R_v; v[:3, 3] = c.t_v
        K = torch.tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1.0]], device=dev)
        vm.append(v); Ks.append(K)
    return torch.stack(vm), torch.stack(Ks)


def init_params(n, extent, sh_deg, dev):
    means = (torch.rand(n, 3, device=dev) * 2 - 1) * extent
    scales = torch.log(torch.full((n, 3), extent * 0.01, device=dev))
    quats = torch.zeros(n, 4, device=dev); quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), 0.1, device=dev))
    n_sh = (sh_deg + 1) ** 2
    sh0 = torch.zeros(n, 1, 3, device=dev)
    shN = torch.zeros(n, n_sh - 1, 3, device=dev)
    p = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means), "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats), "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN)})
    return p


def make_opts(p, scene_scale):
    lr = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
          "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    return {k: torch.optim.Adam([p[k]], lr=lr[k], eps=1e-15) for k in p}


def render(p, vm, K, W, H, sh_deg, bg=1.0):
    colors = torch.cat([p["sh0"], p["shN"]], dim=1)  # [N, n_sh, 3]
    rc, ra, info = gsplat.rasterization(
        means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
        opacities=torch.sigmoid(p["opacities"]), colors=colors,
        viewmats=vm[None], Ks=K[None], width=W, height=H, sh_degree=sh_deg, render_mode="RGB")
    img = rc[0] + (1.0 - ra[0]) * bg
    return img, info


@torch.no_grad()
def evaluate(p, cams, imgs, sh_deg, dev):
    ps, ss, hf = [], [], []
    lap = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=dev).view(1, 1, 3, 3)
    def hfe(x): return F.conv2d(x.permute(2, 0, 1)[:, None], lap, padding=1).pow(2).mean()
    for c, g in zip(cams, imgs):
        vm, K = cams_to_gsplat([c], dev)
        r, _ = render(p, vm[0], K[0], c.W, c.H, sh_deg)
        ps.append(float(metrics.psnr(r, g))); ss.append(float(metrics.ssim(r, g)))
        hf.append(float(hfe(r) / hfe(g).clamp(min=1e-9)))
    return sum(ps) / len(ps), sum(ss) / len(ss), sum(hf) / len(hf)


def save_ply(path, p):
    import struct
    xyz = p["means"].detach().cpu().numpy()
    n = xyz.shape[0]
    f_dc = p["sh0"].detach().cpu().numpy().reshape(n, 3)
    f_rest = p["shN"].detach().transpose(1, 2).reshape(n, -1).cpu().numpy()   # channel-major (inria)
    op = p["opacities"].detach().cpu().numpy().reshape(n, 1)
    sc = p["scales"].detach().cpu().numpy()
    q = F.normalize(p["quats"].detach(), dim=-1).cpu().numpy()
    fields = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
              + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
              + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    arr = np.concatenate([xyz, np.zeros((n, 3), np.float32), f_dc, f_rest, op, sc, q], 1).astype("<f4")
    hdr = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
           + "".join(f"property float {f}\n" for f in fields) + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(hdr.encode()); fh.write(arr.tobytes())
    return n


def to_np(img): return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="ficus,lego")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--cap", type=int, default=250000)
    ap.add_argument("--init-num", type=int, default=100000)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--extent", type=float, default=1.3)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=25)
    ap.add_argument("--out", default="outputs/gsplat_mcmc")
    ap.add_argument("--views-dir", default="docs/gsplat-real-views")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.views_dir, exist_ok=True)

    for scene in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        root = f"data/nerf_synthetic/{scene}"
        tr_c, tr_i = data.load_blender(root, "train", res=args.res, n=args.n_train, device=dev)
        te_c, te_i = data.load_blender(root, "test", res=args.res, n=args.n_test, device=dev)
        p = init_params(args.init_num, args.extent, args.sh_degree, dev)
        opts = make_opts(p, scene_scale=args.extent)
        strat = MCMCStrategy(cap_max=args.cap, refine_stop_iter=int(args.iters * 0.83), verbose=False)
        state = strat.initialize_state()
        ntr = len(tr_c); t0 = time.time()
        for step in range(args.iters):
            i = torch.randint(ntr, (1,)).item()
            sh_deg = min(args.sh_degree, step // 1000)
            vm, K = cams_to_gsplat([tr_c[i]], dev)
            img, info = render(p, vm[0], K[0], tr_c[i].W, tr_c[i].H, sh_deg)
            loss = metrics.loss_fn(img, tr_i[i], 0.2)
            loss.backward()
            for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
            strat.step_post_backward(p, opts, state, step, info, lr=1.6e-4 * args.extent)
            if step % max(1, args.iters // 10) == 0:
                print(f"[{scene}] {step}/{args.iters} loss {float(loss):.4f} G {len(p['means'])} "
                      f"{(step+1)/(time.time()-t0):.1f} it/s", flush=True)
        train_s = time.time() - t0
        ho_p, ho_s, ho_hf = evaluate(p, te_c, te_i, args.sh_degree, dev)
        G = len(p["means"])
        ply = os.path.join(args.out, f"{scene}_mcmc_G{G}_res{args.res}_sh{args.sh_degree}.ply")
        save_ply(ply, p)
        print(f"[{scene}] DONE held-out {ho_p:.2f} dB / {ho_s:.3f} / hf {ho_hf:.3f} | G={G} | "
              f"{train_s/60:.1f} min -> {ply}", flush=True)
        # representative panel: GT | route-A render, held-out view
        with torch.no_grad():
            vm, K = cams_to_gsplat([te_c[1]], dev)
            r, _ = render(p, vm[0], K[0], te_c[1].W, te_c[1].H, args.sh_degree)
        T = 360
        g_t = np.asarray(Image.fromarray(to_np(te_i[1])).resize((T, T)))
        r_t = np.asarray(Image.fromarray(to_np(r)).resize((T, T)))
        gap = np.full((T, 4, 3), 255, np.uint8)
        row = np.concatenate([g_t, gap, r_t], 1)
        full = np.full((T + 16, row.shape[1], 3), 255, np.uint8); full[16:] = row
        im = Image.fromarray(full); d = ImageDraw.Draw(im)
        d.text((2, 3), "GT", fill=(0, 0, 0))
        d.text((T + 6, 3), f"gsplat+MCMC G{G//1000}k SH{args.sh_degree} ({ho_p:.1f}dB)", fill=(0, 0, 0))
        im.save(os.path.join(args.views_dir, f"{scene}_mcmc_hq.png"))
        print(f"[{scene}] panel -> {args.views_dir}/{scene}_mcmc_hq.png", flush=True)


if __name__ == "__main__":
    main()
