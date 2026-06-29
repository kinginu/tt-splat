"""End-to-end matrix-native training with the P x G hot path ON THE BLACKHOLE.

A torch.autograd.Function (`_DevRender`) isolates the DEVICE forward (tile-local, all-G, bf16:
Phi.theta_u -> relu^2 -> WSR) and the DEVICE backward (the verified transposed GEMMs). Everything
else stays host autograd: geometry (project_ewa), the conic/mean -> theta_u map, Adam, MCMC. So the
gradient that updates every parameter is computed on real silicon for the heavy P x G part -> the .ply
this produces is genuinely Blackhole-trained.

Run inside the hw container:
  verify (vs the CPU oracle, fwd + grads):
    podman-compose --profile hw run --rm hw python3 tools/m4_train_device.py verify --res 96 --G 800
  train + export a Blackhole-trained .ply:
    podman-compose --profile hw run --rm hw python3 tools/m4_train_device.py train \
        --res 96 --G 2000 --iters 700 --n-train 6 --n-test 2

Layout matches tools/export_artifacts.py: outputs/artifacts/ficus_bh/{model.ply,train/,test/,metrics.txt}.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import ttnn

from spike import data, forward, geometry, metrics, plyio, evalcard
from spike.model import GaussianModel
from spike.render import render as cpu_render

CG = None
RS = None
_DT = None   # ttnn dtype for device tensors (set in main; bf16 for real runs, fp32 for the logic check)


def up(t):
    return ttnn.from_torch(t.contiguous().float(), dtype=_DT, layout=ttnn.TILE_LAYOUT, device=_DEV)


def dn(t):
    return ttnn.to_torch(t).float()


def T3(t):
    return ttnn.transpose(t, -2, -1)


# ---------------- tile geometry (host, constant) ----------------
class TileMap:
    def __init__(self, H, W):
        assert H % 16 == 0 and W % 16 == 0, "res must be a multiple of 16"
        self.H, self.W = H, W
        self.nty, self.ntx = H // 16, W // 16
        self.T = self.nty * self.ntx
        p = torch.arange(256)
        lx = (p % 16).float() + 0.5
        ly = (p // 16).float() + 0.5
        self.Phi = forward.phi(lx, ly)                                  # [256,6] tile-local, shared
        t = torch.arange(self.T)
        ty, tx = t // self.ntx, t % self.ntx
        self.origins = torch.stack([(tx * 16).float(), (ty * 16).float()], dim=1)   # [T,2] (x=col,y=row)
        gr = ty[:, None] * 16 + (p // 16)[None, :]
        gc = tx[:, None] * 16 + (p % 16)[None, :]
        self.gidx = (gr * W + gc).reshape(-1).long()                    # [T*256] tiled->global flat


def theta_u_tiles(conic, mu2d, origins, k):
    """Per-tile theta_u (folds 1 - Q/k with the tile origin in the mean). -> [T,6,G], host autograd."""
    mu = mu2d[None, :, :] - origins[:, None, :]                         # [T,G,2]
    a, b, c = conic[:, 0][None, :], conic[:, 1][None, :], conic[:, 2][None, :]
    mux, muy = mu[..., 0], mu[..., 1]
    z = torch.zeros_like(mux)
    tQ = torch.stack([a + z, c + z, 2 * b + z,
                      -2 * (a * mux + b * muy), -2 * (b * mux + c * muy),
                      a * mux * mux + 2 * b * mux * muy + c * muy * muy], dim=-1)    # [T,G,6]
    bump = torch.zeros_like(tQ)
    bump[..., 5] = 1.0
    return ((-1.0 / k) * tQ + bump).transpose(1, 2).contiguous()       # [T,6,G]


# ---------------- the device render autograd.Function ----------------
class _DevRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta_u, color_o, o_col, w_b, Phi, gidx, c_b, H, W):
        T = theta_u.shape[0]
        Phi_t = up(Phi.unsqueeze(0).expand(T, 256, 6))
        thU = up(theta_u)
        col = up(color_o.unsqueeze(0).expand(T, *color_o.shape))
        oc = up(o_col.unsqueeze(0).expand(T, *o_col.shape))
        wbf = float(w_b)
        bias = up((w_b * c_b)[None, None, :].expand(T, 256, 3))
        w = ttnn.unary_chain(ttnn.matmul(Phi_t, thU, core_grid=CG), RS)          # [T,256,G]
        num = ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias)
        den = ttnn.add(ttnn.matmul(w, oc, core_grid=CG), wbf)
        C = dn(ttnn.div(num, den)).reshape(T * 256, 3)
        img = torch.zeros(H * W, 3)
        img[gidx] = C
        ctx.save_for_backward(theta_u, color_o, o_col, w_b, Phi, gidx, c_b)
        ctx.shape = (T, H, W)
        return img.reshape(H, W, 3)

    @staticmethod
    def backward(ctx, gimg):
        theta_u, color_o, o_col, w_b, Phi, gidx, c_b = ctx.saved_tensors
        T, H, W = ctx.shape
        Phi_t = up(Phi.unsqueeze(0).expand(T, 256, 6))
        thU = up(theta_u)
        col = up(color_o.unsqueeze(0).expand(T, *color_o.shape))
        oc = up(o_col.unsqueeze(0).expand(T, *o_col.shape))
        wbf = float(w_b)
        bias = up((w_b * c_b)[None, None, :].expand(T, 256, 3))
        relu_Q = ttnn.relu(ttnn.matmul(Phi_t, thU, core_grid=CG))
        w = ttnn.square(relu_Q)
        den = ttnn.add(ttnn.matmul(w, oc, core_grid=CG), wbf)
        C = ttnn.div(ttnn.add(ttnn.matmul(w, col, core_grid=CG), bias), den)

        gtiled = up(gimg.reshape(H * W, 3)[gidx].reshape(T, 256, 3))
        gnum = ttnn.div(gtiled, den)
        gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(gtiled, C), dim=-1, keepdim=True)), den)
        gcol = ttnn.matmul(T3(w), gnum, core_grid=CG)                            # [T,G,3]
        goc = ttnn.matmul(T3(w), gden, core_grid=CG)                            # [T,G,1]
        gw = ttnn.add(ttnn.matmul(gnum, T3(col), core_grid=CG),
                      ttnn.matmul(gden, T3(oc), core_grid=CG))                   # [T,256,G]
        gQ = ttnn.mul(gw, ttnn.mul(relu_Q, 2.0))
        gthU = dn(ttnn.matmul(T3(Phi_t), gQ, core_grid=CG))                     # [T,6,G]
        gcolor_o = dn(gcol).sum(0)                                              # [G,3] (shared)
        go_col = dn(goc).sum(0)                                                 # [G,1]
        gnum_h, gden_h = dn(gnum), dn(gden)
        gw_b = ((gnum_h * c_b[None, None, :]).sum() + gden_h.sum()).float()
        return gthU, gcolor_o, go_col, gw_b, None, None, None, None, None


def render_device(model, cam, tmap, k=4.0, blur_eps=0.3, near=0.2):
    R = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), R)
    mu2d, conic, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, blur_eps, near)
    color = forward.color_from_dc(model.color_dc)
    o = torch.sigmoid(model.opacity_raw)
    w_b = F.softplus(model.w_b_raw)
    keo = (keep.float() * o)[:, None]
    color_o = keo * color
    o_col = keo
    theta_u = theta_u_tiles(conic, mu2d, tmap.origins, k)
    img = _DevRender.apply(theta_u, color_o, o_col, w_b, tmap.Phi, tmap.gidx, model.c_b, cam.H, cam.W)
    return img.reshape(cam.H, cam.W, 3)


# ---------------- modes ----------------
def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def tiled_forward_cpu(m, cam, tmap, k=4.0):
    """The EXACT same tiled decomposition the device runs, but in torch fp32 -- isolates the tiling
    LOGIC (pixel order / origin fold / WSR) from bf16 precision. Should match cpu_render to ~1e-5."""
    R = geometry.quat_to_rotmat(m.quats)
    cov = geometry.cov3d(torch.exp(m.log_scales), R)
    mu2d, conic, depth, keep = geometry.project_ewa(
        m.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, 0.3, 0.2)
    color = forward.color_from_dc(m.color_dc)
    o = torch.sigmoid(m.opacity_raw)
    w_b = F.softplus(m.w_b_raw)
    keo = (keep.float() * o)[:, None]
    color_o, o_col = keo * color, keo
    theta_u = theta_u_tiles(conic, mu2d, tmap.origins, k)               # [T,6,G]
    Q = torch.einsum("pf,tfg->tpg", tmap.Phi, theta_u)                  # [T,256,G]
    w = torch.relu(Q) ** 2
    C = (w @ color_o + w_b * m.c_b) / (w @ o_col + w_b)                 # [T,256,3]
    img = torch.zeros(cam.H * cam.W, 3)
    img[tmap.gidx] = C.reshape(-1, 3)
    return img.reshape(cam.H, cam.W, 3)


def verify(args, dev):
    global _DT
    cams, imgs = data.load_blender(args.scene, "train", res=args.res, n=1)
    cam, gt = cams[0], imgs[0]
    tmap = TileMap(cam.H, cam.W)
    torch.manual_seed(0)
    m = GaussianModel(args.G, extent=1.5, seed=0)

    with torch.no_grad():
        c_cpu = cpu_render(m, cam, "A")
        c_tiled = tiled_forward_cpu(m, cam, tmap)                       # logic check (fp32)
        print(f"  forward CPU-tiled(fp32) vs CPU oracle: rel {_rel(c_tiled, c_cpu):.4f}  (isolates tiling logic)")
        for dtname, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
            _DT = dt
            try:
                c_dev = render_device(m, cam, tmap)
                print(f"  forward device({dtname}) vs CPU oracle:  rel {_rel(c_dev, c_cpu):.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"  forward device({dtname}): ERROR {type(e).__name__}: {str(e)[:70]}")
    _DT = ttnn.float32   # grads checked in fp32 (bf16 all-G is precision-bound; binning is the bf16 path)

    def grads(render_fn):
        mm = GaussianModel(args.G, extent=1.5, seed=0)
        mm.load_state_dict(m.state_dict())
        for p in mm.parameters():
            p.requires_grad_(True)
        ((render_fn(mm) - gt) ** 2).mean().backward()
        return {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in mm.named_parameters()}

    gc = grads(lambda mm: cpu_render(mm, cam, "A"))
    gd = grads(lambda mm: render_device(mm, cam, tmap))
    print("  gradient rel-err device vs CPU autograd:")
    for n in ("means3d", "log_scales", "quats", "opacity_raw", "color_dc", "w_b_raw"):
        print(f"    {n:12s}: {_rel(gd[n], gc[n]):.4f}")


def save_img(path, img):
    Image.fromarray((img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def sbs(gt, pred, gap=4):
    sep = torch.ones(gt.shape[0], gap, 3)
    return torch.cat([gt.clamp(0, 1), sep, pred.clamp(0, 1)], dim=1)


def train(args, dev):
    scene = os.path.basename(args.scene.rstrip("/"))
    paths = evalcard.run_paths(args.method, scene, args.G, args.res, iters=args.iters,
                               seed=args.seed, root=args.out)
    out = paths["dir"]
    for s in ("train", "test"):
        os.makedirs(os.path.join(out, s), exist_ok=True)
    tr_cams, tr_imgs = data.load_blender(args.scene, "train", res=args.res, n=args.n_train,
                                         stride=max(1, 100 // args.n_train))
    te_cams, te_imgs = data.load_blender(args.scene, "test", res=args.res, n=args.n_test,
                                         stride=max(1, 200 // args.n_test))
    te_cams_a, te_rgba = data.load_blender(args.scene, "test", res=args.res, n=args.n_test,
                                           stride=max(1, 200 // args.n_test), keep_alpha=True)
    tmap = TileMap(args.res, args.res)
    torch.manual_seed(args.seed)
    m = GaussianModel(args.G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    opt = torch.optim.Adam(m.param_groups(DEFAULT_LR))
    print(f"[bh-train] {scene} ON BLACKHOLE | res={args.res} G={args.G} T={tmap.T} | "
          f"{len(tr_cams)} train views | {args.iters} iters")
    t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for cam, gt in zip(tr_cams, tr_imgs):
            img = render_device(m, cam, tmap)
            tot = tot + metrics.loss_fn(img, gt, lambda_ssim=0.2)
        (tot / len(tr_cams)).backward()
        opt.step()
        if it % max(1, args.iters // 10) == 0 or it == args.iters - 1:
            print(f"    iter {it:4d}  loss {float(tot)/len(tr_cams):.4f}  ({(time.perf_counter()-t0)/(it+1)*1e3:.0f} ms/it)")
    tt = time.perf_counter() - t0
    n = plyio.save_ply(paths["ply"], m)
    print(f"[bh-train] trained on device in {tt:.0f}s ({tt/args.iters*1e3:.0f} ms/it); {paths['ply']} {n} gaussians")

    lines = [f"scene={scene} TRAINED-ON-BLACKHOLE res={args.res} G={args.G} iters={args.iters}",
             f"train_time_s={tt:.0f} gaussians={n}", ""]
    with torch.no_grad():
        for split, cams, imgs in (("train", tr_cams, tr_imgs), ("test", te_cams, te_imgs)):
            ps = []
            for i, (cam, gt) in enumerate(zip(cams, imgs)):
                r = render_device(m, cam, tmap)
                ps.append(float(metrics.psnr(r, gt)))
                save_img(os.path.join(out, split, f"view{i:02d}_gt.png"), gt)
                save_img(os.path.join(out, split, f"view{i:02d}_render.png"), r)
                save_img(os.path.join(out, split, f"view{i:02d}_sbs.png"), sbs(gt, r))
            lines.append(f"{split} mean PSNR {sum(ps)/len(ps):.2f} dB  " + ", ".join(f"{p:.2f}" for p in ps))
            print(f"[bh-train] {split} mean PSNR {sum(ps)/len(ps):.2f} dB")
    open(os.path.join(out, "metrics.txt"), "w").write("\n".join(lines) + "\n")
    # unified eval card (host cpu_render oracle with c_b swapped for the two-bg coverage trick); guarded
    try:
        tr_mean = float(lines[3].split("PSNR")[1].split("dB")[0]) if len(lines) > 3 else None

        def rfn(mdl, cam, b):
            saved = mdl.c_b.detach().clone()
            mdl.c_b.fill_(b)
            try:
                return cpu_render(mdl, cam, "A")
            finally:
                mdl.c_b.copy_(saved)
        card = evalcard.build(m, te_cams_a, te_rgba, rfn, method=args.method, scene=scene,
                              G=args.G, res=args.res, iters=args.iters, seed=args.seed,
                              train_psnr=tr_mean,
                              perf={"train_s": round(tt, 1), "ms_per_it": round(tt / max(args.iters, 1) * 1e3, 2),
                                    "device": "blackhole"})
        evalcard.save(card, paths["eval_json"])
        print(f"[bh-train] eval card -> {paths['eval_json']}")
    except Exception as e:
        print(f"[bh-train] WARN eval card skipped: {e}")
    print(f"[bh-train] -> {out}/")


def main():
    global CG, RS, _DEV, _DT
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "train"])
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=700)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="routeB_bh", help="rough algo label for the unified output path")
    ap.add_argument("--out", default="outputs", help="root; layout = <out>/<method>/<scene>/G<G>_res<res>")
    args = ap.parse_args()
    _DEV = ttnn.open_device(device_id=0)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        RS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
        _DT = ttnn.bfloat16
        (verify if args.mode == "verify" else train)(args, _DEV)
    finally:
        ttnn.close_device(_DEV)


if __name__ == "__main__":
    main()
