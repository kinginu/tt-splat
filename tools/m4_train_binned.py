"""bf16 + BINNED end-to-end matrix-native training ON the Blackhole -- the fast, high-quality device trainer.

Fixes the two limits of tools/m4_train_device.py (fp32, all-G): per the locked scheme each 16x16
tile renders only its <=K local gaussians (R-tile stencil), so mu_local stays small -> bf16 is
accurate (the all-G bf16 failure, rel 0.12, was large mu_local for far gaussians) AND each tile does
~K not G work -> much faster. a binning-quality sweep proved (R=1,K=128) is lossless vs dense.

The per-tile gather is plain torch advanced-indexing (conic[idx], color_o[idx], ...), so autograd
scatters grads back to the per-gaussian params for free (a gaussian in several tiles -> index_add).
A torch.autograd.Function runs the binned fwd/bwd (bf16) on the device.

Run inside the hw container:
  verify:  podman-compose --profile hw run --rm hw python3 tools/m4_train_binned.py verify --res 96 --G 2000
  train:   podman-compose --profile hw run --rm hw python3 tools/m4_train_binned.py train  --res 96 --G 2000 --iters 700
Output: outputs/artifacts/ficus_bh16/{model.ply,train/,test/,metrics.txt}
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

from spike import data, forward, geometry, metrics, plyio
from spike.model import GaussianModel
from spike.render import render as cpu_render

CG = None
RS = None
CKC = None    # HiFi4 + fp32 dest-accumulate matmul config (bf16 LoFi default loses precision on theta_u)
_DEV = None
K_POLY = 4.0


def up(t):
    return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=_DEV)


def mm(a, b):
    if CKC is not None:
        return ttnn.matmul(a, b, core_grid=CG, compute_kernel_config=CKC)
    return ttnn.matmul(a, b, core_grid=CG)


def dn(t):
    return ttnn.to_torch(t).float()


def T3(t):
    return ttnn.transpose(t, -2, -1)


class TileMap:
    def __init__(self, H, W):
        assert H % 16 == 0 and W % 16 == 0
        self.H, self.W, self.nty, self.ntx = H, W, H // 16, W // 16
        self.T = self.nty * self.ntx
        p = torch.arange(256)
        self.Phi = forward.phi((p % 16).float() + 0.5, (p // 16).float() + 0.5)   # [256,6]
        t = torch.arange(self.T)
        ty, tx = t // self.ntx, t % self.ntx
        self.origins = torch.stack([(tx * 16).float(), (ty * 16).float()], dim=1)  # [T,2]
        gr = ty[:, None] * 16 + (p // 16)[None, :]
        gc = tx[:, None] * 16 + (p % 16)[None, :]
        self.gidx = (gr * W + gc).reshape(-1).long()


@torch.no_grad()
def assign_bins(mu2d, keep, tmap, R, K):
    """Vectorized per-tile gaussian index lists. Returns idx[T,K] long, valid[T,K] bool.

    Fully vectorized: O(G*(2R+1)^2) pair construction + one combined argsort (by tile then
    dist-to-centre) + position-within-tile scatter. No Python loop over T tiles.
    Overflow (>K per tile) is handled by the distance-first sort: the first K elements in each
    tile group are the K nearest to tile centre — same policy as the original topk fallback.
    """
    ntx, nty = tmap.ntx, tmap.nty
    gx = (mu2d[:, 0] / 16).floor().long().clamp(0, ntx - 1)
    gy = (mu2d[:, 1] / 16).floor().long().clamp(0, nty - 1)
    G_total = mu2d.shape[0]
    g_all = torch.arange(G_total)

    # Build (tile_id, g_id) pairs — O(G*(2R+1)^2)
    tiles_l, gs_l = [], []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            tx_n = gx + dx
            ty_n = gy + dy
            ok = keep & (tx_n >= 0) & (tx_n < ntx) & (ty_n >= 0) & (ty_n < nty)
            tiles_l.append((ty_n * ntx + tx_n)[ok])
            gs_l.append(g_all[ok])

    idx = torch.zeros(tmap.T, K, dtype=torch.long)
    valid = torch.zeros(tmap.T, K, dtype=torch.bool)
    if not tiles_l or not any(t.numel() > 0 for t in tiles_l):
        return idx, valid

    all_tiles = torch.cat(tiles_l)
    all_gs = torch.cat(gs_l)

    # Sort by (tile_id, dist_to_tile_centre) so first K per tile group = K nearest.
    # Pack into int64: tile * 2048 + quantized_dist2.  At R=1 max dist2 ≈ 1200 < 2048.
    tx_f = (all_tiles % ntx).float()
    ty_f = (all_tiles // ntx).float()
    dist2 = ((mu2d[all_gs, 0] - (tx_f + 0.5) * 16.0) ** 2 +
             (mu2d[all_gs, 1] - (ty_f + 0.5) * 16.0) ** 2)
    sort_key = all_tiles.long() * 2048 + dist2.long().clamp(0, 2047)
    order = torch.argsort(sort_key)
    all_tiles = all_tiles[order]
    all_gs = all_gs[order]

    # Position within tile group: first K are kept (= K nearest by construction)
    t_bnd = torch.searchsorted(all_tiles.contiguous(), torch.arange(tmap.T + 1))
    run_idx = torch.arange(len(all_tiles), dtype=torch.long)
    pos_within = run_idx - t_bnd[all_tiles]
    mask = pos_within < K
    flat_out = all_tiles[mask] * K + pos_within[mask]
    idx.view(-1)[flat_out] = all_gs[mask]
    valid.view(-1)[flat_out] = True
    return idx, valid


def theta_u_gathered(conic_t, mu_t, origins, k):
    """conic_t[T,K,3], mu_t[T,K,2], origins[T,2] -> theta_u [T,6,K] (per-tile origin fold). autograd."""
    mu = mu_t - origins[:, None, :]
    a, b, c = conic_t[..., 0], conic_t[..., 1], conic_t[..., 2]
    mux, muy = mu[..., 0], mu[..., 1]
    ik = 1.0 / k
    Q = a * mux * mux + 2.0 * b * mux * muy + c * muy * muy
    # Inlined: (-1/k)*tQ + bump  where bump[5]=1, all others 0
    return torch.stack([
        -a * ik, -c * ik, -2.0 * b * ik,
        2.0 * (a * mux + b * muy) * ik,
        2.0 * (b * mux + c * muy) * ik,
        1.0 - Q * ik,
    ], dim=-1).transpose(1, 2).contiguous()   # [T,6,K]


def _fwd_ops(thU, col, oc, wbf, bias, Phi_t):
    relu_Q = ttnn.relu(mm(Phi_t, thU))
    w = ttnn.square(relu_Q)
    den = ttnn.add(mm(w, oc), wbf)
    num = ttnn.add(mm(w, col), bias)
    return relu_Q, w, num, den


class _DevRenderBinned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta_u, color_o, o_col, w_b, Phi, gidx, c_b, H, W):
        T = theta_u.shape[0]
        Phi_t = up(Phi.unsqueeze(0).expand(T, 256, 6))
        bias = up((w_b * c_b)[None, None, :].expand(T, 256, 3))
        _, _, num, den = _fwd_ops(up(theta_u), up(color_o), up(o_col), float(w_b), bias, Phi_t)
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
        bias = up((w_b * c_b)[None, None, :].expand(T, 256, 3))
        col, oc = up(color_o), up(o_col)
        relu_Q, w, num, den = _fwd_ops(up(theta_u), col, oc, float(w_b), bias, Phi_t)
        C = ttnn.div(num, den)
        gtiled = up(gimg.reshape(H * W, 3)[gidx].reshape(T, 256, 3))
        gnum = ttnn.div(gtiled, den)
        gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(gtiled, C), dim=-1, keepdim=True)), den)
        gcol = dn(mm(T3(w), gnum))                                              # [T,K,3] per-tile
        goc = dn(mm(T3(w), gden))                                               # [T,K,1] per-tile
        gw = ttnn.add(mm(gnum, T3(col)), mm(gden, T3(oc)))
        gthU = dn(mm(T3(Phi_t), ttnn.mul(gw, ttnn.mul(relu_Q, 2.0))))
        gnum_h, gden_h = dn(gnum), dn(gden)
        gw_b = ((gnum_h * c_b[None, None, :]).sum() + gden_h.sum()).float()
        return gthU, gcol, goc, gw_b, None, None, None, None, None


def _operands(model, cam, tmap, R, K, cached_bins=None):
    """Host: geometry + binning + per-tile gather (autograd-tracked except the bin indices).
    cached_bins: optional (idx, valid) pair to skip recomputing assign_bins this iter.
    Returns operands + the (idx, valid) pair that was used (fresh or cached).
    """
    Rm = geometry.quat_to_rotmat(model.quats)
    cov = geometry.cov3d(torch.exp(model.log_scales), Rm)
    mu2d, conic, depth, keep = geometry.project_ewa(
        model.means3d, cov, cam.R_v, cam.t_v, cam.fx, cam.fy, cam.cx, cam.cy, 0.3, 0.2)
    color = forward.color_from_dc(model.color_dc)
    o = torch.sigmoid(model.opacity_raw)
    w_b = F.softplus(model.w_b_raw)
    keo = (keep.float() * o)[:, None]
    color_o, o_col = keo * color, keo
    if cached_bins is None:
        idx, valid = assign_bins(mu2d.detach(), keep.detach(), tmap, R, K)
    else:
        idx, valid = cached_bins
    vf = valid[..., None].float()
    conic_t, mu_t = conic[idx], mu2d[idx]                                       # [T,K,3],[T,K,2]
    theta_u = theta_u_gathered(conic_t, mu_t, tmap.origins, K_POLY) * valid[:, None, :].float()
    color_o_t = color_o[idx] * vf
    o_col_t = o_col[idx] * vf
    return theta_u, color_o_t, o_col_t, w_b, (idx, valid)


def render_binned_device(model, cam, tmap, R=1, K=128, cached_bins=None, return_bins=False):
    thU, col, oc, w_b, bins = _operands(model, cam, tmap, R, K, cached_bins)
    img = _DevRenderBinned.apply(thU, col, oc, w_b, tmap.Phi, tmap.gidx, model.c_b, cam.H, cam.W)
    img = img.reshape(cam.H, cam.W, 3)
    return (img, bins) if return_bins else img


def render_binned_cpu(model, cam, tmap, R=1, K=128):
    """Same binned math in torch fp32 -- the oracle that isolates the device/bf16 error from binning."""
    thU, col, oc, w_b = _operands(model, cam, tmap, R, K)
    T = thU.shape[0]
    w = torch.relu(torch.einsum("pf,tfg->tpg", tmap.Phi, thU)) ** 2             # [T,256,K]
    C = (w @ col + w_b * model.c_b) / (w @ oc + w_b)                            # [T,256,3]
    img = torch.zeros(cam.H * cam.W, 3)
    img[tmap.gidx] = C.reshape(-1, 3)
    return img.reshape(cam.H, cam.W, 3)


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def verify(args):
    cams, imgs = data.load_blender(args.scene, "train", res=args.res, n=1)
    cam, gt = cams[0], imgs[0]
    tmap = TileMap(cam.H, cam.W)
    m = GaussianModel(args.G, extent=1.5, seed=0)
    with torch.no_grad():
        dense = cpu_render(m, cam, "A")
        bcpu = render_binned_cpu(m, cam, tmap)
        bdev = render_binned_device(m, cam, tmap)
    print(f"  binned-CPU vs dense (binning approx, K=128): rel {_rel(bcpu, dense):.4f}")
    print(f"  binned-device(bf16) vs binned-CPU:           rel {_rel(bdev, bcpu):.4f}")
    print(f"  binned-device(bf16) vs dense (total):        rel {_rel(bdev, dense):.4f}")

    def grads(fn):
        mm = GaussianModel(args.G, extent=1.5, seed=0)
        mm.load_state_dict(m.state_dict())
        for p in mm.parameters():
            p.requires_grad_(True)
        ((fn(mm) - gt) ** 2).mean().backward()
        return {n: p.grad for n, p in mm.named_parameters()}

    gc = grads(lambda mm: render_binned_cpu(mm, cam, tmap))
    gd = grads(lambda mm: render_binned_device(mm, cam, tmap))
    print("  gradient rel-err device(bf16) vs binned-CPU autograd:")
    for n in ("means3d", "log_scales", "quats", "opacity_raw", "color_dc", "w_b_raw"):
        print(f"    {n:12s}: {_rel(gd[n], gc[n]):.4f}")


def save_img(path, img):
    Image.fromarray((img.clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def sbs(gt, pred, gap=4):
    return torch.cat([gt.clamp(0, 1), torch.ones(gt.shape[0], gap, 3), pred.clamp(0, 1)], dim=1)


def train(args):
    out = "outputs/artifacts/ficus_bh16"
    for s in ("train", "test"):
        os.makedirs(os.path.join(out, s), exist_ok=True)
    tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, stride=max(1, 100 // args.n_train))
    te_c, te_i = data.load_blender(args.scene, "test", res=args.res, n=args.n_test, stride=max(1, 200 // args.n_test))
    tmap = TileMap(args.res, args.res)
    torch.manual_seed(args.seed)
    m = GaussianModel(args.G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    opt = torch.optim.Adam(m.param_groups(DEFAULT_LR))
    print(f"[bh16] BLACKHOLE bf16+binned | res={args.res} G={args.G} T={tmap.T} K={args.K} | {len(tr_c)} views | {args.iters} it")
    t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for cam, gt in zip(tr_c, tr_i):
            tot = tot + metrics.loss_fn(render_binned_device(m, cam, tmap, K=args.K), gt, lambda_ssim=0.2)
        (tot / len(tr_c)).backward()
        opt.step()
        if it % max(1, args.iters // 14) == 0 or it == args.iters - 1:
            print(f"    iter {it:4d}  loss {float(tot)/len(tr_c):.4f}  ({(time.perf_counter()-t0)/(it+1)*1e3:.0f} ms/it)")
    tt = time.perf_counter() - t0
    n = plyio.save_ply(os.path.join(out, "model.ply"), m)
    print(f"[bh16] trained on device in {tt:.0f}s ({tt/args.iters*1e3:.0f} ms/it); {n} gaussians")
    lines = [f"scene=ficus TRAINED-ON-BLACKHOLE-bf16-binned res={args.res} G={args.G} K={args.K} iters={args.iters}",
             f"train_time_s={tt:.0f} ms_per_it={tt/args.iters*1e3:.0f} gaussians={n}", ""]
    with torch.no_grad():
        for split, cs, ims in (("train", tr_c, tr_i), ("test", te_c, te_i)):
            ps = []
            for i, (cam, gt) in enumerate(zip(cs, ims)):
                r = render_binned_device(m, cam, tmap, K=args.K)
                ps.append(float(metrics.psnr(r, gt)))
                save_img(os.path.join(out, split, f"view{i:02d}_gt.png"), gt)
                save_img(os.path.join(out, split, f"view{i:02d}_render.png"), r)
                save_img(os.path.join(out, split, f"view{i:02d}_sbs.png"), sbs(gt, r))
            lines.append(f"{split} mean PSNR {sum(ps)/len(ps):.2f} dB  " + ", ".join(f"{p:.2f}" for p in ps))
            print(f"[bh16] {split} mean PSNR {sum(ps)/len(ps):.2f} dB")
    open(os.path.join(out, "metrics.txt"), "w").write("\n".join(lines) + "\n")
    print(f"[bh16] -> {out}/")


def main():
    global CG, RS, CKC, _DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "train"])
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--K", type=int, default=128, help="per-tile gaussian budget (overflow dropped by nearest)")
    ap.add_argument("--iters", type=int, default=700)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lofi", action="store_true", help="disable HiFi4 config (debug)")
    args = ap.parse_args()
    _DEV = ttnn.open_device(device_id=0)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        RS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
        if not args.lofi:
            try:
                CKC = ttnn.init_device_compute_kernel_config(
                    _DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True, math_approx_mode=False)
                print("compute_kernel_config: HiFi4 + fp32_dest_acc")
            except Exception as e:  # noqa: BLE001
                print("CKC setup failed, falling back to default LoFi:", type(e).__name__, str(e)[:80])
        (verify if args.mode == "verify" else train)(args)
    finally:
        ttnn.close_device(_DEV)


if __name__ == "__main__":
    main()
