"""New execution model for the train step. A profiling breakdown showed
render fwd+bwd ~= 22 ms/view (~50%, recompute + autograd) dominates; geometry/binning/gather are cheap
host work. So: keep geometry/binning/gather/theta + their backward on HOST torch autograd (cheap),
and replace ONLY the render with FastRender -- a manual, cached (no fwd-recompute), ttnn-TRACED fwd+bwd,
BATCHED over all views (B = n_views * T tiles) so the device dispatch + per-view host overhead amortize.

Because gather/theta/geometry stay host-autograd, FastRender only needs to return grads w.r.t the render
inputs (theta_u, color_o_t, o_col_t, bias, wb); torch autograd then backprops theta-build + gather +
geometry to the params for free. No new device backward, no manual scatter.

Oracle: same held-out PSNR as tools/m4_train_binned.py. Metric: ms/it traced-batched vs untraced.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m6_faststep.py --compare --res 128 --G 8000 --K 256
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike import data, metrics
from spike.model import GaussianModel
import m4_train_binned as mtb
from m4_train_binned import TileMap, _operands, up, dn, T3, render_binned_device
from m6_operands_batched import operands_batched

DEV = None
CG = None


class FastRender:
    """Manual, cached, traced render fwd+bwd over B tiles. Inputs swapped in place per iter."""
    def __init__(self, B, K, Phi):
        z = lambda *s: ttnn.from_torch(torch.zeros(*s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)
        self.thU, self.col, self.oc = z(B, 6, K), z(B, K, 3), z(B, K, 1)
        self.bias, self.wb, self.gC = z(B, 256, 3), z(B, 256, 1), z(B, 256, 3)
        self.Phi = ttnn.from_torch(Phi.unsqueeze(0).expand(B, 256, 6).contiguous(),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)

        def fwd():
            reluQ = ttnn.relu(ttnn.matmul(self.Phi, self.thU, core_grid=CG))
            w = ttnn.square(reluQ)
            den = ttnn.add(ttnn.matmul(w, self.oc, core_grid=CG), self.wb)
            num = ttnn.add(ttnn.matmul(w, self.col, core_grid=CG), self.bias)
            C = ttnn.div(num, den)
            return reluQ, w, den, C

        # warmup + capture fwd (cache tensors become persistent handles)
        self.reluQ, self.w, self.den, self.C = fwd(); ttnn.synchronize_device(DEV)
        self.fid = ttnn.begin_trace_capture(DEV, cq_id=0)
        self.reluQ, self.w, self.den, self.C = fwd()
        ttnn.end_trace_capture(DEV, self.fid, cq_id=0); ttnn.synchronize_device(DEV)

        def bwd():
            gnum = ttnn.div(self.gC, self.den)
            gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(self.gC, self.C), dim=-1, keepdim=True)), self.den)
            gcol = ttnn.matmul(T3(self.w), gnum, core_grid=CG)
            goc = ttnn.matmul(T3(self.w), gden, core_grid=CG)
            gw = ttnn.add(ttnn.matmul(gnum, T3(self.col), core_grid=CG),
                          ttnn.matmul(gden, T3(self.oc), core_grid=CG))
            gthU = ttnn.matmul(T3(self.Phi), ttnn.mul(gw, ttnn.mul(self.reluQ, 2.0)), core_grid=CG)
            return gthU, gcol, goc, gnum, gden

        bwd(); ttnn.synchronize_device(DEV)
        self.bid = ttnn.begin_trace_capture(DEV, cq_id=0)
        self.bouts = bwd()
        ttnn.end_trace_capture(DEV, self.bid, cq_id=0); ttnn.synchronize_device(DEV)

    def _set(self, buf, t):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), buf)

    def forward(self, thU, col, oc, bias, wb):
        self._set(self.thU, thU); self._set(self.col, col); self._set(self.oc, oc)
        self._set(self.bias, bias); self._set(self.wb, wb)
        ttnn.execute_trace(DEV, self.fid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        return dn(self.C)

    def backward(self, gC):
        self._set(self.gC, gC)
        ttnn.execute_trace(DEV, self.bid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
        return [dn(x) for x in self.bouts]   # gthU, gcol, goc, gnum, gden


_FR = {}


def train_step_fast(model, cams, gts, tmap, K, opt):
    """One batched iter: host geometry/gather (autograd) -> FastRender (traced) -> host autograd backprop."""
    opt.zero_grad(set_to_none=True)
    T = tmap.T
    thU_all, col_all, oc_all, w_b = operands_batched(model, cams, tmap, 1, K)   # host, autograd, batched
    B = thU_all.shape[0]
    bias_all = (w_b * model.c_b)[None, None, :].expand(B, 256, 3)               # w_b view-independent
    wb_all = w_b.reshape(1, 1, 1).expand(B, 256, 1)

    key = (len(cams) * T, K)
    if key not in _FR:
        _FR[key] = FastRender(key[0], K, tmap.Phi)
    fr = _FR[key]
    C_all = fr.forward(thU_all.detach(), col_all.detach(), oc_all.detach(),
                       bias_all.detach(), wb_all.detach())             # [B,256,3] host

    # loss + gC via a host leaf (gives dL/dC for the manual render backward)
    C_leaf = C_all.clone().requires_grad_(True)
    loss = 0.0
    for v, gt in enumerate(gts):
        H, W = gt.shape[0], gt.shape[1]
        img = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, C_leaf[v * T:(v + 1) * T].reshape(T * 256, 3))
        loss = loss + metrics.loss_fn(img.reshape(H, W, 3), gt, lambda_ssim=0.2)
    loss = loss / len(gts)
    loss.backward()
    gC_all = C_leaf.grad                                              # [B,256,3]

    gthU, gcol, goc, gnum, gden = fr.backward(gC_all)
    torch.autograd.backward([thU_all, col_all, oc_all, bias_all, wb_all],
                            [gthU, gcol, goc, gnum, gden])            # host autograd -> params
    opt.step()
    return float(loss)


def fit_eval(use_fast, label, args):
    tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, stride=max(1, 100 // args.n_train))
    te_c, te_i = data.load_blender(args.scene, "test", res=args.res, n=args.n_test, stride=max(1, 200 // args.n_test))
    tmap = TileMap(args.res, args.res)
    torch.manual_seed(args.seed)
    m = GaussianModel(args.G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    opt = torch.optim.Adam(m.param_groups(DEFAULT_LR))
    if use_fast:
        train_step_fast(m, tr_c, tr_i, tmap, args.K, opt)            # warm (JIT+capture) off-clock
    else:
        render_binned_device(m, tr_c[0], tmap, 1, args.K)
    t0 = time.perf_counter()
    for it in range(args.iters):
        if use_fast:
            train_step_fast(m, tr_c, tr_i, tmap, args.K, opt)
        else:
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for cam, gt in zip(tr_c, tr_i):
                tot = tot + metrics.loss_fn(render_binned_device(m, cam, tmap, 1, args.K), gt, lambda_ssim=0.2)
            (tot / len(tr_c)).backward(); opt.step()
    tt = time.perf_counter() - t0
    with torch.no_grad():
        te = sum(float(metrics.psnr(render_binned_device(m, c, tmap, 1, args.K), g))
                 for c, g in zip(te_c, te_i)) / len(te_c)
    print(f"   [{label:10s}] {tt/args.iters*1e3:7.0f} ms/it | held-out {te:.2f} dB")
    return tt / args.iters * 1e3, te


def main():
    global DEV, CG
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/nerf_synthetic/ficus")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--G", type=int, default=8000)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        mtb._DEV, mtb.CG, mtb.CKC = DEV, CG, None
        print(f"== fast train step (batched+traced render) res={args.res} G={args.G} K={args.K} "
              f"{args.n_train}tr/{args.n_test}te {args.iters}it ==")
        fa = fit_eval(True, "fast(v1+v2)", args)
        if args.compare:
            pa = fit_eval(False, "partial-BH", args)
            print(f"\n   speed: fast {fa[0]:.0f} vs partial-BH {pa[0]:.0f} ms/it "
                  f"({pa[0]/fa[0]:.2f}x faster) | held-out {fa[1]:.2f} vs {pa[1]:.2f} dB")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
