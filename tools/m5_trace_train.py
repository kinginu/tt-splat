"""Trace integration: speed up the actual training loop by tracing the device render fwd+bwd. Per the
findings, host geometry/gather (vectorized torch) is fast; the device render is dispatched per view per
iter and is dispatch-bound. The render ops are FIXED-shape and camera-independent, so ONE fwd trace +
ONE bwd trace are captured once and replayed for every view/iter -- inputs (theta_u, color_o_t, o_col_t,
w_b, gtiled) are swapped in place via ttnn.copy_host_to_device_tensor. Geometry/gather/binning/Adam stay
host (the faster choice); only _DevRenderBinned is replaced by TracedRender.

Oracle: same held-out PSNR as tools/m4_train_binned.py. Metric: ms/it traced vs untraced.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/m5_trace_train.py --compare --res 128 --G 8000 --K 256
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

DEV = None
CG = None
_TR = {}   # trace state keyed by (T, K)


def _buf(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)


def _host(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _set(buf, t):
    ttnn.copy_host_to_device_tensor(_host(t), buf)


def _init_traces(T, K, Phi, c_b):
    """Allocate persistent buffers + capture the fwd and bwd render traces once (shapes fixed)."""
    b = {}
    b["thU"] = _buf(torch.zeros(T, 6, K))
    b["col"] = _buf(torch.zeros(T, K, 3))
    b["oc"] = _buf(torch.zeros(T, K, 1))
    b["Phi"] = _buf(Phi.unsqueeze(0).expand(T, 256, 6))
    b["bias"] = _buf(torch.zeros(T, 256, 3))
    b["wb"] = _buf(torch.zeros(T, 256, 1))
    b["gt"] = _buf(torch.zeros(T, 256, 3))

    def fwd_ops():
        relu_Q = ttnn.relu(ttnn.matmul(b["Phi"], b["thU"], core_grid=CG))
        w = ttnn.square(relu_Q)
        den = ttnn.add(ttnn.matmul(w, b["oc"], core_grid=CG), b["wb"])
        num = ttnn.add(ttnn.matmul(w, b["col"], core_grid=CG), b["bias"])
        return ttnn.div(num, den)

    def bwd_ops():
        relu_Q = ttnn.relu(ttnn.matmul(b["Phi"], b["thU"], core_grid=CG))
        w = ttnn.square(relu_Q)
        den = ttnn.add(ttnn.matmul(w, b["oc"], core_grid=CG), b["wb"])
        num = ttnn.add(ttnn.matmul(w, b["col"], core_grid=CG), b["bias"])
        C = ttnn.div(num, den)
        gnum = ttnn.div(b["gt"], den)
        gden = ttnn.div(ttnn.neg(ttnn.sum(ttnn.mul(b["gt"], C), dim=-1, keepdim=True)), den)
        gcol = ttnn.matmul(T3(w), gnum, core_grid=CG)
        goc = ttnn.matmul(T3(w), gden, core_grid=CG)
        gw = ttnn.add(ttnn.matmul(gnum, T3(b["col"]), core_grid=CG),
                      ttnn.matmul(gden, T3(b["oc"]), core_grid=CG))
        gthU = ttnn.matmul(T3(b["Phi"]), ttnn.mul(gw, ttnn.mul(relu_Q, 2.0)), core_grid=CG)
        return gthU, gcol, goc, gnum, gden

    fwd_ops(); ttnn.synchronize_device(DEV)                       # warmup (JIT) before capture
    fid = ttnn.begin_trace_capture(DEV, cq_id=0)
    Cout = fwd_ops()
    ttnn.end_trace_capture(DEV, fid, cq_id=0); ttnn.synchronize_device(DEV)
    bwd_ops(); ttnn.synchronize_device(DEV)
    bid = ttnn.begin_trace_capture(DEV, cq_id=0)
    bouts = bwd_ops()
    ttnn.end_trace_capture(DEV, bid, cq_id=0); ttnn.synchronize_device(DEV)
    b.update(fid=fid, bid=bid, Cout=Cout, bouts=bouts, c_b=c_b)
    return b


class TracedRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta_u, color_o, o_col, w_b, Phi, gidx, c_b, H, W):
        T, K = theta_u.shape[0], theta_u.shape[2]
        key = (T, K)
        if key not in _TR:
            _TR[key] = _init_traces(T, K, Phi, c_b)
        b = _TR[key]
        _set(b["thU"], theta_u); _set(b["col"], color_o); _set(b["oc"], o_col)
        _set(b["bias"], (w_b * c_b)[None, None, :].expand(T, 256, 3))
        _set(b["wb"], torch.full((T, 256, 1), float(w_b)))
        ttnn.execute_trace(DEV, b["fid"], cq_id=0, blocking=False)
        ttnn.synchronize_device(DEV)
        C = dn(b["Cout"]).reshape(T * 256, 3)
        img = torch.zeros(H * W, 3); img[gidx] = C
        ctx.save_for_backward(theta_u, color_o, o_col, w_b, gidx, c_b)
        ctx.shape = (T, K, H, W)
        return img.reshape(H, W, 3)

    @staticmethod
    def backward(ctx, gimg):
        theta_u, color_o, o_col, w_b, gidx, c_b = ctx.saved_tensors
        T, K, H, W = ctx.shape
        b = _TR[(T, K)]
        # re-seat forward inputs (in case another step ran since), set gtiled
        _set(b["thU"], theta_u); _set(b["col"], color_o); _set(b["oc"], o_col)
        _set(b["bias"], (w_b * c_b)[None, None, :].expand(T, 256, 3))
        _set(b["wb"], torch.full((T, 256, 1), float(w_b)))
        _set(b["gt"], gimg.reshape(H * W, 3)[gidx].reshape(T, 256, 3))
        ttnn.execute_trace(DEV, b["bid"], cq_id=0, blocking=False)
        ttnn.synchronize_device(DEV)
        gthU, gcol, goc, gnum, gden = (dn(x) for x in b["bouts"])
        gw_b = ((gnum * c_b[None, None, :]).sum() + gden.sum()).float()
        return gthU, gcol, goc, gw_b, None, None, None, None, None


def render_traced(model, cam, tmap, R=1, K=128):
    thU, col, oc, w_b = _operands(model, cam, tmap, R, K)
    img = TracedRender.apply(thU, col, oc, w_b, tmap.Phi, tmap.gidx, model.c_b, cam.H, cam.W)
    return img.reshape(cam.H, cam.W, 3)


def fit_eval(render_fn, label, args):
    tr_c, tr_i = data.load_blender(args.scene, "train", res=args.res, n=args.n_train, stride=max(1, 100 // args.n_train))
    te_c, te_i = data.load_blender(args.scene, "test", res=args.res, n=args.n_test, stride=max(1, 200 // args.n_test))
    tmap = TileMap(args.res, args.res)
    torch.manual_seed(args.seed)
    m = GaussianModel(args.G, extent=1.5, seed=args.seed)
    from spike.train import DEFAULT_LR
    opt = torch.optim.Adam(m.param_groups(DEFAULT_LR))
    render_fn(m, tr_c[0], tmap, 1, args.K)            # warm (JIT + trace capture) outside timing
    t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for cam, gt in zip(tr_c, tr_i):
            tot = tot + metrics.loss_fn(render_fn(m, cam, tmap, 1, args.K), gt, lambda_ssim=0.2)
        (tot / len(tr_c)).backward()
        opt.step()
    tt = time.perf_counter() - t0
    with torch.no_grad():
        te = sum(float(metrics.psnr(render_fn(m, c, tmap, 1, args.K), g)) for c, g in zip(te_c, te_i)) / len(te_c)
    ms = tt / args.iters * 1e3
    print(f"   [{label:10s}] {ms:7.0f} ms/it | held-out {te:.2f} dB")
    return ms, te


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
    DEV = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
    try:
        CG = ttnn.CoreGrid(x=11, y=10)
        mtb._DEV, mtb.CG, mtb.CKC = DEV, CG, None
        print(f"== trace-integrated training (res={args.res} G={args.G} K={args.K} "
              f"{args.n_train}tr/{args.n_test}te {args.iters}it) ==")
        tr = fit_eval(render_traced, "traced", args)
        if args.compare:
            un = fit_eval(render_binned_device, "untraced", args)
            print(f"\n   speed: traced {tr[0]:.0f} vs untraced {un[0]:.0f} ms/it "
                  f"({un[0]/tr[0]:.2f}x faster) | held-out {tr[1]:.2f} vs {un[1]:.2f} dB")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
