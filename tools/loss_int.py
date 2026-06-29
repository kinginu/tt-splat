"""Debug: isolate the device loss-grad INTEGRATION plumbing (gidx gathers + reshape/transpose +
typecast) from the trainer, since the full trainer hung. Runs the exact loss_grad_dev op chain (1) EAGER
then (2) inside a TRACE, each compared to the host oracle. Pinpoints whether a specific op or trace-capture
is the hang.

Run:  podman-compose --profile hw run --rm hw python3 tools/loss_int.py --res 96
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike import data
from m4_train_binned import TileMap
from loss_manual import gauss_1d, band_matrix, filt as hfilt, loss_manual, C1 as L_C1, C2 as L_C2, LAMBDA as L_LAM

DEV = None
DT = ttnn.float32
BF = ttnn.bfloat16


def u(t, dt=DT):
    return ttnn.from_torch(t.contiguous().float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV)


def dn(t):
    return ttnn.to_torch(t).float()


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--trace", action="store_true", help="also test inside a trace")
    args = ap.parse_args()
    res = args.res

    cams, imgs = data.load_blender("data/nerf_synthetic/ficus", "train", res=res, n=1)
    gt = imgs[0]
    tmap = TileMap(res, res)
    T = tmap.T
    H = W = res
    torch.manual_seed(0)
    C_h = torch.rand(T, 256, 3)        # fake render output

    # ---- host oracle gC ----
    img_flat = torch.zeros(H * W, 3).index_copy(0, tmap.gidx, C_h.reshape(T * 256, 3))
    img_chw = img_flat.reshape(H, W, 3).permute(2, 0, 1).double()
    gt_chw = gt.permute(2, 0, 1).contiguous().double()
    g1d = gauss_1d(); Mh_np = band_matrix(H, g1d); Mw_np = band_matrix(W, g1d)
    _, g_img = loss_manual(img_chw, gt_chw, Mh_np, Mw_np)          # [3,H,W]
    g_img_flat = g_img.permute(1, 2, 0).reshape(H * W, 3)
    gC_host = g_img_flat[tmap.gidx].reshape(T, 256, 3)

    DEV = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        Hout, Win = Mh_np.shape[0], Mw_np.shape[1]
        Ns = float(3 * Mh_np.shape[0] * Mw_np.shape[0]); N = float(3 * H * W)
        Mh_b = u(Mh_np.unsqueeze(0).expand(3, *Mh_np.shape).contiguous())
        MwT_b = u(Mw_np.t().unsqueeze(0).expand(3, Win, Mw_np.shape[0]).contiguous())
        MhT_b = u(Mh_np.t().unsqueeze(0).expand(3, Mh_np.shape[1], Hout).contiguous())
        Mw_b = u(Mw_np.unsqueeze(0).expand(3, *Mw_np.shape).contiguous())
        muy_h = hfilt(gt_chw, Mh_np, Mw_np); muy2_h = muy_h * muy_h
        sy_h = hfilt(gt_chw * gt_chw, Mh_np, Mw_np) - muy2_h
        muy_d, muy2_d, sy_d, y_d = u(muy_h), u(muy2_h), u(sy_h), u(gt_chw)
        inv = torch.empty(H * W, dtype=torch.long); inv[tmap.gidx] = torch.arange(T * 256)
        gidx_u = ttnn.from_torch(tmap.gidx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        inv_u = ttnn.from_torch(inv.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
        C_buf = ttnn.from_torch(C_h.contiguous(), dtype=BF, layout=ttnn.TILE_LAYOUT, device=DEV)
        gC_buf = u(torch.zeros(T, 256, 3), BF)

        def blk(tag):
            print(f"   [{tag}] reshape C ...", flush=True)
            Cf = ttnn.reshape(C_buf, (T * 256, 3))
            print(f"   [{tag}] embedding inv ...", flush=True)
            img = ttnn.embedding(inv_u, Cf)
            print(f"   [{tag}] transpose+reshape ...", flush=True)
            img = ttnn.reshape(ttnn.transpose(img, -2, -1), (3, H, W))
            x = ttnn.typecast(img, DT)
            print(f"   [{tag}] filt ...", flush=True)
            def filt(t):   return ttnn.matmul(ttnn.matmul(Mh_b, t), MwT_b)
            def filt_T(t): return ttnn.matmul(ttnn.matmul(MhT_b, t), Mw_b)
            fx, fx2, fxy = filt(x), filt(ttnn.mul(x, x)), filt(ttnn.mul(x, y_d))
            mux = fx; mux2 = ttnn.mul(mux, mux)
            sx = ttnn.sub(fx2, mux2); sxy = ttnn.sub(fxy, ttnn.mul(mux, muy_d))
            A1 = ttnn.add(ttnn.mul(ttnn.mul(mux, muy_d), 2.0), L_C1)
            A2 = ttnn.add(ttnn.mul(sxy, 2.0), L_C2)
            Bd1 = ttnn.add(ttnn.add(mux2, muy2_d), L_C1); Bd2 = ttnn.add(ttnn.add(sx, sy_d), L_C2)
            D = ttnn.mul(Bd1, Bd2); S = ttnn.div(ttnn.mul(A1, A2), D)
            dS_dfx2 = ttnn.neg(ttnn.div(S, Bd2)); dS_dfxy = ttnn.div(ttnn.mul(A1, 2.0), D)
            term = ttnn.sub(ttnn.mul(muy_d, ttnn.sub(A2, A1)), ttnn.mul(ttnn.mul(S, mux), ttnn.sub(Bd2, Bd1)))
            dS_dfx = ttnn.mul(ttnn.div(term, D), 2.0)
            dmeanS = ttnn.div(ttnn.add(ttnn.add(filt_T(dS_dfx), ttnn.mul(ttnn.mul(x, filt_T(dS_dfx2)), 2.0)),
                                       ttnn.mul(y_d, filt_T(dS_dfxy))), Ns)
            g_ssim = ttnn.mul(dmeanS, -L_LAM)
            g_l1 = ttnn.mul(ttnn.sign(ttnn.sub(x, y_d)), (1.0 - L_LAM) / N)
            gimg = ttnn.add(g_l1, g_ssim)
            print(f"   [{tag}] gather back ...", flush=True)
            gflat = ttnn.transpose(ttnn.reshape(ttnn.typecast(gimg, BF), (3, H * W)), -2, -1)
            gCt = ttnn.reshape(ttnn.embedding(gidx_u, gflat), (T, 256, 3))
            ttnn.copy(gCt, gC_buf)
            print(f"   [{tag}] done", flush=True)
            return gCt

        # (1) EAGER
        blk("eager"); ttnn.synchronize_device(DEV)
        rel = ((dn(gC_buf) - gC_host).norm() / gC_host.norm()).item()
        print(f"   EAGER  gC rel vs host = {rel:.3e}  {'PASS' if rel < 3e-2 else 'FAIL'}", flush=True)

        # (2) TRACE
        if args.trace:
            print("   capturing trace ...", flush=True)
            tid = ttnn.begin_trace_capture(DEV, cq_id=0)
            blk("trace")
            ttnn.end_trace_capture(DEV, tid, cq_id=0); ttnn.synchronize_device(DEV)
            ttnn.execute_trace(DEV, tid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
            rel2 = ((dn(gC_buf) - gC_host).norm() / gC_host.norm()).item()
            print(f"   TRACE  gC rel vs host = {rel2:.3e}  {'PASS' if rel2 < 3e-2 else 'FAIL'}", flush=True)
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
