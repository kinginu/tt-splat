"""GeomDevice -- traced, resident device geometry fwd+bwd (batched over all views, P=N*G).
Wraps the verified device_fwd_core/device_bwd_core (geom_device) in persistent buffers + two ttnn
traces (fwd, bwd). Params/camera live in device buffers (camera set once; params copied each iter); fwd
trace -> conic/mu2d (device); bwd trace -> per-gaussian-view grads. This moves the now-dominant host
geometry on-device and traces it (dispatch collapse).

Oracle: torch autograd (per view, accumulated). Verifies traced output + times traced vs untraced.

Run inside the hw container:
    podman-compose --profile hw run --rm hw python3 tools/geom_resident.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike.model import GaussianModel
import geom_bwd as gbh
from geom_device import device_fwd_core, device_bwd_core, device_fwd, device_bwd, _cams, rel

NEAR = 0.2


class GeomDevice:
    """Resident, traced device geometry. P = n_views * G."""
    def __init__(self, dev, dt, P):
        self.dev, self.dt, self.P = dev, dt, P
        z = lambda: ttnn.from_torch(torch.zeros(P, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        self.cols = tuple(z() for _ in range(10))               # mx,my,mz,qw,qx,qy,qz,sx,sy,sz
        self.Rv = [[z() for _ in range(3)] for _ in range(3)]
        self.tv = [z() for _ in range(3)]
        self.fx, self.fy, self.cx, self.cy = z(), z(), z(), z()
        self.ga, self.gb, self.gc, self.gmux, self.gmuy = z(), z(), z(), z(), z()

        def fwd():
            return device_fwd_core(self.cols, self.Rv, self.tv, self.fx, self.fy, self.cx, self.cy)

        def bwd():
            return device_bwd_core(self.cache, self.ga, self.gb, self.gc, self.gmux, self.gmuy)

        self.conic, self.mu2d, self.cache = fwd(); ttnn.synchronize_device(dev)     # warmup (JIT)
        self.fid = ttnn.begin_trace_capture(dev, cq_id=0)
        self.conic, self.mu2d, self.cache = fwd()
        ttnn.end_trace_capture(dev, self.fid, cq_id=0); ttnn.synchronize_device(dev)
        self.gout = bwd(); ttnn.synchronize_device(dev)
        self.bid = ttnn.begin_trace_capture(dev, cq_id=0)
        self.gout = bwd()
        ttnn.end_trace_capture(dev, self.bid, cq_id=0); ttnn.synchronize_device(dev)

    def _set(self, buf, t):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t.reshape(-1, 1).contiguous(), dtype=self.dt, layout=ttnn.TILE_LAYOUT), buf)

    def set_camera(self, Rv_cols, tv_cols, fx_col, fy_col, cx_col, cy_col):
        for i in range(3):
            for j in range(3):
                self._set(self.Rv[i][j], Rv_cols[i][j])
        for i in range(3):
            self._set(self.tv[i], tv_cols[i])
        self._set(self.fx, fx_col); self._set(self.fy, fy_col)
        self._set(self.cx, cx_col); self._set(self.cy, cy_col)

    def set_params(self, means_t, quat_t, scale_t):
        cs = (means_t[:, 0], means_t[:, 1], means_t[:, 2],
              quat_t[:, 0], quat_t[:, 1], quat_t[:, 2], quat_t[:, 3],
              scale_t[:, 0], scale_t[:, 1], scale_t[:, 2])
        for buf, t in zip(self.cols, cs):
            self._set(buf, t)

    def forward(self):
        ttnn.execute_trace(self.dev, self.fid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.dev)
        return self.conic, self.mu2d                            # device tensors

    def backward(self, gconic_h, gmu2d_h):
        self._set(self.ga, gconic_h[:, 0]); self._set(self.gb, gconic_h[:, 1]); self._set(self.gc, gconic_h[:, 2])
        self._set(self.gmux, gmu2d_h[:, 0]); self._set(self.gmuy, gmu2d_h[:, 1])
        ttnn.execute_trace(self.dev, self.bid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.dev)
        return self.gout                                        # dict of device tensors


def _cam_cols(cams, G, dt):
    def col(vals):
        return torch.cat([torch.full((G, 1), float(x)) for x in vals]).reshape(-1)
    Rv_cols = [[col([c[0][i, j] for c in cams]) for j in range(3)] for i in range(3)]
    tv_cols = [col([c[1][i] for c in cams]) for i in range(3)]
    return Rv_cols, tv_cols, col([c[2] for c in cams]), col([c[3] for c in cams]), \
        col([c[4] for c in cams]), col([c[5] for c in cams])


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=512 * 1024 * 1024)
    try:
        dt = ttnn.float32
        N, G = 6, 8000
        P = N * G
        torch.manual_seed(0)
        m = GaussianModel(G, extent=1.5, seed=0)
        means, quat, logs = m.means3d.detach(), m.quats.detach(), m.log_scales.detach()
        cams = _cams(N)
        gconic_all, gmu_all = torch.randn(P, 3), torch.randn(P, 2)

        # oracle: per-view autograd
        gmeans_a = torch.zeros(G, 3); gquat_a = torch.zeros(G, 4); glogs_a = torch.zeros(G, 3)
        conic_os, mu_os = [], []
        for v, (Rv_t, tv_t, fx, fy, cx, cy) in enumerate(cams):
            mv = means.clone().requires_grad_(True); qv = quat.clone().requires_grad_(True)
            lv = logs.clone().requires_grad_(True)
            Rv = [[float(Rv_t[i, j]) for j in range(3)] for i in range(3)]
            tv = [float(tv_t[i]) for i in range(3)]
            co, mo, _ = gbh.fwd(mv, qv, torch.exp(lv), Rv, tv, fx, fy, cx, cy)
            conic_os.append(co.detach()); mu_os.append(mo.detach())
            (co * gconic_all[v * G:(v + 1) * G]).sum().add_((mo * gmu_all[v * G:(v + 1) * G]).sum()).backward()
            gmeans_a += mv.grad; gquat_a += qv.grad; glogs_a += lv.grad
        conic_o, mu_o = torch.cat(conic_os), torch.cat(mu_os)

        gd = GeomDevice(dev, dt, P)
        Rv_cols, tv_cols, fx_c, fy_c, cx_c, cy_c = _cam_cols(cams, G, dt)
        gd.set_camera(Rv_cols, tv_cols, fx_c, fy_c, cx_c, cy_c)
        means_t, quat_t, scale_t = means.repeat(N, 1), quat.repeat(N, 1), torch.exp(logs).repeat(N, 1)
        gd.set_params(means_t, quat_t, scale_t)

        conic_d, mu_d = gd.forward()
        gout = gd.backward(gconic_all, gmu_all)

        def d(t):
            return ttnn.to_torch(t).float().reshape(-1)
        gm = torch.stack([d(gout["gmx"]), d(gout["gmy"]), d(gout["gmz"])], -1).reshape(N, G, 3).sum(0)
        gq = torch.stack([d(gout["gqw"]), d(gout["gqx"]), d(gout["gqy"]), d(gout["gqz"])], -1).reshape(N, G, 4).sum(0)
        gscale = torch.stack([d(gout["gsx"]), d(gout["gsy"]), d(gout["gsz"])], -1).reshape(N, G, 3).sum(0)
        glogs_m = gscale * torch.exp(logs)
        cd, md = ttnn.to_torch(conic_d).float(), ttnn.to_torch(mu_d).float()

        print(f"== GeomDevice (traced, resident) vs autograd, N={N} P={P} ==")
        print(f"   fwd conic {rel(cd, conic_o):.3e}  mu2d {rel(md, mu_o):.3e}")
        print(f"   gmeans {rel(gm, gmeans_a):.3e}  gquats {rel(gq, gquat_a):.3e}  glog_scales {rel(glogs_m, glogs_a):.3e}")
        worst = max(rel(gm, gmeans_a), rel(gq, gquat_a), rel(glogs_m, glogs_a))
        print(f"   -> worst grad rel {worst:.3e}  ({'PASS' if worst < 0.03 else 'see note'})")

        # timing: traced fwd+bwd vs untraced device_fwd/device_bwd
        Rv_T = [[ttnn.from_torch(Rv_cols[i][j].reshape(-1, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                 for j in range(3)] for i in range(3)]
        tv_T = [ttnn.from_torch(tv_cols[i].reshape(-1, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev) for i in range(3)]
        fxT = ttnn.from_torch(fx_c.reshape(-1, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        fyT = ttnn.from_torch(fy_c.reshape(-1, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        cxT = ttnn.from_torch(cx_c.reshape(-1, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        cyT = ttnn.from_torch(cy_c.reshape(-1, 1), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        N_REP = 20

        def untraced():
            cc, mm2, cache = device_fwd(dev, dt, means_t, quat_t, scale_t, Rv_T, tv_T, fxT, fyT, cxT, cyT)
            device_bwd(dev, dt, cache, gconic_all, gmu_all)
        for _ in range(2):
            untraced()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(N_REP):
            untraced()
        ttnn.synchronize_device(dev)
        tu = (time.perf_counter() - t0) / N_REP

        def traced():
            gd.set_params(means_t, quat_t, scale_t)
            gd.forward()
            gd.backward(gconic_all, gmu_all)
        for _ in range(2):
            traced()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(N_REP):
            traced()
        ttnn.synchronize_device(dev)
        tt = (time.perf_counter() - t0) / N_REP

        # compute-only: params RESIDENT (no set_params copy), only the fwd+bwd trace replays + grad copy
        def traced_resident():
            gd.forward()
            gd.backward(gconic_all, gmu_all)   # still copies 5 grad cols; params resident (no 10 param copies)

        def traced_pure():
            ttnn.execute_trace(dev, gd.fid, cq_id=0, blocking=False)
            ttnn.execute_trace(dev, gd.bid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
        for _ in range(2):
            traced_resident(); traced_pure()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(N_REP):
            traced_resident()
        ttnn.synchronize_device(dev)
        tr = (time.perf_counter() - t0) / N_REP
        t0 = time.perf_counter()
        for _ in range(N_REP):
            traced_pure()
        ttnn.synchronize_device(dev)
        tp = (time.perf_counter() - t0) / N_REP
        print(f"\n   geometry fwd+bwd (P={P}):")
        print(f"     untraced (+copies)        : {tu*1e3:.1f} ms")
        print(f"     traced + set_params copies : {tt*1e3:.1f} ms")
        print(f"     traced, params RESIDENT    : {tr*1e3:.1f} ms  (no 10 param copies; 5 grad copies remain)")
        print(f"     traced, PURE replay        : {tp*1e3:.1f} ms  (no host copies at all)")
        print(f"   host geometry (geom_bwd, 6 views) reference ~45 ms")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
