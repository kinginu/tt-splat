"""Fast-sweep keystone: verify a TRACED geom_fwd that reads the camera from DEVICE BUFFERS (not baked
python scalars) produces the correct per-view conic/mu2d when the camera buffers are swapped per iter.
This is the one new risk for the multi-view fast-path sweep (device_fwd_core already accepts tensor camera,
verified eager in geom_device.run_batched; here we confirm it works inside a trace + buffer swap).

Run:  podman-compose --profile hw run --rm hw python3 tools/cambuf_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import ttnn

from spike.model import GaussianModel
from geom_device import device_fwd_core, device_fwd, rel

DEV = None
DT = ttnn.float32


def u(t):
    return ttnn.from_torch(t.reshape(-1, 1).contiguous().float(), dtype=DT, layout=ttnn.TILE_LAYOUT, device=DEV)


def uh(t):   # HOST tensor (no device) -- valid source for copy_host_to_device_tensor
    return ttnn.from_torch(t.reshape(-1, 1).contiguous().float(), dtype=DT, layout=ttnn.TILE_LAYOUT)


def main():
    global DEV
    G = 2000
    torch.manual_seed(0)
    m = GaussianModel(G, extent=1.5, seed=0)
    means, quat = m.means3d, m.quats
    scale = torch.exp(m.log_scales)

    def cam(v):
        a = 0.3 * v
        Rv = [[float(torch.cos(torch.tensor(a))), 0.1, float(-torch.sin(torch.tensor(a)))],
              [-0.08, 0.95, 0.2], [float(torch.sin(torch.tensor(a))), -0.18, 0.98]]
        tv = [0.1 * v, -0.2, 4.0 + 0.1 * v]
        return Rv, tv, 128 * 1.2, 128 * 1.2, 64.0, 64.0

    DEV = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
    try:
        # resident param buffers
        P = dict(mx=u(means[:, 0]), my=u(means[:, 1]), mz=u(means[:, 2]),
                 qw=u(quat[:, 0]), qx=u(quat[:, 1]), qy=u(quat[:, 2]), qz=u(quat[:, 3]),
                 sx=u(scale[:, 0]), sy=u(scale[:, 1]), sz=u(scale[:, 2]))
        # camera BUFFERS [G,1] (filled per view)
        Rvb = [[u(torch.zeros(G)) for _ in range(3)] for _ in range(3)]
        tvb = [u(torch.zeros(G)) for _ in range(3)]
        fxb, fyb, cxb, cyb = u(torch.zeros(G)), u(torch.zeros(G)), u(torch.zeros(G)), u(torch.zeros(G))

        def setcam(v):
            Rv, tv, fx, fy, cx, cy = cam(v)
            for i in range(3):
                for j in range(3):
                    ttnn.copy_host_to_device_tensor(uh(torch.full((G,), Rv[i][j])), Rvb[i][j])
                ttnn.copy_host_to_device_tensor(uh(torch.full((G,), tv[i])), tvb[i])
            for buf, val in ((fxb, fx), (fyb, fy), (cxb, cx), (cyb, cy)):
                ttnn.copy_host_to_device_tensor(uh(torch.full((G,), float(val))), buf)

        def geom():
            cols = (P["mx"], P["my"], P["mz"], P["qw"], P["qx"], P["qy"], P["qz"], P["sx"], P["sy"], P["sz"])
            conic, mu2d, _ = device_fwd_core(cols, Rvb, tvb, fxb, fyb, cxb, cyb)
            return conic, mu2d

        # warmup (JIT) BEFORE capture
        setcam(0)
        conic, mu2d = geom()
        ttnn.synchronize_device(DEV)
        # capture
        gfid = ttnn.begin_trace_capture(DEV, cq_id=0)
        conic, mu2d = geom()
        ttnn.end_trace_capture(DEV, gfid, cq_id=0); ttnn.synchronize_device(DEV)

        ok = True
        for v in (0, 3, 7):
            setcam(v)
            ttnn.execute_trace(DEV, gfid, cq_id=0, blocking=False); ttnn.synchronize_device(DEV)
            cd, md = ttnn.to_torch(conic).float(), ttnn.to_torch(mu2d).float()
            # oracle: baked-scalar device_fwd for this view
            Rv, tv, fx, fy, cx, cy = cam(v)
            co, mo, _ = device_fwd(DEV, DT, means, quat, scale, Rv, tv, fx, fy, cx, cy)
            rc, rm = rel(cd, ttnn.to_torch(co).float()), rel(md, ttnn.to_torch(mo).float())
            print(f"   view {v}: conic rel {rc:.3e}  mu2d rel {rm:.3e}  {'ok' if max(rc, rm) < 1e-3 else 'MISMATCH'}")
            ok = ok and max(rc, rm) < 1e-3
        print(f"== camera-buffer traced geom: {'PASS' if ok else 'FAIL'} ==")
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
