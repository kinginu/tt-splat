"""Device (ttnn) spherical-harmonics colour eval, deg-3, verified against spike.sh.eval_sh_color.

The foundational brick for putting SH colour fully on-device (coeffs = resident params, optimised
on-device like every other gaussian param -- no host SH eval / no host round-trip). color[ch] =
relu( sum_l b_l(viewdir) * coeff_l[ch] + 0.5 ), all ttnn. Oracle: the CPU eval at the bf16 floor.

Run (hw): podman ... python3 tools/sh_device.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ttnn
from spike import sh

BF, DT, TILE = ttnn.bfloat16, ttnn.float32, ttnn.TILE_LAYOUT
C0, C1, C2, C3 = sh.C0, sh.C1, sh.C2, sh.C3


def sh_color_device(vx, vy, vz, coeff, dev):
    """vx/vy/vz [G,1] device (view dir, un-normalised); coeff = list of 3 [G,16] device (r,g,b coeffs).
    Returns [color_r, color_g, color_b] each [G,1] device = relu(0.5 + sum_l b_l*coeff_l[ch])."""
    M, A, S = ttnn.mul, ttnn.add, ttnn.sub
    # --- normalise the direction ---
    n2 = A(A(M(vx, vx), M(vy, vy)), M(vz, vz))
    inv = ttnn.rsqrt(A(n2, 1e-12))
    x, y, z = M(vx, inv), M(vy, inv), M(vz, inv)
    xx, yy, zz, xy, yz, xz = M(x, x), M(y, y), M(z, z), M(x, y), M(y, z), M(x, z)
    one = A(M(x, 0.0), 1.0)
    # --- the 16 basis terms b_l [G,1] (constants folded in) ---
    b = [M(one, C0),
         M(y, -C1), M(z, C1), M(x, -C1),
         M(xy, C2[0]), M(yz, C2[1]), M(S(M(zz, 2.0), A(xx, yy)), C2[2]), M(xz, C2[3]), M(S(xx, yy), C2[4]),
         M(M(y, S(M(xx, 3.0), yy)), C3[0]),
         M(M(xy, z), C3[1]),
         M(M(y, S(M(zz, 4.0), A(xx, yy))), C3[2]),
         M(M(z, S(M(zz, 2.0), A(M(xx, 3.0), M(yy, 3.0)))), C3[3]),
         M(M(x, S(M(zz, 4.0), A(xx, yy))), C3[4]),
         M(M(z, S(xx, yy)), C3[5]),
         M(M(x, S(xx, M(yy, 3.0))), C3[6])]
    bmat = ttnn.concat(b, dim=1)                                   # [G,16]
    out = []
    for cf in coeff:                                              # per channel: relu(0.5 + sum_l b_l*cf_l)
        s = ttnn.sum(M(bmat, cf), dim=1, keepdim=True)            # [G,1]
        out.append(ttnn.relu(A(s, 0.5)))
    return out, bmat


def sh_grad_device(bmat, color, gcolor):
    """Backward: dL/dcoeff_l[ch] = gcolor[ch] * b_l * (color[ch]>0). Returns 3 x [G,16] device.
    color/gcolor = lists of 3 [G,1]; bmat [G,16] from the forward (the basis, reused)."""
    g = []
    for col, gc in zip(color, gcolor):
        gg = ttnn.mul(gc, ttnn.gtz(col))                         # [G,1] grad through the relu gate
        g.append(ttnn.mul(bmat, gg))                             # [G,16] = b_l * gg  (broadcast)
    return g


def main():
    torch.manual_seed(0)
    G = 4000
    coeffs = torch.randn(G, 16, 3) * 0.5
    coeffs[:, 0, :] = torch.randn(G, 3) * 1.5                     # DC term larger
    vdir = torch.randn(G, 3)
    color_ref = sh.eval_sh_color(3, coeffs, vdir)                 # CPU oracle [G,3]

    dev = ttnn.open_device(device_id=0)
    try:
        u = lambda t: ttnn.from_torch(t.reshape(G, -1).contiguous(), dtype=DT, layout=TILE, device=dev)
        vx, vy, vz = u(vdir[:, 0]), u(vdir[:, 1]), u(vdir[:, 2])
        coeff = [u(coeffs[:, :, ch]) for ch in range(3)]          # [G,16] per channel
        out, bmat = sh_color_device(vx, vy, vz, coeff, dev)
        color_dev = torch.cat([ttnn.to_torch(o).float() for o in out], dim=1)   # [G,3]
        rel = ((color_dev - color_ref).norm() / color_ref.norm().clamp(min=1e-9)).item()
        print(f"FWD device SH deg3 vs spike.eval_sh_color: rel-err {rel:.4f}  max|err| "
              f"{(color_dev - color_ref).abs().max().item():.4f}")

        # ---- backward: device dL/dcoeff vs autograd ----
        gcolor = torch.randn(G, 3)
        gc_dev = [u(gcolor[:, ch]) for ch in range(3)]
        dcoeff = sh_grad_device(bmat, out, gc_dev)                # 3 x [G,16]
        ct = coeffs.clone().requires_grad_(True)
        sh.eval_sh_color(3, ct, vdir).backward(gcolor)
        gref = ct.grad                                            # [G,16,3]
        relb = max(((ttnn.to_torch(dcoeff[ch]).float() - gref[:, :, ch]).norm()
                    / gref[:, :, ch].norm().clamp(min=1e-9)).item() for ch in range(3))
        print(f"BWD device dL/dcoeff vs autograd: rel-err {relb:.4f}")
        print("PASS (bf16 floor)" if rel < 0.05 and relb < 0.05 else "FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
