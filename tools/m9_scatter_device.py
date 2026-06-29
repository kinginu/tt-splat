"""device gather-reduce scatter (ttnn) vs host index_add. Validates the on-device path before
integration. Mirrors the bin->gaussian scatter as: embedding(inv_u, grad_pad) -> sum over Smax.

embedding weight must be bf16 (ttnn constraint); we typecast the gathered result to fp32 before the
sum so the reduction accumulates in fp32 (matching host: bf16 grads -> fp32 index_add).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import ttnn
from m9_scatter_oracle import build_inv

DT, BF = ttnn.float32, ttnn.bfloat16


def main():
    torch.manual_seed(0)
    G, T, K, R = 2000, 64, 128, 1
    Smax = (2 * R + 1) ** 2
    C = 9
    TK = T * K

    idx = torch.zeros(T, K, dtype=torch.long); valid = torch.zeros(T, K, dtype=torch.bool)
    fill = torch.randint(0, K, (T,))
    for t in range(T):
        n = int(fill[t]); idx[t, :n] = torch.randint(0, G, (n,)); valid[t, :n] = True
    grad = torch.randn(T, K, C)
    vmask = valid[..., None].float()
    grad_m = (grad * vmask)                                        # mask invalid slots to 0

    # quantize to bf16 to mimic the real render-bwd grads, so host ref and device see the SAME inputs
    grad_bf = grad_m.to(torch.bfloat16).float()
    ref = torch.zeros(G, C).index_add_(0, idx.reshape(-1), grad_bf.reshape(-1, C))   # host (fp32 accum)

    inv = build_inv(idx, valid, G, Smax)                          # [G,Smax] long, pad=TK

    dev = ttnn.open_device(device_id=0)
    try:
        # grad_pad [TK+1, C] bf16 TILE  (last row = zero sentinel)
        grad_pad = torch.cat([grad_m.reshape(TK, C), torch.zeros(1, C)], 0)
        gp = ttnn.from_torch(grad_pad.contiguous(), dtype=BF, layout=ttnn.TILE_LAYOUT, device=dev)
        inv_u = ttnn.from_torch(inv.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        emb = ttnn.embedding(inv_u, gp)                           # [G, Smax, C] bf16
        emb = ttnn.typecast(emb, DT)                              # accumulate in fp32
        g = ttnn.sum(emb, dim=1, keepdim=False)                   # [G, C]
        got = ttnn.to_torch(g).float().reshape(G, C)
        # verify the 9-way column split (how geom_bwd consumes it: a,b,c,mx,my,col0,1,2,op)
        cols = ttnn.split(g, 1, dim=1)                            # list of [G,1]
        split_ok = len(cols) == C and all(
            (ttnn.to_torch(cols[i]).float().reshape(G) - got[:, i]).abs().max().item() < 1e-4 for i in range(C))
        print(f"  ttnn.split -> {len(cols)} cols, match={split_ok}")
    finally:
        ttnn.close_device(dev)

    err = (got - ref).abs().max().item(); rel = err / (ref.abs().max().item() + 1e-12)
    print(f"G={G} Smax={Smax} C={C}  max|err|={err:.3e}  rel={rel:.3e}")
    multi = int(torch.bincount(idx.reshape(-1)[valid.reshape(-1)], minlength=G).argmax())
    print(f"  busiest gaussian {multi}: ref={[round(x,4) for x in ref[multi,:3].tolist()]} "
          f"got={[round(x,4) for x in got[multi,:3].tolist()]}")
    ok = rel < 5e-2          # bf16 gather floor
    print("DEVICE SCATTER", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
