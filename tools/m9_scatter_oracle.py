"""Oracle: device-style gather-reduce scatter == host torch.index_add (the bin->gaussian scatter).

Current host scatter (resident_traced / sweep_resident):
    flat = idx.reshape(-1)                                   # [T*K] gaussian id per slot
    g_param = zeros(G,C).index_add_(0, flat, grad_slots.reshape(-1,C))   # sum slots per gaussian

Device plan (gather-reduce, no atomics, GEMM/gather-friendly): build at binning time the TRANSPOSE
    inv[G, Smax] = the valid-slot positions for each gaussian (Smax=(2R+1)^2, sentinel=T*K -> zero row)
then per iter on-device:
    grad_pad = concat([grad_slots.reshape(T*K,C), zeros(1,C)], 0)   # [T*K+1, C]
    g_param  = sum_over_Smax( grad_pad[inv] )                       # [G, Smax, C] -> [G, C]

This oracle validates the INDEX LOGIC (inv construction + gather + reduce) on CPU before the device port.
We mask ALL channels by `valid` (invalid slots excluded via inv); the reference index_add masks the same
way (clean semantics — invalid slots contribute 0 to every channel, incl. gct/gmt).
"""
import torch


def build_inv(idx, valid, G, Smax):
    """idx[T,K] gaussian-id per slot, valid[T,K] bool -> inv[G,Smax] long (slot positions; pad=T*K)."""
    TK = idx.numel()
    flat_idx = idx.reshape(-1)
    flat_val = valid.reshape(-1)
    slots = torch.nonzero(flat_val, as_tuple=False).squeeze(1)      # valid slot positions [Nv]
    g_of = flat_idx[slots]                                          # gaussian per valid slot [Nv]
    order = torch.argsort(g_of, stable=True)                       # group by gaussian
    g_s, slot_s = g_of[order], slots[order]
    counts = torch.bincount(g_of, minlength=G)                     # valid slots per gaussian
    assert int(counts.max()) <= Smax, f"overflow: a gaussian has {int(counts.max())} slots > Smax {Smax}"
    csum = torch.cumsum(counts, 0) - counts                        # group start offset
    within = torch.arange(len(g_s)) - csum[g_s]                    # rank within gaussian group
    inv = torch.full((G, Smax), TK, dtype=torch.long)             # sentinel -> zero row
    inv[g_s, within] = slot_s
    return inv


def gather_reduce(grad_slots, inv, C):
    """grad_slots[T,K,C] + inv[G,Smax] -> g_param[G,C] (device-style gather + sum)."""
    TK = grad_slots.shape[0] * grad_slots.shape[1]
    grad_pad = torch.cat([grad_slots.reshape(TK, C), torch.zeros(1, C)], 0)   # [TK+1, C]
    return grad_pad[inv.reshape(-1)].reshape(inv.shape[0], inv.shape[1], C).sum(dim=1)  # [G,C]


def main():
    torch.manual_seed(0)
    G, T, K, R = 2000, 64, 128, 1
    Smax = (2 * R + 1) ** 2
    C = 9
    # synthesise a realistic binning: each gaussian lands in <=Smax tiles, fixed-K overflow drop
    idx = torch.zeros(T, K, dtype=torch.long)
    valid = torch.zeros(T, K, dtype=torch.bool)
    fill = torch.randint(0, K, (T,))                       # how many slots used per tile (<=K)
    for t in range(T):
        n = int(fill[t])
        idx[t, :n] = torch.randint(0, G, (n,))
        valid[t, :n] = True
    grad_slots = torch.randn(T, K, C)

    # reference: host index_add, masked by valid (clean semantics)
    vmask = valid[..., None].float()
    flat = idx.reshape(-1)
    ref = torch.zeros(G, C).index_add_(0, flat, (grad_slots * vmask).reshape(-1, C))

    # device-style gather-reduce (mask by valid first so invalid slots are 0 everywhere)
    inv = build_inv(idx, valid, G, Smax)
    got = gather_reduce(grad_slots * vmask, inv, C)

    err = (got - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-12)
    print(f"G={G} T={T} K={K} Smax={Smax}  max|err|={err:.2e}  rel={rel:.2e}")
    # spot-check a gaussian appearing in multiple slots
    multi = int(torch.bincount(flat[valid.reshape(-1)], minlength=G).argmax())
    print(f"  busiest gaussian {multi}: ref={ref[multi,:3].tolist()} got={got[multi,:3].tolist()}")
    ok = err < 1e-4
    print("ORACLE", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
