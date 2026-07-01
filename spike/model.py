"""The learnable gaussian set for the matrix-native reference.

One model holds every parameter; each arm reads the subset it needs. Init follows the spec
with matched initial opacity across arms (opacity_sh DC inits to the same o=0.1 as opacity_raw,
so arm C/C0 are not advantaged at init). `gt=True` makes a denser/opaquer/coloured scene for
the synthetic self-consistency oracle.
"""
import math

import torch
import torch.nn as nn

from . import sh


class GaussianModel(nn.Module):
    def __init__(self, G, extent=1.3, dtype=torch.float32, device="cpu", seed=0,
                 gt=False, init_scale=0.05, depth_tau=3.0):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)

        def rand(*s):
            return torch.rand(*s, generator=g, dtype=dtype, device=device)

        def randn(*s):
            return torch.randn(*s, generator=g, dtype=dtype, device=device)

        self.means3d = nn.Parameter((rand(G, 3) * 2 - 1) * extent)
        self.log_scales = nn.Parameter(
            torch.full((G, 3), math.log(init_scale), dtype=dtype, device=device) + 0.1 * randn(G, 3))
        q = torch.zeros(G, 4, dtype=dtype, device=device)
        q[:, 0] = 1.0
        self.quats = nn.Parameter(q + 0.1 * randn(G, 4))

        o0 = 0.7 if gt else 0.1
        self.opacity_raw = nn.Parameter(
            torch.full((G,), math.log(o0 / (1 - o0)), dtype=dtype, device=device))
        osh = torch.zeros(G, 9, dtype=dtype, device=device)
        osh[:, 0] = math.log(o0 / (1 - o0)) / sh.C0          # DC -> o=o0, matches opacity_raw
        self.opacity_sh = nn.Parameter(osh)

        # color = clamp(0.5 + C0*dc, 0): gt gets varied colors, fit starts near gray.
        self.color_dc = nn.Parameter((randn(G, 3) * 1.5) if gt else (0.01 * randn(G, 3)))
        # view-dependent colour SH rest (deg 1-3 = 15 coeffs x RGB); zeros => DC-only behaviour.
        self.color_rest = nn.Parameter(torch.zeros(G, 15, 3, dtype=dtype, device=device))

        self.w_b_raw = nn.Parameter(torch.tensor(-3.0, dtype=dtype, device=device))    # w_b≈0.049
        self.depth_beta = nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.depth_tau = nn.Parameter(torch.tensor(depth_tau, dtype=dtype, device=device))
        self.softz_beta = nn.Parameter(torch.tensor(4.0, dtype=dtype, device=device))  # arm SZ gate sharpness
        self.softmin_tau = nn.Parameter(torch.tensor(2.0, dtype=dtype, device=device))  # arm SM temperature
        self.bp_tau = nn.Parameter(torch.tensor(2.0, dtype=dtype, device=device))   # arm BP temperature
        self.e_tau = nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))   # arm E soft surrogate temperature
        self.pw_tau = nn.Parameter(torch.tensor(0.01, dtype=dtype, device=device))  # arm PW pairwise soft-compare temperature
        self.register_buffer("c_b", torch.ones(3, dtype=dtype, device=device))         # white bg

    @property
    def G(self):
        return self.means3d.shape[0]

    @torch.no_grad()
    def init_from_points(self, xyz, rgb, seed=0):
        """Initialise gaussians from a COLMAP sparse cloud (real scenes need point-init, not random).
        means <- points (sampled/padded to G), color_dc <- inverse-SH of rgb, scales <- median kNN dist."""
        G = self.G
        g = torch.Generator().manual_seed(seed)
        P = xyz.shape[0]
        if P >= G:
            sel = torch.randperm(P, generator=g)[:G]
            pts, cols = xyz[sel], rgb[sel]
        else:                                                    # pad: jitter existing points
            extra = G - P
            j = torch.randint(0, P, (extra,), generator=g)
            span = (xyz.max(0).values - xyz.min(0).values).norm() * 0.01
            pts = torch.cat([xyz, xyz[j] + torch.randn(extra, 3, generator=g) * span], 0)
            cols = torch.cat([rgb, rgb[j]], 0)
        # global init scale = median nearest-neighbour distance over a subset (3DGS-style, cheap)
        sub = pts[torch.randperm(G, generator=g)[:min(G, 4000)]]
        d = torch.cdist(sub, sub); d.fill_diagonal_(float("inf"))
        nn = d.min(1).values.median().clamp(min=1e-4)
        self.means3d.copy_(pts.to(self.means3d.dtype))
        self.log_scales.copy_(torch.full((G, 3), float(torch.log(nn)), dtype=self.log_scales.dtype))
        self.color_dc.copy_(((cols - 0.5) / sh.C0).to(self.color_dc.dtype))   # render: relu(C0*dc+0.5)≈rgb
        return self

    def param_groups(self, lr):
        """Per-parameter Adam groups. `lr` is a dict of learning rates (see train.DEFAULT_LR)."""
        return [
            {"params": [self.means3d], "lr": lr["means"]},
            {"params": [self.log_scales], "lr": lr["scales"]},
            {"params": [self.quats], "lr": lr["quats"]},
            {"params": [self.opacity_raw], "lr": lr["opacity"]},
            {"params": [self.opacity_sh], "lr": lr["opacity_sh"]},
            {"params": [self.color_dc], "lr": lr["color"]},
            {"params": [self.color_rest], "lr": lr.get("color_rest", lr["color"] / 20.0)},
            {"params": [self.w_b_raw], "lr": lr["wb"]},
            {"params": [self.depth_beta, self.depth_tau], "lr": lr["depth"]},
            {"params": [self.softz_beta], "lr": lr["depth"]},
            {"params": [self.softmin_tau], "lr": lr["depth"]},
            {"params": [self.bp_tau], "lr": lr["depth"]},
            {"params": [self.e_tau], "lr": lr["depth"]},
            {"params": [self.pw_tau], "lr": lr["depth"]},
        ]
