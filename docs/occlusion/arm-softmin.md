# Hand-off — arm `SM` (softmin / Boltzmann depth weight) · candidate #2

> **One Sonnet agent owns this file.** Read `01-arm-authoring-contract.md` (host scaffolding,
> shared for all arms) and `02-device-map.md` (device interfaces) first. §3-gated: you ADD arm `SM`
> alongside the locked arm A; you change no default and remove nothing. Branch: `feat/arm-softmin`.
>
> **Device difficulty: LOW** — this is an arm-B clone (per-gaussian multiplier folded into `keo`),
> the easiest device port. Do it first; it also de-risks the shared device pattern for B′.

## What it is

A per-gaussian **Boltzmann front-weight**. Where arm B uses a sigmoid front-bias `σ(β(τ−z))`, softmin
uses an exponential:

```
ρᵢ = exp( −(zᵢ − z_ref) / τ )            # z_ref = detached anchor (min or median of visible z)
Wᵢ = oᵢ · w_geo · ρᵢ                      # fold into the WSR weight column; _wsr as usual
```

`z_ref` subtraction = numerical stability + makes the nearest gaussian's weight ≈ 1 (this folds in
candidate #4's "argmin anchor": front-relative depth, so the arm does not depend on absolute scene
depth). One learnable scalar **`τ > 0`**:

- **τ → 0** ⇒ front-takes-all (hard occlusion: only `z ≈ z_ref` survives).
- **τ → ∞** ⇒ uniform `ρ ≡ 1` ⇒ exactly arm A (WSR).

So a single scalar **continuously connects occlusion ↔ WSR**, and the τ→0 limit is *provably* correct
front-takes-all — the advantage over arm B's monotone-but-unnormalised sigmoid. This is the doc's
"first practical pick"; get it landed and measured before B′/moment.

## Host implementation (per `01-arm-authoring-contract.md`)

**`spike/arms.py`** — add:

```python
def blend_SM(w_geo, opacity_raw, depth, softmin_tau, color, w_b, c_b):
    """Softmin / Boltzmann depth weight (candidate #2): per-gaussian front-weight
    rho = exp(-(z - z_ref)/tau), z_ref = detached min(z). tau->0 = front-takes-all (hard
    occlusion), tau->inf = uniform = arm A. Sort-free (per-gaussian exp = lane-wise), z stays
    ATTACHED so occlusion produces a z-force (C3). One learnable scalar tau>0."""
    o = torch.sigmoid(opacity_raw)
    tau = softmin_tau.clamp(min=1e-3)
    z_ref = depth.min().detach()                       # front anchor; subgradient detached
    rho = torch.exp(-(depth - z_ref) / tau)            # [G]
    return _wsr(o[None, :] * w_geo * rho[None, :], color, w_b, c_b)
```

Keep `depth` **attached** (C3). Only `z_ref` is detached.

**`spike/model.py`** — add scalar + register:
```python
self.softmin_tau = nn.Parameter(torch.tensor(2.0, dtype=dtype, device=device))   # ~ scene z-range
# in param_groups:
{"params": [self.softmin_tau], "lr": lr["depth"]},
```
Init `τ = 2.0` sits in the soft regime (scene z ≈ [1,5], range ≈ 4) so it starts near arm-A behaviour
and learns *down* toward occlusion. (Do NOT init tiny — that starts at hard front-takes-all and
starves the early gradient.)

**`spike/render.py`** — `ARMS += ("SM",)`; dispatch:
```python
elif arm == "SM":
    img = arms.blend_SM(w_geo, model.opacity_raw, depth, model.softmin_tau, color, w_b, c_b)
```

**`spike/m05_spike.py`** — add `"SM"` to `CANDIDATE_ARMS`.

**`spike/tests/test_toy_occlusion.py`** — add (use a moderately small τ so the 2-gaussian probe
occludes; the near gaussian z=2 is the anchor so its ρ=1, far z=5 gets ρ=exp(−3/τ)):
```python
def test_softmin_occludes():
    s = _setup(); tau = _f(0.8)
    C = arms.blend_SM(s["w_geo"], s["opacity_raw"], s["depth"], tau, s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["red"]).norm() < (C - s["blue"]).norm(), C
    assert C[0].item() > 0.85 and C[2].item() < 0.15, C
    # smaller tau -> harder occlusion (monotone)
    C2 = arms.blend_SM(s["w_geo"], s["opacity_raw"], s["depth"], _f(0.3), s["color"], s["w_b"], s["c_b"])[0]
    assert C2[0].item() >= C[0].item()

def test_softmin_depth_order():
    s = _setup(); zsw = _f([5.0, 2.0])
    C = arms.blend_SM(s["w_geo"], s["opacity_raw"], zsw, _f(0.8), s["color"], s["w_b"], s["c_b"])[0]
    assert (C - s["blue"]).norm() < (C - s["red"]).norm(), C

def test_softmin_zgrad():
    s = _setup(); z = s["depth"].clone().requires_grad_(True)
    C = arms.blend_SM(s["w_geo"], s["opacity_raw"], z, _f(0.8), s["color"], s["w_b"], s["c_b"])
    (g,) = torch.autograd.grad(C.sum(), z)
    assert g.abs().sum() > 0
    # the FAR gaussian must feel a push (its rho<1, so dL/dz_far != 0)
    assert g[1].abs().item() > 0
```

## Analytic backward (for the oracle + device)

With `ρᵢ = exp(−(zᵢ − z_ref)/τ)` and the WSR hook `grad_ρᵢ = gkeo_i · keep_i · o_i`:

```
∂ρᵢ/∂zᵢ = −ρᵢ / τ
∂ρᵢ/∂τ  =  ρᵢ · (zᵢ − z_ref) / τ²
grad_zᵢ  +=  grad_ρᵢ · (−ρᵢ/τ)               # ADD to the mcz→means gradient (see device note)
grad_τ    =  Σᵢ  grad_ρᵢ · ρᵢ · (zᵢ − z_ref) / τ²
```

**Oracle** — `tools/softmin_oracle.py`, mirror `tools/m8_depthweight_oracle.py`: autograd through
`blend_SM` for `grad_τ` (and `grad_z`), vs your manual formulas; assert `rel < 1e-5`. PASS/FAIL exit.

## Device implementation (per `02-device-map.md`)

Per-gaussian multiplier ⇒ the **arm-B fold-in**, near-identical to the existing depth-weight path.

**Forward** — in `geom_fwd()` (trace `gfid`), replace/parallel arm-B's `rho`:
```python
# z_ref: a detached anchor. Compute once at warmup (median/min of dn(mcz)) -> z_ref scalar broadcast
# into a [G,1] buffer zref_buf; refresh every ~500 iters (avoids a per-iter device->host min/sync).
rho = ttnn.exp(ttnn.mul(ttnn.neg(A(cache["mcz"], ttnn.neg(zref_buf))), inv_tau_buf))  # exp(-(z-zref)/tau)
keo = M(M(keep, o), rho)
```
Store `inv_tau_buf = u(full((G,), 1.0/tau))` (broadcast `1/τ`) and `zref_buf`. (`exp` is a lane-wise
SFPU op — cheap, no GEMM.)

**Backward** — in `geom_bwd()`, off `gkeo`:
```python
grad_rho = M(M(gkeo, keep), o)                       # [G,1]
# tau grad (host-Adam scalar):
gtau_col = M(grad_rho, M(rho, M(A(cache["mcz"], ttnn.neg(zref_buf)), inv_tau2_buf)))  # grad_rho·rho·(z-zref)/tau^2
ttnn.copy(gtau_col, gtau_buf)                         # host reduces .sum() in adam step
# z-force (C3) — ADD into the mcz gradient that geom_bwd feeds to means3d:
gmcz_occ = M(grad_rho, M(rho, ttnn.neg(inv_tau_buf))) # grad_rho·(-rho/tau)
#   accumulate gmcz_occ into the existing mcz-gradient accumulator BEFORE device_bwd_core maps it to means.
```

> **Critical C3 wiring (applies to softmin/B/B′ on device):** the host arm gets `dL/dz→means` from
> autograd for free; the device must explicitly **add `gmcz_occ` to the mcz gradient** consumed by
> `device_bwd_core`'s z→means path. Verify arm B already does this; if arm B only learns τ (no
> means z-force), fixing that is in-scope here and shared with B′. Add a toy/oracle check that the
> means feel the occlusion z-force on device (compare device `grad_means` vs host autograd).

**Host-Adam scalar `τ`** — add `--smtau-0` (init 2.0) + `--lr-smtau`; allocate `inv_tau_buf`,
`inv_tau2_buf` (=1/τ²), `gtau_buf`, host `{val,m,v}`; step in `adam_bt`-style after `gbid`+sync;
`setbuf` the refreshed `1/τ`, `1/τ²`. `zref_buf` refreshed at warmup + every ~500 iters.

**Device oracle** — extend `tools/softmin_oracle.py` (or a device variant) to check the device
`grad_τ`/`grad_mcz` against host autograd on a small G.

## Sizing / cost
- Forward +1 `exp` + 1 mul per gaussian (lane-wise). Backward +a few muls + the existing `.sum()`.
- **Negligible** vs the GEMM hot path — strictly cheaper than B′ (no `[G,G]`). No re-capture of
  `rfid`/`rbid` (per-gaussian only) — only `gfid`/`gbid` get the new ops, added before capture.

## Acceptance
- Host: `test_toy_occlusion` (incl. 3 new) passes; `softmin_oracle.py` PASS; arm A unchanged (run A
  alone, bit-identical); `m05_spike --arms A,B,SM,D` table includes `SM`.
- Device: `sweep_resident.py --arm-softmin` (new flag, mirror `--depth-weight`) runs a smoke
  (`--G 1000 --iters 30 --res 128`) on hardware, loss decreases, device grads match host oracle on
  small G, and the means feel the z-force.
- Result note: SM vs A vs B holdout PSNR on the lego solid-occluder views; does the τ→0 limit help.

## Risks / open
- **z_ref staleness**: a warmup-frozen anchor can drift as geometry moves; refresh cadence is a knob
  (measure 100 vs 500 iters). A per-iter device min costs one sync — only use if staleness hurts.
- τ collapsing to 0 (degenerate hard) — clamp `τ ≥ 1e-3`; watch for instability and report.
