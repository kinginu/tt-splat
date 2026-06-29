# The math of the Blackhole port — how 3DGS collapses to GEMMs

This document derives, from scratch, why every heavy stage of our 3D Gaussian Splatting
renderer+trainer (our **matrix-native 3DGS** approach) is a **matrix multiply** — forward *and*
backward — and what the two small non-matrix residuals are. It is the formal companion to the one-line
claim in the README. ("matrix-native 3DGS" is the project's internal "matrix-native".)

Notation:

- `P` — pixels in a tile (256 for a 16×16 tile). `G` — gaussians. `T` — tiles.
- `Φ` — per-pixel monomial basis, shared across gaussians.
- `θ` — per-gaussian 6-vector (conic + mean, folded).
- `Q` — the quadratic form `dᵀΣ⁻¹d`. `w` — the splat weight. `C` — the rendered color.
- `o` — opacity, `c` — color (SH-decoded).

The spine oracle for everything below is the matrix-native PyTorch reference in `spike/`
(`geometry.py`, `forward.py`, `arms.py`, `backward.py`); every device kernel is diffed against it.

---

## 0. Why this is even possible — one sentence

> **Differentiation is a linear map; a linear map is a matrix.** So if we can write the *forward*
> render as a chain of matrix multiplies, the *backward* pass is automatically a chain of transposed
> matrix multiplies. The entire job is therefore to find a form of the forward render that is GEMMs.

Standard 3DGS cannot do this because its splat weight is `exp(−Q)`, and `exp` does **not** factor into
(pixel part) × (gaussian part). Matrix-native 3DGS replaces `exp` with a polynomial precisely so the factorization
survives all the way through.

The per-core throughput ratio on Tensix is the entire reason this matters:

```
MVMUL  ≈ 16 × ELWMUL  ≈ 64 × SFPU      (matrix engine ≫ elementwise ≫ transcendental)
```

Standard 3DGS spends its time on SFPU (`exp`), a depth sort (branches), and irregular scatter — Tensix's
three weakest paths. Matrix-native 3DGS moves the dominant per-pixel-per-gaussian work onto the matrix engine.

---

## 1. The quadratic form is an *exact* GEMM:  `Q = Φ θᵀ`

The gaussian weight depends on the Mahalanobis distance of a pixel from the gaussian's 2D center:

```
Q = dᵀ Σ⁻¹ d,    d = p − μ = (pₓ − μₓ, p_y − μ_y),    Σ⁻¹ = conic = [[a, b], [b, c]]
```

Expand it in the **pixel coordinates** `pₓ, p_y`:

```
Q = a(pₓ−μₓ)² + 2b(pₓ−μₓ)(p_y−μ_y) + c(p_y−μ_y)²
  = a·pₓ²  + c·p_y²  + 2b·pₓ·p_y                        ← quadratic in pixel
    + (−2aμₓ − 2bμ_y)·pₓ  + (−2bμₓ − 2cμ_y)·p_y          ← linear in pixel
    + (aμₓ² + 2bμₓμ_y + cμ_y²)                           ← constant
```

Every term is (a function of the pixel) × (a function of the gaussian). Collect the **pixel** parts
into a 6-vector `Φ` and the **gaussian** parts into a 6-vector `θ`:

```
Φ(p) = [ pₓ²,  p_y²,  pₓ·p_y,  pₓ,  p_y,  1 ]                       (shared across ALL gaussians)

θ(g) = [ a,  c,  2b,  −2aμₓ−2bμ_y,  −2bμₓ−2cμ_y,  aμₓ²+2bμₓμ_y+cμ_y² ]   (per gaussian)

Q(p, g) = Φ(p) · θ(g)
```

Stack all pixels into `Φ[P×6]` and all gaussians into `θ[G×6]`:

```
        Q[P×G] = Φ[P×6] · θᵀ[6×G]
```

**This is an identity, not an approximation** — monomial expansion is exact. The pixel basis `Φ` is the
same for every gaussian (it is a property of the tile, computed once), so the entire `P×G` field of
Mahalanobis distances is a single matrix multiply with contraction dimension 6.

> Implementation: `spike/forward.py::quad_form` (global) and `quad_form_tilelocal` (the bf16-safe form,
> below). The fat-shallow `K=6` contraction means the GEMM runs at ~3–6× elementwise rather than the
> full 16× — still a win; batch `P` and `G` large to raise utilization.

### 1.1 Tile-local coordinates (mandatory for bf16)

Global pixel coordinates make `pₓ²` as large as `res²` (e.g. 640k), which overflows the bf16 mantissa
and kills the kernel. Fix: use **tile-local** pixel coordinates `0…15` (so `pₓ² ≤ 256`) and fold each
tile's pixel-space origin into the gaussian mean `μ` inside `θ`. This is exact in fp64 and reduces the
bf16 error on an O(1) `Q` from ~219 (global) to ~0.04 (tile-local).

> Verified: `spike/tests/test_tile_local.py` (`quad_form_tilelocal == quad_form` to 1e-9 in fp64).

---

## 2. The polynomial splat (no `exp`)

The weight that standard 3DGS writes as `o·exp(−½Q)` becomes a **polynomial splat**:

```
w = o · (1 − Q/k)₊²            where (x)₊ = max(0, x)
```

`(1 − Q/k)` is **affine in Q**, so it folds into `θ` for free: `Φ`'s sixth monomial is the constant `1`,
so scaling `θ` by `−1/k` and adding `1` to the constant term turns the matmul output directly into
`1 − Q/k`. The only non-matrix work left in the weight is then:

```
relu(·)      one SFPU op   (the max(0,·))
square(·)    one ELWMUL    (the ²)
```

**Transcendental count = 0.** On Blackhole these two fuse into a single `unary_chain[relu, square]` SFPU
pass. This is the first of the two non-matrix residuals.

> Implementation: `spike/arms.py::blend_A`; device fusion in `tools/m4_fuse_forward.py`.

---

## 3. The blend is a GEMM:  Weighted Sum Rendering

Standard 3DGS sorts gaussians by depth and alpha-composites front-to-back — inherently sequential and
branch-heavy. Matrix-native 3DGS uses **Weighted Sum Rendering** (WSR), an order-independent blend:

```
C = (Σ_g w_g c_g) / (Σ_g w_g)
```

Both the numerator and denominator are contractions over `g`, i.e. matrix multiplies:

```
num[P×3] = w[P×G] · c[G×3]          (color, weighted)
den[P×1] = w[P×G] · o[G×1]          (the opacity is folded so den is also a GEMM)
C[P×3]   = num / den                 ← one pointwise divide
```

No sort, no front-to-back recurrence — the blend is two GEMMs and a divide.

> **The known limitation:** because `w` is depth-free, `C` is a screen-space *average*; it cannot model
> occlusion (a near solid surface hiding a far one). This is the lego gap in the README, and the WIP
> "occlusion without a sort" work buys it back with order-independent terms that multiply only `o`.
> Implementation: `spike/arms.py`.

### Forward, in one line

```
  Φθ (GEMM, K=6)  →  1−Q/k (folded)  →  relu·square (1 SFPU pass)  →  WSR num/den (2 GEMMs)  →  ÷
  └──────────── matrix engine ───────────┘   └─ residual #1 ─┘     └──── matrix engine ────┘
```

This is exactly the shape of a tiny two-layer network: **GEMM → pointwise → GEMM**.

---

## 4. The backward pass — transposed GEMMs, for free

Reverse-mode autodiff of `y = W·x` gives the vector-Jacobian product `∂L/∂x = Wᵀ·(∂L/∂y)`. So each
forward GEMM becomes a transposed GEMM in the backward pass, in reverse order, separated by the
derivatives of the same two pointwise ops. With the incoming gradient `gC = ∂L/∂C`:

```
gC ──(divide bwd, pointwise)──▶  gnum = gC / den,   gden = −Σ(gC·C)/den
gnum, gden ──(transposed GEMMs)──▶
        gc  = wᵀ · gnum            [G×3]   (color grad)
        go  = wᵀ · gden            [G×1]   (opacity grad)
        gw  = gnum · cᵀ + gden · oᵀ   [P×G]   (weight grad)
gw ──(poly bwd, pointwise)──▶     gQ = gw · ∂w/∂Q
gQ ──(transposed GEMM)──▶         gθ = Φᵀ · gQ        [6×G]   (Φ is constant ⇒ no gΦ)
```

Three forward GEMMs (`Φθ`, `w·c`, `w·o`) ↔ three transposed GEMMs in the backward, with the same two
pointwise residuals in between. Fully symmetric — the matrix engine dominates both directions.

> Verified end-to-end against finite-difference gradcheck **and** autograd to 1e-9
> (`spike/backward.py`, `spike/tests/test_backward.py`).

### 4.1 The only two genuinely new derivatives

Everything above except two pieces is standard calculus; those two are the only hand-derived terms:

```
poly-splat:   ∂w/∂Q  = −2 o (1 − Q/k)₊ / k

WSR quotient: ∂C/∂w_g = (c_g − C) / den
```

Both are simple, pointwise, and verified. (The "contribution-preserving relocation" that MCMC needs is
also trivial here: because `w` is *linear* in `o`, splitting a gaussian is the exact `o → o/n` split —
no nonlinear `1−(1−α)` bookkeeping.)

---

## 5. The geometry stage and its Jacobian (reused, and cheap)

`θ` is built from the raw trainable parameters (mean3d, scale, quaternion, color, opacity) by the
standard 3DGS geometry: EWA projection to 2D, the 2D covariance/conic, SH color decode. This is
**per-gaussian, O(G)** — not the `P×G` hot loop — so it is cheap and can live wherever is convenient
(host or a light device pass).

Its backward is the standard 3DGS geometry **Jacobian**, ported formula-for-formula from
gsplat / diff-gaussian-rasterization (these are renderer-agnostic — they describe the *shared* geometry,
not the blend):

- conic inverse: `∂(M⁻¹) = −M⁻¹ (∂M) M⁻¹`
- EWA projection: `Wᵀ Jᵀ (·) J W`
- quaternion → rotation: `∂R/∂q`
- covariance decomposition `Σ = R S Sᵀ Rᵀ`

These are also matrix expressions, and being O(G) they never dominate. So the full gradient flow is:

```
gC →(WSR bwd: transposed GEMMs)→ gw →(poly bwd)→ gQ →(Φᵀ GEMM)→ gθ →(geometry Jacobian, O(G))→ g(params) → Adam
```

> Device geometry fwd/bwd verified on silicon: `tools/m5_geom_device.py`, `tools/geom_bwd.py`
> (the Jacobian, rel ~5e-7).

---

## 6. Binning — the one residual that is *not* a matmul (and need not be)

To avoid materializing the dense `[P,G]` (which is the whole O(P·G) cost), we **bin**: split screen
space into a tile grid, scatter each gaussian to one cell (single destination), and let each tile gather
a fixed budget `K` of gaussians (overflow dropped). The render then runs, per tile, a small
`[256 × K]` GEMM instead of `[P × G]` — the same math at ~32× less work, L1-resident.

Binning decomposes into parts of different character:

| part | nature | matrix form? |
|---|---|---|
| tile assignment `⌊μ/16⌋` | integer / SFPU, O(G) | no (but cheap) |
| **applying** the bins (gather θ / scatter-add grad) | selection / segmented reduce | **yes** — `embedding` is a 0/1 selection matmul; its transpose is the grad scatter |
| compaction / prefix-sum | scan | yes — a triangular-ones matmul |
| top-K-by-distance selection | sort | **no** — an integer sort |

So the *application* of binning is already matrix-form (this is why on-device gather is `ttnn.embedding`
and the gradient scatter-add is its transpose — an `inv[G,Smax]` gather + small reduce). The
*construction* keeps an irreducible integer-scatter core. That core is the **second** non-matrix residual
(single-destination bucketing on CPU) — and forcing it into a dense `[T,G]` matmul would
re-introduce the O(T·G) cost binning exists to remove, so we deliberately don't.

> Measured lossless at (R=1, K=128) vs the dense render (`tools/m2v_binning_quality.py`). On-device as a
> ttnn-op graph; the dense O(T·G) construction is the at-scale
> bottleneck the custom NoC-atomic kernel targets (README → WIP).
>
> Binning is implemented **both** on the host (O(G) CPU scatter, but a per-iter sync) **and** on-device
> (sync-free, but the dense O(T·G) ttnn graph above). Because that dense construction is compute-bound
> and does not ride the matrix engine, **host binning is currently faster at scale** (res800/G10k: 201
> vs 287 ms/it) — the on-device path wins only once the dense O(T·G) is replaced by the sparse
> NoC-atomic scatter. See the README "host binning vs. on-device binning" note.

---

## 7. Precision and the pipeline split

- **Precision.** bf16/fp16 for the GEMMs and SH; the SFPU evaluates fp32 internally with bf16 I/O. The
  Adam second moment `v` must stay fp32 (it drifts in bf16); `m` can be bf16. bf16 *un-binned* all-G
  training degrades (far gaussians → large `μ_local` → large `θ` → bf16 loss), which is another reason
  binning (tile-local coords) is mandatory at bf16.
- **Split.** Host CPU = light, branchy glue (bin-index control, MCMC relocation indices, COLMAP for real
  scenes). Blackhole = all FLOP-heavy + matrix work. The scene (params + Adam state + intermediates)
  stays **resident on-card**; per-iteration host↔device traffic must be O(small control data), never the
  bulk scene (PCIe ≈ 46 GB/s vs on-card 512 GB/s).

---

## 8. Summary

| stage | forward | backward | engine |
|---|---|---|---|
| quadratic form | `Q = Φ·θᵀ` | `gθ = Φᵀ·gQ` | **matrix** |
| splat weight | `relu(1−Q/k)²` | `∂w/∂Q = −2o(1−Q/k)₊/k` | SFPU residual #1 |
| blend (WSR) | `num=w·c`, `den=w·o`, `C=num/den` | `gc=wᵀgnum`, `go=wᵀgden`, `gw=gnum cᵀ+gden oᵀ` | **matrix** |
| geometry | EWA project → θ (O(G)) | geometry Jacobian (O(G)) | matrix, cheap |
| binning | spatial-hash scatter, fixed K | gather-transpose (`embedding`ᵀ) | residual #2 (scatter) |

The heavy `P×G` work — forward and backward — is matrix multiplies. Two small residuals remain: a
`relu+square` (SFPU) and a fixed-capacity bucket scatter (integer). That is the target shape, reached
and verified against the matrix-native reference at every step.
