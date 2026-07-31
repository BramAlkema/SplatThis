# Provenance & historical benchmarks

Salvaged from fourteen retired design documents (July 2026). Everything here is
either **unrecoverable from the code** — upstream commit SHAs, algorithm
attribution, measurements of configurations that no longer exist — or the only
written record of a convention the code assumes but never states.

Nothing in this file is current guidance. For how the pipeline works today see
[`adr-002-png2splat-python-pipeline.md`](adr-002-png2splat-python-pipeline.md)
(architecture contract), [`adr-003-fidelity-roadmap.md`](adr-003-fidelity-roadmap.md)
(active roadmap) and [`SVG_PPTX_GAUSSIAN_TRICKS.md`](SVG_PPTX_GAUSSIAN_TRICKS.md)
(format catalog).

---

## 1. Upstream provenance

`external/` and `ml-sharp/` are gitignored local clones, so this table is the
only committed record of which upstream commits the pipeline was derived from.
All eight SHAs were re-verified against the local checkouts on 2026-07-28.

| Repo | Local path | Commit | Focus |
|---|---|---:|---|
| GaussianImage | `external/GaussianImage` | `d53393b` | 2D Gaussian fitting + quantization |
| Instant-GI | `external/Instant-GI` | `91bba42` | Learned gaussian initialization + short fine-tuning |
| image-gs | `external/image-gs` | `0308836` | Error-guided progressive optimization for 2D Gaussians |
| gsplat | `external/gsplat` | `6f37836` | CUDA rasterization + split/prune strategies |
| 2d-gaussian-splatting | `external/2d-gaussian-splatting` | `335ad61` | Densify/clone/split/prune training loop |
| diffvg | `external/diffvg` | `85802a7` | Differentiable vector graphics baseline |
| ml-sharp | `ml-sharp` | `1eaa046` | Feed-forward gaussian prediction architecture patterns |
| svg2ooxml | `../svg2ooxml` | `4a16e34` | DrawingML/PPTX builder architecture reference |

### Restoring the reference clones

The clones were removed from disk on 2026-07-28 (577 MB, never committed, no
local modifications and no unpushed commits at deletion — every HEAD matched
the SHAs above). Nothing in `src/` imports them. The former optional gsplat
adapter depended on private legacy 2D APIs, was not a packaged dependency, and
was removed; `auto` now resolves deterministically to the Torch reference
renderer.

To restore the exact state they were studied at:

```bash
mkdir -p external && cd external
while read -r name url sha; do
  git clone "$url" "$name" && git -C "$name" checkout "$sha"
done <<'REPOS'
GaussianImage         https://github.com/Xinjie-Q/GaussianImage.git          d53393b
Instant-GI            https://github.com/whoiszzj/Instant-GI.git             91bba42
image-gs              https://github.com/nyu-icl/image-gs.git                0308836
gsplat                https://github.com/nerfstudio-project/gsplat.git       6f37836
2d-gaussian-splatting https://github.com/hbb1/2d-gaussian-splatting.git      335ad61
diffvg                https://github.com/BachiLi/diffvg.git                  85802a7
REPOS
cd .. && git clone https://github.com/apple/ml-sharp.git ml-sharp \
  && git -C ml-sharp checkout 1eaa046
```

### Which algorithm came from where

The live source carries no attribution comments; this mapping exists nowhere
else.

| Live behaviour | Borrowed from |
|---|---|
| Alpha compositing (not weighted-average blending) | `image-gs/model.py:356-364` |
| Mixed content-adaptive + random coverage init | `image-gs/model.py:220-246` |
| Error-guided densification, residual-informed colour init | `image-gs/model.py:547-570` |
| Plateau LR decay + early stop | `image-gs/model.py:528-545` |
| Rotation+scale covariance parameterization | `GaussianImage/utils.py:96-121` |
| Scale-regularization loss (**deliberately deferred, still unbuilt**) | `Instant-GI/train_init_net.py:161` |

### Image-GS paper reference numbers

Zhang et al., *Image-GS: Content-Adaptive Image Representation via 2D
Gaussians*, SIGGRAPH 2025 — <https://github.com/NYU-ICL/image-gs>.

- `λ_init = 0.3` balancing content-adaptive vs uniform sampling — the
  provenance of the live `init_gradient_weight` default (`converter.py`).
- `K = 10` top-gaussians per pixel (top-K normalization; **not** adopted — the
  live renderer composites `weighted` / `alpha-over` instead).
- Per-parameter learning rates μ:5e-4, c:5e-3, s:2e-3, θ:2e-3; `loss = L1 + 0.1·SSIM`.
- 5K-step convergence, 95% of final quality by step 400.
- 16×16 tiles — the provenance of `MlxBatchedGaussianRenderer` `tile_size=16`.
- Published rate-distortion: 30.41 dB PSNR at 160 KB (JPEG 25.43 dB); at
  0.366 bpp, PSNR 32.99 ± 4.49 dB, MS-SSIM 0.966 ± 0.020, LPIPS 0.083 ± 0.057.

---

## 2. Why `--device mps` was abandoned

Measured May 2026 on an M4 MacBook Air, 400 px / 2000-splat chameleon, identical
command except `--device`:

| Device | Total | Optimize | Final splats | Internal SSIM | Proxy SSIM |
|---|---:|---:|---:|---:|---:|
| CPU | 59.47 s | 57.56 s | 1928 | 0.7540 | 0.7534 |
| MPS | 269.20 s | 266.40 s | 1924 | 0.7597 | 0.7585 |

The tiled torch renderer is **~4.5× slower on MPS than on CPU** — correct, but
not worth it. This is why `--device` advertises only `cpu`/`cuda` and why the
torch-MPS path should not be re-attempted without new evidence.

The smoke test that redirected the effort to MLX instead:

| Backend | Smoke result |
|---|---|
| MLX GPU | 5 matmuls in 0.0376 s |
| MLX CPU | 5 matmuls in 0.3279 s |

---

## 3. MLX vs torch (May 2026)

Chameleon 1000 px, 2000 splats, `balanced`, stages 100/50/25.

| Pipeline | Wall clock | Final splats | Internal SSIM | Export SSIM (sRGB) | Pixel mean err |
|---|---:|---:|---:|---:|---:|
| torch + linear training | 172.0 s | 1551 | 0.7338 | 0.6831 | 14.61 |
| torch + sRGB training | 172.0 s | 1551 | **0.7529** | **0.7187** | **10.08** |
| MLX + sRGB training | **17.4 s** | 1551 | 0.7375 | 0.7022 | 10.93 |

Two findings: sRGB-space training roughly halved the train→deploy gap on both
backends, and MLX gave a 9.2× wall-clock speedup at ~95% of torch's quality.

> **Caveat — do not quote the 0.702-vs-0.719 gap as current.** These runs
> predate the MLX loss defaulting to `oklab-l1-ssim`; at the time MLX used
> `linear-l1`. Closing that gap was exactly the follow-up, and it landed.

Renderer-only medians, 1923 splats at 400 px:

| Backend | Device | Mode | Tile | Batch | Median |
|---|---|---|---:|---:|---:|
| `mlx-batched` | mlx-default | forward | 16 | 16 | 93.8 ms |
| `mlx-batched` | mlx-default | forward | 16 | 32 | 94.6 ms |
| `torch-batched` | mps | forward | 16 | 16 | 178.2 ms |
| `torch-batched` | mps | forward | 16 | 32 | 198.7 ms |
| `mlx-batched` | mlx-default | forward+backward | 16 | 16 | 300.8 ms |

**MLX↔torch renderer parity was measured at 1.19e-7 max absolute difference** in
both linear and sRGB modes. The unit tests only assert `atol=1e-5`, so that
figure is the tighter empirical bound and worth preserving.

---

## 4. Layer model semantics

`splat.py` defines `LAYER_BASE/MASS/DETAIL/EDGE = 0/1/2/3` with bare names. What
each layer is *for* was only ever written down in the retired layered-saliency
spec:

| Layer | Purpose | Placement |
|---|---|---|
| **L0 base** | Cover the entire canvas; low-frequency colour fields, shadows, background gradients. Exists specifically to **prevent the transparent/black star-field fallback** that early builds suffered. | Stratified full-canvas, very broad sigma from native pixel area |
| **L1 mass** | Large foreground masses and silhouettes; medium-scale colour fields | Foreground/saliency mask sampling, medium isotropic-to-mild-anisotropic |
| **L2 detail** | Salient detail — faces, eyes, markings | Saliency-weighted, smaller splats |
| **L3 edge** | Edges and speculars | Edge-band sampling, anisotropic along the edge |

Ordering is ascending (L0 back, L3 front) and is folded into the compositing key
as `layer + local_importance`. The live budget split comes from
`base_layer_fraction` in `budgets.py` (per time budget), **not** from the fixed
table the retired spec proposed. The layers additionally collapse into three
parallax planes in `generate_native_canvas_html` and the historical
`generate_parallax_pixel_runtime_html` — a use the spec never had.

---

## 5. Deleted documents

Removed 2026-07-28 as stale. All recoverable from git history if ever needed.

**Retired `splat_this` pipeline** (dead APIs, dead CLI flags): `API.md`,
`EXAMPLES.md`, `TROUBLESHOOTING.md`, `ADAPTIVE_SPLATS_SPEC.md`,
`ADAPTIVE_IMPLEMENTATION_PLAN.md`, `adr-001-image-gs-insights.md`,
`critical-analysis-splatthis-vs-image-gs.md`.

**Superseded plans for the live pipeline** (shipped, then overtaken):
`MLX_RENDERER_OPTIMIZER_SPEC.md`, `MLX_IMPLEMENTATION_STATUS.md`,
`APPLE_GPU_RENDERER_SPEC.md`, `LAYERED_SALIENCY_PASSES_SPEC.md`,
`adr-000-image-gs.md`, `png2splat_cpu_algorithm_adaptation.md`,
`png2splat_repo_research.md`.

Also removed: `.agent-os/` (60+ files from a retired workflow tool).
