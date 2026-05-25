# SplatThis

**Photo → SVG / PPTX / HTML via 2D Gaussian splatting. Apple-Silicon native.**

Takes a PNG, fits a few thousand 2D Gaussian splats to it, and emits one of:

- **SVG** — editable, scalable, browser-friendly
- **PPTX** — drop-in PowerPoint slide with native DrawingML shapes
- **Canvas HTML** — JS runtime that does linear-light alpha-over (near-photorealistic)

Most other Gaussian-splatting projects optimize for 3D scenes or training-throughput on
CUDA GPUs. This one optimizes for **deployable vector documents**: SVG you can put
in a webpage, PPTX you can paste into a deck.

## What it's good at

- Photo backgrounds in slide decks that don't look mosaic-like.
- Painterly / abstracted photo representations in editable vector form.
- HTML5 canvas renders that match the optimizer's linear-light math exactly.
- Running on Apple Silicon — MLX optimizer is ~5× faster than torch on M-series.

## What it's not for

- Photorealistic SVG at arbitrary scale — there's a perceptual ceiling around
  LPIPS 0.30 on photo content (see `docs/SVG_PPTX_GAUSSIAN_TRICKS.md`).
- Crisp text or icons — use real vectors, not splats.
- Real-time 3D scenes — use [gsplat](https://github.com/nerfstudio-project/gsplat).
- Highest training throughput — CUDA-native splatters (gsplat, original 3DGS) are faster.

## Quick start

```bash
git clone https://github.com/BramAlkema/SplatThis.git
cd SplatThis
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Canvas (HTML, near-photorealistic, large file)
splatlify input.png -o out.html --format canvas \
  --splats 4000 --stages 500,250,125

# SVG (small file, editable, perceptual cap ~LPIPS 0.30)
splatlify input.png -o out.svg --format svg --training-export-target svg \
  --splats 2000 --stages 1000,500,250

# PPTX (PowerPoint native shapes)
splatlify input.png -o out.pptx --format pptx --pptx-splat-style blur \
  --splats 2000 --blur-postfit-iters 120
```

## Recommended recipe per deploy target

The dominant lever is **training time, not splat count**. Both formats hit a
perceptual ceiling, but with format-specific training you get there cleanly.

| Target | Splats | Stages | Notes |
|---|---|---|---|
| Canvas (HTML) | 2 k – 4 k | `--stages 500,250,125` | Best perceptual quality (LPIPS ≈ 0.10). |
| SVG | 500 – 2 k | `--stages 1000,500,250` | Fewer splats with more iters wins — fewer gradient-stop fan artifacts. |
| PPTX | 2 k – 4 k | `--stages 500,250,125 --blur-postfit-iters 120` | Use `--pptx-splat-style blur` and the blur-aware postfit. |

For both SVG and PPTX, pass `--training-export-target <format>` so the optimizer
trains against the deploy compositor, not a generic loss. ~17% perceptual lift on
SVG vs naive train-once-emit-anywhere.

## Three things this project actually does that others don't

1. **PPTX export from splats, calibrated against real PowerPoint.** DrawingML's
   `<a:blur>` calibration constant (σ = rad / 3.25) was measured via erf-fit
   edge-response in real PowerPoint. See `docs/SVG_PPTX_GAUSSIAN_TRICKS.md` and
   the writeup in [svg2ooxml's research notes](../svg2ooxml/docs/reference/research/blur-fidelity-results.md).

2. **Format-aware training pipeline.** `--training-export-target {canvas, svg,
   pptx-softedge}` — the trainer optimizes against the deploy format's compositor
   during training, not as a postfit afterthought. Closes ~0.07 LPIPS on SVG.

3. **Documented perceptual ceilings.** SVG caps around LPIPS 0.30, PPTX around
   0.40, canvas keeps improving with splat count. The catalog
   (`docs/SVG_PPTX_GAUSSIAN_TRICKS.md`) shows the LPIPS-vs-SSIM table that
   surfaced these ceilings — SSIM systematically over-ranks the blur recipe;
   LPIPS reverses several findings.

## Honest quality numbers

Measured against the chameleon test image (`input.png`, 476×502) at 2 k splats
with `--stages 1000,500,250` and `--training-export-target` matching the format:

| Format | SSIM | LPIPS↓ | PSNR | File size |
|---|---|---|---|---|
| Canvas (HTML JS runtime) | 0.92 | **0.09** | 30 dB | 700 KB |
| SVG (standard recipe) | 0.76 | **0.32** | 22 dB | 1.0 MB |
| PPTX (blur recipe) | 0.55 | ~0.40 | 14 dB | 100 KB |

LPIPS: ~0.10 excellent, ~0.20 acceptable, ~0.30 visibly different, ~0.50 clearly
different. SSIM in isolation mis-ranks splat-rendered output — always cross-check
with LPIPS or visual judgment.

## How it works

1. **Content-adaptive initialization** — 2D Gaussians seeded where image
   gradients say there's detail.
2. **Differentiable optimization** (MLX on Apple Silicon, torch elsewhere) —
   refines position, anisotropic covariance, color, alpha against L1+SSIM loss.
3. **Progressive densification & pruning** — staged loop adds splats in
   high-error regions and prunes low-impact ones, up to the splat budget.
4. **Format-specific postfit** — color/alpha refinement against the deploy
   format's compositor proxy. `--svg-proxy-postfit-iters`,
   `--pptx-proxy-postfit-iters`, `--blur-postfit-iters`.
5. **Emit** — SVG with quantized-sigma `<feGaussianBlur>` filters, PPTX with
   calibrated `<a:blur>`, or HTML with a JS canvas runtime doing real
   linear-light alpha-over.

## Architecture

```
src/png2svg_gs/
├── cli.py         # `splatlify` entry point
├── converter.py   # orchestration, training stages, postfit dispatch
├── renderer.py    # differentiable renderer + L1+SSIM loss
├── splat.py       # GaussianSplat model + raw schema
├── io.py          # SVG / PPTX / Canvas emit + quality metrics
├── mlx_stage.py   # MLX-native optimizer (Apple Silicon)
├── mlx_renderer.py
├── mlx_losses.py
└── ...
```

## Development

```bash
PYTHONPATH=. pytest tests/unit/ --cov=src --cov-report=term-missing --tb=short

# Lint
black src/ tests/
flake8 src/ tests/
mypy src/
```

Requires Python ≥ 3.10. On Apple Silicon, MLX 0.31+ is the default optimizer
backend; pass `--optimizer-backend torch` for CUDA/CPU runs.

## Useful flags

| Flag | What it does |
|---|---|
| `--format {svg,pptx,canvas}` | Output format. |
| `--training-export-target {auto,canvas,svg,pptx-softedge}` | Loss-target compositor. |
| `--pptx-splat-style {gradient,soft-edge,blur}` | DrawingML primitive (`blur` recommended). |
| `--svg-recipe {standard,blur,palette-quantized,…}` | SVG emit recipe (`standard` is best perceptual). |
| `--splats N` | Splat budget. |
| `--stages a,b,c` | Per-stage iteration counts. |
| `--blur-postfit-iters N` | Color/alpha refinement against Gaussian-conv proxy. |
| `--optimizer-backend {mlx,torch}` | MLX is default on Apple Silicon. |
| `--max-edge N` | Downscale long edge to N px. |
| `--artifacts-dir DIR` | Save per-stage splat snapshots + run manifest. |

## Related work

- **[gsplat](https://github.com/nerfstudio-project/gsplat)** — CUDA-native
  Gaussian rasterizer; vendored under `external/gsplat/`. Faster training, no
  vector export.
- **[Image-GS](https://github.com/<...>)** — academic 2D splat work; reference
  under `external/image-gs/`.
- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)**
  — Kerbl et al. 2023, the reference 3DGS implementation.
- **[svg2ooxml](https://github.com/BramAlkema/svg2ooxml)** — sibling project for
  SVG → OOXML conversion; the `<a:blur>` calibration research lives there.

## License

MIT.
