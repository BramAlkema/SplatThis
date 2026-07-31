# CSS Linear Compositor MVP

**Status:** Chameleon MVP completed; not integrated into the production CSS exporter
**Date:** 2026-07-31

## Question

Can scriptless CSS approximate the mathematical pixel runtime materially better
than the current SVG-like CSS compositor?

## Setup

- Source: `result/corpus/images/chameleon.png`, 364 x 384.
- Population: the same 1,615 splats from
  `result/corpus/runs/chameleon_svg_s0_art/final.raw.json`.
- Browser: Chrome 140.0.7339.81, DPR 1.
- No retraining, bitmap embedding, SVG, Canvas, or JavaScript at runtime.
- Every candidate was captured at native size after a warm-up navigation.

The first sweep independently tested linear-gradient interpolation,
`color(srgb-linear ...)`, an alpha-gradient mask over a solid fill, nine exact
Gaussian mask stops, and reversed DOM draw order. A bounded second sweep varied
the Gaussian footprint, stop curve, and global alpha scale only on the best
operator family.

## Result

| Artifact | SSIM_sRGB | LPIPS | PSNR_sRGB | MS-SSIM luma | HTML size |
|---|---:|---:|---:|---:|---:|
| Current-style CSS baseline | 0.701735 | 0.318214 | 20.361 | 0.717498 | 485 KB |
| Standard browser SVG | 0.707568 | 0.317296 | 20.441 | 0.722552 | 906 KB |
| Reversed DOM order plus linear color | 0.843273 | 0.168291 | 27.200 | 0.915713 | 674 KB |
| Reversed order plus linear fill/mask | 0.856337 | 0.164309 | 27.148 | 0.919597 | 618 KB |
| Reversed linear fill/mask, nine exact stops | **0.874831** | **0.145616** | **28.090** | **0.939233** | 834 KB |
| Exact pixel runtime | 0.904468 | 0.144848 | 29.776 | 0.958307 | 290 KB |

The winning CSS candidate improves SSIM_sRGB by 0.17310 and LPIPS by 0.17260
without changing the splat population. Its LPIPS is only 0.00077 above the
pixel runtime, although local structure and color metrics still favor the
pixel runtime.

Seven warmed browser navigations gave a median two-animation-frame settle time
of 83.0 ms for the experimental baseline and 176.4 ms for the winner. These are
load/style/layout/paint settle measurements, not isolated compositor timings.

## What mattered

Draw order dominated the result. The mathematical renderer consumes the stored
population front-to-back, while CSS paints later DOM elements over earlier
ones. Emitting the same ordered population directly therefore reversed the
intended transmittance relationship. Reversing the DOM elements recovered
0.1415 SSIM before any mask refinement.

Linear-gradient interpolation alone was effectively neutral: 0.701735 to
0.701741 SSIM. Separating color and opacity into a solid linear-sRGB fill plus
an alpha mask helped after order correction. Replacing the adaptive two-to-eight
stop mask with nine exact Gaussian samples added a further 0.01849 SSIM. The
original 2.875-sigma footprint beat the bounded alternatives for that recipe;
global alpha scaling did not help.

## Verdict

Pure CSS can come much closer than the current exporter. The strongest next
slice is an opt-in `reverse-linear-mask` CSS recipe, followed by a full-corpus
Chrome gate. It should not replace the compact default yet: on Chameleon the
winner is about 72% larger than the like-for-like MVP baseline and takes about
2.1 times as long to settle.

The order finding should also be tested independently in SVG. It is a shared
export-order question, but this MVP does not change SVG or production CSS.

Detailed local artifacts are under `./tmp/css-linear-mvp/chameleon/`.

## Reproduce

```bash
./venv/bin/python tools/css_linear_compositor_mvp.py \
  result/corpus/images/chameleon.png \
  --splats-json result/corpus/runs/chameleon_svg_s0_art/final.raw.json \
  --manifest result/corpus/runs/chameleon_svg_s0_art/run_manifest.json \
  --output-dir ./tmp/css-linear-mvp/repro \
  --repeats 7
```

The runner writes live HTML, native-size PNG captures, a metric report, and a
local comparison overview.
