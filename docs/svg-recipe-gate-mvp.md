# Full-Corpus Browser SVG Recipe Gate MVP

**Status:** complete; browser selector cleared for default-off integration
**Date:** 2026-07-31

## Question

Can SplatThis safely improve existing browser SVG splat populations by
selecting a different native export recipe per image, without retraining or
changing the shape population?

The experiment compares the current `standard` recipe with
`palette-quantized` and native `blur`. It deliberately excludes post-fitting,
residual paths, and population changes so recipe choice is the only variable.

## Renderer

The governing target is Chromium, not librsvg. `tools/capture_svg.py` opens the
emitted SVG directly in Playwright Chromium with an exact `width × height`
viewport, device scale factor 1, no page crop, disabled screenshot animations,
and explicit font and render-frame waits. Each artifact receives one
unmeasured warm-up capture followed by three measured captures. All 63
artifacts produced identical measured PNG hashes.

This replaces the sibling `svg2pptx` screenshot helper's hard-coded full-page
viewport and blind one-second wait. It also keeps the renderer open across the
corpus, so browser startup is not charged to every recipe.

## Method

The seed-0 requested-2k SVG `final.raw.json` for each of the 21 corpus images
was exported through all three recipes. The 63 emitted SVGs were captured in
Chrome 140.0.7339.81 at native dimensions. Full frames were scored; no crop
stands in for an image.

Each image's standard render fixes eight worst-error ROIs. Candidate recipes
reuse those ROIs and must pass whole-frame SSIM, MS-SSIM, LPIPS, OKLab p95,
edge, worst-ROI, file-size, and capture-time guards. A candidate also needs a
meaningful gain and a positive balanced score. Otherwise selection atomically
falls back to standard.

The corpus go/no-go was declared before the run:

- at least 5 of 21 images must accept a non-standard recipe;
- accepted images must have a meaningful median LPIPS or OKLab-p95 gain;
- selected corpus median size growth must stay at or below 10%; and
- selected corpus median capture-time growth must stay at or below 20%.

## Result

All 21 images completed with zero capture or scoring failures. Seven accepted
`palette-quantized`; no native-blur candidate passed. Fourteen reverted to the
standard artifact.

| Image | SSIM gain | LPIPS gain | OKLab p95 gain | SVG size | Capture time |
|---|---:|---:|---:|---:|---:|
| Brick | +0.01301 | +0.01795 | +0.00256 | -69.1% | -19.9% |
| Cell | +0.02024 | +0.04032 | +0.01101 | -66.1% | -19.3% |
| Chameleon | +0.01981 | +0.00514 | +0.01222 | -70.1% | -16.9% |
| Hubble deep field | +0.00497 | +0.01527 | +0.00800 | -65.7% | -11.6% |
| Immunohistochemistry | +0.00393 | +0.00290 | +0.02934 | -70.9% | -10.2% |
| Retina | +0.01511 | +0.01014 | +0.01612 | -70.2% | -3.4% |
| Rocket | +0.00629 | +0.00458 | +0.00732 | -67.0% | -7.6% |

Among these seven, median LPIPS gain was `0.01014` and median OKLab-p95 gain
was `0.01101`. Because fourteen images retain the baseline, corpus-median
selected size and capture-time change are both zero.

The verdict is **eligible for default-off browser selector integration**: 7
safe wins clear the predeclared minimum of 5. This is evidence to implement the
selector, not a claim that `splatthis` already invokes Playwright or selects a
recipe automatically. `palette-quantized` remains available explicitly.

The earlier librsvg run accepted only Cell, Chameleon, Retina, and Rocket. That
difference is precisely why renderer identity is part of the gate: the Chrome
result governs browser-delivered SVG, while librsvg evidence cannot be silently
transferred to it.

The complete per-image comparison, all candidate rejection reasons, actual
SVGs, and Chrome frames are in `./tmp/svg-recipe-gate-chromium/index.html` and
`./tmp/svg-recipe-gate-chromium/summary.json`. Compact release evidence is in
`data/svg-recipe-gate-mvp.json`.

## Reproduction

```bash
PYTHONPATH=src venv/bin/python tools/evaluate_svg_recipe_gate.py \
  --corpus-root result/corpus \
  --output-dir ./tmp/svg-recipe-gate-chromium \
  --browser-executable "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --render-repeats 3 \
  --minimum-accepted-images 5
```

The next isolated change is a default-off browser recipe selector with the
same accept-or-revert policy. Bounded center adjustment remains the next
independent geometry operator after that integration slice.
