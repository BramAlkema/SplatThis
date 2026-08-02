# SplatThis

[![CI](https://github.com/BramAlkema/SplatThis/actions/workflows/ci.yml/badge.svg)](https://github.com/BramAlkema/SplatThis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/splatthis)](https://pypi.org/project/splatthis/)
[![Python](https://img.shields.io/pypi/pyversions/splatthis)](https://pypi.org/project/splatthis/)
[![License](https://img.shields.io/pypi/l/splatthis)](https://github.com/BramAlkema/SplatThis/blob/main/LICENSE)

Convert a bitmap into Gaussian splats and deploy them as browser-rendered SVG,
scriptless CSS, browser-native Canvas primitives, an accelerated pixel runtime
with exact CPU fallbacks, or native editable PowerPoint shapes.

SplatThis is target-aware. It does not pretend that native Canvas, CSS, SVG,
PowerPoint, and a generated pixel framebuffer render the same primitives: each
output is named and evaluated for what it actually does.

| Output | What you get | Governing evaluation | Best fit |
|---|---|---|---|
| Pixel runtime HTML | WebGL2 evaluates the splat formula; exact Worker/main-thread CPU fallbacks | Selected Chrome canvas pixel buffer | Highest fidelity; accelerated procedural bitmap output |
| Canvas HTML | One Canvas 2D radial-gradient primitive per splat | Native-size Playwright Chromium capture | Fast browser-native splats and optional parallax |
| CSS HTML | Scriptless DOM ellipses with CSS radial gradients | Native-size Playwright Chromium capture | No-script embedding and CSS-only hover parallax |
| SVG | Real gradients, blur primitives, or compact scripted splats | Native-size Playwright Chromium capture | Browser delivery and vector editability |
| PowerPoint | Native DrawingML shapes; no embedded preview PNG | Microsoft PowerPoint slideshow capture | Editable slides |

Chromium is the governing pixel-runtime, native Canvas, SVG, and CSS target. CairoSVG,
librsvg, and the
internal NumPy renderer cannot approve a browser-native candidate or support a
deployed-fidelity claim.

## See it

One 476x502 photograph, fitted to Gaussian splats and emitted to three
compositors that share no rendering code — vector, stylesheet, and DrawingML.

| Source | SVG | Scriptless CSS | PowerPoint |
|:---:|:---:|:---:|:---:|
| <img src="https://bramalkema.github.io/SplatThis/demo/source.png" alt="Source photograph" width="210"> | <img src="https://bramalkema.github.io/SplatThis/demo/chameleon.svg" alt="Browser-rendered SVG" width="210"> | <img src="https://bramalkema.github.io/SplatThis/demo/chameleon-css.png" alt="Chromium capture of the scriptless CSS build" width="210"> | <img src="https://bramalkema.github.io/SplatThis/demo/chameleon-pptx.png" alt="Microsoft PowerPoint slideshow capture" width="210"> |
| the input | **0.8665** SSIM<br>vector · 1.5 MB | **0.8748** SSIM<br>DOM + CSS · 0.8 MB | **0.8395** SSIM<br>editable shapes · 161 KB |
| — | live vector | Chromium capture | PowerPoint capture |

The SVG is the real thing: your browser is drawing 1,615 gradient ellipses, not
displaying a picture of them. The other two are captures because they have to
be. GitHub and PyPI both sanitize README markup through an allowlist that omits
the `<style>` tag and the `style` attribute — PyPI's `readme_renderer` runs
`nh3.clean()` with no CSS sanitizer configured — so a build made entirely of
CSS radial gradients arrives as 1,615 unstyled `<div>` elements on either site.
PowerPoint, obviously, does not run in a browser at all.

Both captures are therefore lower bounds: they carry the losses of screen
capture and rescaling on top of whatever the compositor itself costs.

**[Open the scriptless CSS build →](https://bramalkema.github.io/SplatThis/demo/chameleon-css.html)**
— no script, no canvas, no SVG, no bitmap. View source on it; the capture above
cannot show that the page is 1,615 DOM elements composited from a stylesheet.

Also available: the
[editable .pptx](https://bramalkema.github.io/SplatThis/demo/chameleon.pptx),
the historical self-contained
[pixel-runtime HTML](https://bramalkema.github.io/SplatThis/demo/canvas.html),
and a larger [corpus overview](https://bramalkema.github.io/SplatThis/).

<sub>Scores are SSIM_sRGB against the source at 364x384, measured on the
deployed artifact in its governing renderer — Chromium for SVG and CSS, a real
PowerPoint slideshow capture for the deck. SVG and CSS share one 1,615-splat
population, so their difference is purely compositor; the deck is a separate
1,674-splat run and is not a like-for-like comparison against them. The SVG and
CSS candidates were selected by re-scoring every surviving artifact rather than
by trusting recorded metrics — rerun with
`python tools/refresh_showcase.py --dry-run`.</sub>

## Install

SplatThis requires Python 3.13 or newer and an installed Google Chrome for
governing pixel-runtime/Canvas/CSS/SVG capture.

```bash
git clone https://github.com/BramAlkema/SplatThis.git
cd SplatThis
python3.13 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[capture]"
```

The `capture` extra installs the Playwright client and uses the installed
Chrome. It does not depend on the sibling `svg2pptx` repository.

On Apple Silicon, add MLX:

```bash
pip install -e ".[capture,mlx]"
```

On CPU or CUDA machines, select Torch explicitly:

```bash
splatthis input.png --optimizer-backend torch
```

If Chrome is unavailable, pixel-runtime, native Canvas, SVG, and CSS exports
can still be written, but their acceptance fails closed. Any internal proxy
metrics are marked as diagnostics rather than deployed-artifact evidence.

## Quick start

Create each supported output from the same image:

```bash
# Highest-fidelity runtime; accelerated splat equations with exact CPU fallbacks.
splatthis input.png --format pixel-runtime -o output-pixels.html

# Browser-native Canvas 2D gradient splats.
splatthis input.png --format canvas -o output-canvas.html

# Scriptless DOM/CSS splats; no canvas, SVG, JavaScript, or embedded bitmap.
splatthis input.png --format css -o output-css.html

# Static, editable SVG evaluated in Chromium.
splatthis input.png --format svg -o output.svg

# Native DrawingML splats; gradient is the conservative default.
splatthis input.png --format pptx -o output.pptx
```

Keep a complete audit trail with `--artifacts-dir`:

```bash
splatthis input.png --format svg -o output.svg \
  --artifacts-dir ./tmp/input-svg-run
```

The directory contains the run manifest, stage checkpoints, metrics, renderer
identity, and acceptance decision.

## Choose a quality budget

The default budget is 2,000 splats. More splats only help when initialization
and training are allowed to use them.

```bash
# Practical larger pixel-runtime run.
splatthis input.png --format pixel-runtime -o output-4k.html \
  --splats 4000 --initial-splat-cap 4000

# Bound resolution and let a preset choose the schedule and detail budget.
splatthis input.png --format pixel-runtime -o output.html \
  --max-edge 384 --time-budget 10m
```

For an explicit quality target, the default-off pixel-runtime controller can
stop before later stages once an observed checkpoint reaches the desired exact
CPU-boundary score; the selected final browser backend is graded separately:

```bash
splatthis input.png --format pixel-runtime -o output.html \
  --splats 4000 --initial-splat-cap 4000 \
  --adaptive-compute --adaptive-target-ssim-srgb 0.98
```

This controller does not predict future quality or stop on a plateau. It only
acts on already-rendered checkpoints.

Static pixel-runtime HTML selects one runtime in this order: RGBA32F WebGL2,
RGBA16F WebGL2, exact Worker/OffscreenCanvas CPU, then exact main-thread CPU.
The 16F path must also pass a cheap deterministic sample against the exact
formula. The selected path and its compute/end-to-end timings are exposed in
the document metadata, and governing Chromium capture grades that actual
canvas buffer. `?splatthisPixelBackend=rgba32f|rgba16f|worker|main` is available
for diagnostics. The current 21-image Chrome gate selected 32F everywhere,
kept its worst source SSIM_sRGB change to -0.0000014, and found exact
Worker/main parity. Cross-browser GPU qualification remains open; unsupported
or rejected GPU paths fall back rather than preventing rendering.

## SVG workflows

The standard recipe is the safe static default. Other recipes are explicit:

| Recipe | Characteristics |
|---|---|
| `standard` | One standards-based radial gradient per splat; static and editable |
| `palette-quantized` | Shared color gradients; often much smaller, with possible color quantization |
| `blur` | Native SVG blur primitives; compositor-sensitive |
| `scripted-matrix` | Compact data expanded by JavaScript at load time; browser use only |
| `browser-compatible` | Conservative browser-gradient encoding |

```bash
splatthis input.png --format svg -o compact.svg \
  --svg-recipe palette-quantized

splatthis input.png --format svg -o polished.svg \
  --fidelity-stage max --artifacts-dir ./tmp/polished-svg-run

# Force the stricter adaptive stop policy without artifact search.
splatthis input.png --format svg -o high.svg \
  --svg-gradient-quality high --no-svg-compositor-gate
```

SVG elements are emitted back-to-front so their painter's order matches the
front-to-back transmittance renderer. The max-fidelity profile additionally
browser-grades legacy order, corrected standard gradients, and corrected high
gradients, then accepts or reverts the complete artifact. Its decision and
fixed ROIs are stored under `svg_compositor_gate` in the manifest. See the
[SVG compositor gate](https://github.com/BramAlkema/SplatThis/blob/main/docs/svg-compositor-gate.md).

The separate fidelity stage emits every splat-parameter candidate and captures it in Chromium. A proxy
may reject a cheap loser early, but only the browser artifact can promote a
candidate. `--svg-optimize` can additionally run `svgo` when it is available on
`PATH`.

The bounded browser recipe study accepted palette quantization on 7 of 21
corpus images, with 66–71% smaller accepted files and a median accepted LPIPS
gain of 0.01014. This does not make it a universal default; see the
[browser SVG recipe gate](https://github.com/BramAlkema/SplatThis/blob/main/docs/svg-recipe-gate-mvp.md).

## Scriptless CSS compositor

The CSS target represents every Gaussian as one absolutely positioned ellipse.
Its background is a CSS `radial-gradient` with adaptive alpha stops matching
the standard SVG Gaussian curve. The browser performs the final alpha-over
composition; SplatThis does not pre-render a pixel buffer.

```bash
# Static CSS splats.
splatthis input.png --format css -o splats.html

# Scriptless 10x10 hover-grid parallax from saliency depth layers.
splatthis input.png --format css -o parallax-css.html \
  --layered-saliency --css-parallax-strength 28
```

The parallax output uses transparent hover cells and CSS sibling selectors to
move the midground and foreground planes. It remains interactive with a strict
no-script policy. The governing quality capture measures the neutral,
non-hovered frame. This target trades runtime code for DOM size: one element
per splat is convenient and inspectable, but thousands of DOM nodes can cost
more layout and paint work than the single Canvas element.

## PowerPoint workflows

PowerPoint output contains native shapes rather than a bitmap masquerading as
a slide:

```bash
# Recommended general-purpose PowerPoint output.
splatthis input.png --format pptx -o output.pptx \
  --pptx-splat-style gradient

# Historical shape order, for reproducing pre-0.2.6 decks.
splatthis input.png --format pptx -o output-legacy.pptx \
  --pptx-painter-order legacy

# Deliberately target real PowerPoint's soft-edge compositor.
splatthis input.png --format pptx -o output-softedge.pptx \
  --pptx-splat-style soft-edge \
  --training-export-target pptx-softedge
```

PowerPoint and LibreOffice do not render every DrawingML effect identically.
The `pptx-softedge` target is calibrated for Microsoft PowerPoint and may look
washed out elsewhere. In-converter PPTX previews remain proxies; benchmark
claims use real PowerPoint slideshow captures.

A same-population, 21-image PowerPoint corpus test found that corrected
back-to-front shape order improved median SSIM by 0.02662 and median LPIPS by
0.03346, but Hubble regressed. The strict artifact policy selected corrected
order for 14 images and retained legacy for seven. Corrected back-to-front
order is the default as of 0.2.6, since it matches the renderer's
transmittance model and wins the corpus median; `--pptx-painter-order legacy`
reproduces the historical stack for the rare regressing image. The resumable external
PowerPoint runner writes the accepted candidate atomically as `selected.pptx`;
ordinary headless conversion never launches PowerPoint. See the
[PowerPoint painter-order MVP](https://github.com/BramAlkema/SplatThis/blob/main/docs/pptx-order-compositor-mvp.md).

## Layered Canvas parallax

Splat layers can be displaced by mouse position to suggest depth:

```bash
splatthis input.png --format canvas -o parallax.html \
  --layered-saliency --canvas-parallax-strength 28
```

This version draws every Gaussian through the Canvas 2D API before moving the
three resulting Canvas planes. The software-rasterized equivalent is available
explicitly with `--format pixel-runtime` and
`--pixel-runtime-parallax-strength`.

This changes presentation, not the underlying 2D reconstruction. PowerPoint
hover/grid parallax remains an MVP design rather than a released exporter
feature.

## What quality to expect

**Fidelity is predicted by content, not by format.** Across the 21-image
governing corpus, mean gradient magnitude correlates with SSIM at **r = −0.84**.
Smooth, structured content reaches 0.84–0.86; broadband texture bottoms out
around 0.35. Splitting the corpus at its median gradient:

| | median SSIM |
|---|---:|
| smooth half — `cell` 0.86, `moon` 0.84, `brick` 0.84 | **0.72** |
| textured half — `gravel` 0.41, `checkerboard` 0.36, `grass` 0.35 | **0.53** |

That gap is larger than the gap between any two output formats, and it is
knowable before you spend four minutes finding out: if your image is foliage,
gravel or fine repeating texture, a splat representation will struggle
regardless of which compositor you deploy it to.

The relationship is weaker on LPIPS (r = +0.53), so treat it as a statement
about structural agreement rather than perceptual distance. Both numbers are
full-population; a 13-image subset of the same corpus reports −0.94 and +0.86,
which is a good illustration of why this project publishes the whole set.

If you are supplying your own splats rather than fitting them — projecting 3D
Gaussians, for instance — none of the above applies to you, because it is
dominated by fitting error. Call `splatthis.expected_fidelity()` and read its
`compositor_*` fields for the emitter term on its own.

<!-- corpus-results:begin -- generated by tools/update_readme.py; edit the ledgers, not this block -->

Numbers in this block are computed from the committed ledgers by `tools/update_readme.py`; `tests/unit/test_readme_results.py` fails when they go stale. All are seed-0 measurements over the 21-image governing corpus at a maximum edge of roughly 384 px, scored on the deployed artifact in its governing renderer -- Chromium for the browser targets, Microsoft PowerPoint for the deck.

**Declarative emitters.** One fitted population per image, emitted by the current exporters and compared with the original image:

| Artifact | LPIPS ↓ median | LPIPS p90 | SSIM median | Median size |
|---|---:|---:|---:|---:|
| SVG (`standard` gradients, default) | 0.2433 | 0.4619 | 0.7404 | 765 KB |
| SVG (`--svg-gradient-quality high`) | 0.2439 | 0.4532 | 0.7483 | 1,331 KB |
| Scriptless CSS | 0.2429 | 0.4586 | 0.7517 | not measured |

The rows above measure each emitter family unconditionally. A bare default run (`max-fidelity` profile) additionally applies the SVG compositor gate, which chooses per image against the default corrected-standard emitter as incumbent. Validated on the corpus after the 0.2.6 incumbent change: the gate kept `high` gradients on 15 images and the default `standard` on 6, never legacy, for gate medians of SSIM 0.7483 / LPIPS 0.2439 -- up from 0.7111 under the July legacy incumbent.

Against the original, the declarative emitters are indistinguishable: median LPIPS 0.2433 / 0.2439 / 0.2429 for svg / svg-high / css, a spread of 0.001, and an SSIM spread of 0.011 that sits below the 0.029 seed noise floor. Emitter choice is not where deployed quality comes from; the fit is. Compositor-only figures make the emitters look far more different than they are, and must not be quoted as quality.

**Fitted and deployed, per format.** Each format trains against its own export target; this is best-effort per format, not one splat set exported several ways. The pixel-runtime rows are the historical governing ledger passes; an August 2026 re-emission of all 21 artifacts from current code reproduced the 2k medians exactly (LPIPS 0.2443 / SSIM 0.7751), so the ledger and `expected_fidelity("pixel-runtime")` agree:

| Artifact | Budget | Median final splats | SSIM ↑ | LPIPS ↓ | Median size | Median training |
|---|---:|---:|---:|---:|---:|---:|
| Pixel runtime HTML | 2k | 1,395 | 0.7751 | 0.2443 | 226 KB | 3.6 min |
| Pixel runtime HTML | effective 4k | 2,382 | 0.8406 | 0.1612 | 391 KB | 9.9 min |
| PowerPoint (back-to-front order, default) | 2k | 1,374 | 0.6279 | 0.3200 | 127 KB | not recorded |
| PowerPoint (`--pptx-painter-order legacy`) | 2k | 1,374 | 0.6019 | 0.3750 | 127 KB | not recorded |

The strict PowerPoint artifact gate, choosing per image, kept the corrected order on 14 of 21 images and legacy on 7; corrected is the default as of 0.2.6, and legacy remains available for the rare regressing image.

21 of 21 images improved from 2k to effective 4k in both SSIM and LPIPS. The effective-4k runtime rendered in a median 117 ms in Chrome. None reached 0.99 SSIM (best 0.9837).

**Emitter loss alone** (the same splats through the deployed compositor versus the internal reference render -- the number that matters only if you supply your own splats): median LPIPS 0.0310 (svg), 0.0181 (svg-high), 0.0221 (css), 0.0001 (pixel-runtime). Query it programmatically with `splatthis.expected_fidelity()`.

<details>
<summary>Per-image deployed LPIPS for the declarative emitters (lower is better)</summary>

| Image | SVG | SVG high | CSS |
|---|---:|---:|---:|
| colorwheel | 0.0155 | 0.0127 | 0.0122 |
| logo | 0.0863 | 0.0789 | 0.0823 |
| text | 0.1287 | 0.1267 | 0.1337 |
| brick | 0.1470 | 0.1401 | 0.1388 |
| chameleon | 0.1672 | 0.1491 | 0.1456 |
| checkerboard | 0.1733 | 0.1557 | 0.1614 |
| rocket | 0.2136 | 0.1896 | 0.1856 |
| coffee | 0.2349 | 0.2337 | 0.2386 |
| moon | 0.2404 | 0.2351 | 0.2114 |
| astronaut | 0.2422 | 0.2326 | 0.2351 |
| coins | 0.2433 | 0.2439 | 0.2429 |
| chelsea | 0.2459 | 0.2447 | 0.2520 |
| cell | 0.2990 | 0.2645 | 0.2568 |
| stereo_motorcycle | 0.3090 | 0.3127 | 0.3149 |
| camera | 0.3278 | 0.3198 | 0.3143 |
| retina | 0.3704 | 0.3616 | 0.3651 |
| immunohistochemistry | 0.4078 | 0.4169 | 0.4226 |
| hubble_deep_field | 0.4402 | 0.4276 | 0.4271 |
| page | 0.4619 | 0.4532 | 0.4586 |
| grass | 0.5114 | 0.5223 | 0.5222 |
| gravel | 0.5403 | 0.5523 | 0.5470 |

</details>

<!-- corpus-results:end -->

An initial same-population Chameleon check makes the distinction concrete. The
population contains 1,615 SVG-trained splats. Historical forward DOM order
scored 0.7076 SSIM, corrected standard order scored 0.8494, and corrected
adaptive high gradients scored 0.8665. Native Canvas scored 0.7072 under its
historical order. Replaying the same parameters through the mathematical pixel
runtime scored 0.9045, demonstrating the remaining vector-to-pixel-runtime
gap. The older internal preview
scored 0.8803 but was not an SVG render and must not be compared as one. Native
Canvas rendered the gradients in about 11 ms and produced 156 KB of HTML; the
CPU pixel runtime took roughly 80-102 ms and produced about 290 KB. These are
one-image MVP measurements, not corpus guarantees.

On a separate 1,788-splat, 476 x 502 Chameleon checkpoint, the selected 32F
runtime completed in roughly 16-19 ms after warm-up versus roughly 127-140 ms
for exact main-thread CPU. It differed on six pixels by one byte and preserved
source SSIM_sRGB at 0.90494. The quality-gated 16F path completed in about
20 ms; its source SSIM_sRGB was 0.90486. These remain local Chrome measurements,
not a cross-browser guarantee.

SplatThis is also not a replacement for PNG, JPEG, WebP, or AVIF compression.
If editability, animation, or the splat representation is unnecessary, a
normal bitmap will usually be smaller and more faithful.

See [historical pixel-runtime scaling](https://github.com/BramAlkema/SplatThis/blob/main/docs/canvas-scaling-mvp.md) for paired
per-image results and
[SVG/PPTX compositor findings](https://github.com/BramAlkema/SplatThis/blob/main/docs/SVG_PPTX_GAUSSIAN_TRICKS.md) for the
format-specific analysis.

## How it works

1. Content-adaptive initialization places anisotropic splats.
2. Torch or MLX optimizes position, scale, rotation, color, and alpha.
3. Densification adds detail and pruning removes low-impact splats.
4. Target-aware post-fit stages approximate the deployment compositor.
5. Monotonic gates keep only measured improvements.
6. The final SVG, CSS/Canvas/pixel-runtime HTML, or DrawingML package is written atomically.

The public `converter.py` module is a small compatibility facade over the
internal numerical engine and isolated prepare, fit, and deployment phases.
Each run starts from an immutable configuration snapshot, produces one
`SplatScene`, and delegates emission plus governing evaluation to a registered
artifact backend. See [Architecture](https://github.com/BramAlkema/SplatThis/blob/main/docs/ARCHITECTURE.md) for the module
boundaries and extension rules.

Pixel-runtime, SVG, and CSS repeat-render noise is zero in the calibrated
corpus captures (21 artifacts × 5 repeats per target). Native Canvas still
needs its own full-corpus noise calibration. The versioned target floors and
capture provenance live
in [`data/artifact-gates.json`](https://github.com/BramAlkema/SplatThis/blob/main/data/artifact-gates.json).

## Main flags

| Flag | Purpose |
|---|---|
| `--format {svg,pptx,canvas,css,pixel-runtime}` | Select the deployed container and compositor |
| `--profile {m4-fast-loop,fast,balanced,max-fidelity}` | Select the quality profile (default `max-fidelity`) |
| `--splats N` | Set the maximum splat population |
| `--time-budget PRESET` | Select a content-aware schedule and detail budget |
| `--max-edge N` | Bound the input resolution while preserving aspect ratio |
| `--optimizer-backend {mlx,torch}` | Select the optimizer implementation |
| `--training-export-target {auto,pixel-runtime,browser-gradient,svg,pptx-gradient,pptx-softedge}` | Select the training compositor (`canvas` is a legacy alias for `pixel-runtime`) |
| `--svg-recipe RECIPE` | Select the emitted SVG primitive family |
| `--svg-gradient-quality {standard,high}` | Select compact or stricter adaptive SVG gradients |
| `--svg-painter-order {back-to-front,legacy}` | Select corrected or historical SVG element order |
| `--[no-]svg-compositor-gate` | Accept or revert complete browser SVG compositor candidates |
| `--pptx-splat-style STYLE` | Select DrawingML gradient, soft-edge, or blur splats |
| `--pptx-painter-order {legacy,back-to-front}` | Emit the historical or corrected DrawingML shape stack |
| `--fidelity-stage {off,balanced,max}` | Enable accept-or-revert browser SVG polish |
| `--layered-saliency` | Export base, mass, detail, and edge layers |
| `--canvas-parallax-strength PX` | Enable native Canvas plane parallax |
| `--pixel-runtime-parallax-strength PX` | Enable ImageData-runtime plane parallax |
| `--css-parallax-strength PX` | Enable scriptless CSS hover parallax |
| `--artifacts-dir DIR` | Retain the manifest and intermediate checkpoints |

Run `splatthis --help` for the full research and backend surface.

## Project status

The supported path includes target-aware training, native Canvas/CSS/SVG,
explicit pixel-runtime and PPTX export, browser pixel-runtime/Canvas/SVG/CSS
grading, native
PowerPoint generation, provenance-complete manifests, and full-corpus
reporting. See [Reproducibility](#reproducibility) for what a fixed seed does
and does not guarantee per backend.

Top-K teacher/student distillation, mixed native primitives, automatic SVG
recipe selection, adaptive compute, and PowerPoint hover parallax remain
default-off or experimental. Their current evidence is retained under
[`docs/`](https://github.com/BramAlkema/SplatThis/tree/main/docs) rather than presented as release guarantees. The architecture
and acceptance roadmap are in [ADR-003](https://github.com/BramAlkema/SplatThis/blob/main/docs/adr-003-fidelity-roadmap.md).

## Development

```bash
pip install -e ".[dev,capture]"

isort --check-only src tests tools
black --check src tests tools
flake8 src tests tools
pytest -q
python -m build
python -m twine check dist/*
```

### Reproducibility

A fixed `--seed` reproduces the reported metrics on both backends, but only
Torch reproduces the emitted artifact byte for byte.

| `--optimizer-backend` | Reported metrics | Emitted artifact |
|---|---|---|
| `torch` | identical | byte-identical |
| `mlx` (default) | identical to nine significant figures | not byte-identical |

MLX orders float32 reductions on the Metal device nondeterministically, so
repeated single-process seeded runs differ by roughly one float32 ULP (~3e-8)
in splat parameters. That is far below any quality threshold — two seeded runs
of the same image agreed on SSIM to nine significant figures — but it is
enough to tip a rounded SVG attribute across a formatting boundary, so
artifact hashes are not stable under MLX. The differences observed so far have
been geometrically inert, such as the `rotate()` angle of an isotropic splat,
where rotation is a no-op. Select `--optimizer-backend torch` when you need
bit-identical output or a stable artifact hash.

The corpus medians quoted above are unaffected: they are reported to four
decimal places, five orders of magnitude above this noise.

Corpus runs are content-addressed and resumable. Independent conversions can
run concurrently with `python tools/corpus_benchmark.py --run --jobs 2 ...`.
This is restricted to Torch/CPU runs: concurrent seeded MLX processes share one
Metal device and compound the nondeterminism described above. MLX therefore
requires `--jobs 1`; result-file writes remain serialized for every backend.

CI launches the installed Chrome before running the suite. See
[CONTRIBUTING.md](https://github.com/BramAlkema/SplatThis/blob/main/CONTRIBUTING.md) and [CHANGELOG.md](https://github.com/BramAlkema/SplatThis/blob/main/CHANGELOG.md).

## License

[MIT](https://github.com/BramAlkema/SplatThis/blob/main/LICENSE)
