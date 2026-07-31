# Changelog

All notable changes to SplatThis are documented here.

## Unreleased

### Added

- A static pixel-runtime selection chain: RGBA32F WebGL2, quality-gated
  RGBA16F WebGL2, exact Worker/OffscreenCanvas CPU, then exact main-thread CPU.
  It records the selected backend plus compute/end-to-end timing, and supports
  diagnostic backend forcing through the URL query string.
- Governing Chromium capture of the selected pixel-runtime canvas buffer, so
  acceptance and manifests use the deployed path rather than an internal
  proxy. On a 1,788-splat 476 x 502 Chameleon checkpoint, 32F completed in
  roughly 16-19 ms and differed from exact CPU on six pixels by one byte.
- A 21-image Chrome backend gate: 32F was selected on every image with at most
  one byte error and -0.0000014 worst source SSIM_sRGB delta; 16F passed its
  runtime sample on 20 images and correctly fell back to exact Worker CPU on
  checkerboard.
- A browser-native Canvas compositor that submits one transformed Canvas 2D
  radial gradient per splat, with optional three-plane mouse parallax and
  governing Chromium capture.
- A reproducible scriptless CSS linear-compositor MVP. Reversing Chameleon's
  DOM draw order and using linear-sRGB fills with exact Gaussian alpha masks
  improved the unchanged 1,615-splat population from 0.7017 to 0.8748
  SSIM_sRGB and from 0.3182 to 0.1456 LPIPS.
- A `css` export target that emits one scriptless DOM/CSS gradient
  ellipse per Gaussian, with exact-size Chromium grading and optional CSS-only
  10x10 hover-grid parallax over saliency-derived depth planes.
- Versioned, target-specific repeat-render noise floors for the Chrome pixel runtime,
  Playwright Chromium SVG, and Microsoft PowerPoint slideshow captures.
- A resumable full-corpus calibration tool that retains repeated captures and
  records renderer, artifact, capture, timing, and metric provenance.
- An offline adaptive pixel-runtime simulator for replaying stage checkpoints and
  comparing existing 2k/4k artifacts under guarded SSIM and LPIPS policies.
- A default-off online pixel-runtime controller that can stop before densification,
  later stages, and residual detail after the current run reaches an explicit
  margin-adjusted Chrome quality target.
- Content-addressed adaptive-policy options in the full-corpus benchmark.
- A resumable full-frame pixel-runtime checkpoint parity calibrator and versioned
  48-checkpoint model-to-Chrome evidence.
- Versioned exact replay of all 84 raw pixel-runtime stage checkpoints across the
  21-image corpus, including a predeclared compute go/no-go comparison for
  SSIM targets 0.98 and 0.979.
- A `capture` installation extra for exact Chrome pixel-runtime, native Canvas,
  CSS, and SVG capture from SplatThis's own virtual environment.
- A reusable native-dimension Playwright SVG capture CLI with explicit
  viewport, DPR, resource-wait, warm-up, timing, and repeat-hash provenance.
- A deterministic SVG recipe gate and full-corpus comparison overview for
  standard, palette-quantized, and native-blur artifacts.
- Versioned evidence for 63 native-dimension Chromium SVG captures:
  palette-quantized safely won seven images, blur won none, and the result
  cleared the predeclared five-image gate for a default-off integration slice.
- A max-fidelity SVG compositor gate that browser-grades historical order,
  corrected standard gradients, and corrected adaptive high gradients before
  accepting or reverting the complete artifact. Its fixed ROIs, full metric
  vector, compressed size, latency, and selection are retained in the manifest.
- A 33-population, 21-image SVG compositor corpus pass. Correct painter order
  improved median SSIM by 0.08513 and LPIPS by 0.09097 at unchanged gzip size;
  adaptive high stops added 0.01019 median SSIM with 56.8% gzip growth.
- A 21-image native-PPTX painter-order corpus pass using 42 real Microsoft
  PowerPoint slideshow captures. Corrected order improved median SSIM by
  0.02662 and LPIPS by 0.03346; the guarded per-image policy selected it for
  14 images and retained legacy order for seven.
- Explicit `--pptx-painter-order legacy|back-to-front` production emission,
  manifest provenance, and a resumable external real-PowerPoint selector that
  atomically materializes the accepted native deck as `selected.pptx`.

### Changed

- The former `canvas` output is now named `pixel-runtime`, because it
  software-rasterizes the splat equations into an `ImageData` framebuffer.
  `canvas` now means the genuine Canvas 2D primitive compositor. Historical
  corpus and noise results are labelled as pixel-runtime evidence.
- Same-population Chameleon documentation now distinguishes the actual
  browser SVG (0.7076 SSIM), browser-gated palette recipe (0.7274), internal
  software preview (0.8803), and pixel runtime (0.9045).
- Pixel-runtime capture can retain every repeated browser rendering for calibration
  instead of keeping only the final PNG.
- Adaptive pixel-runtime targets now denote desired Chrome quality and are evaluated
  with a deployed scorer that mirrors JavaScript double math, Float32Array
  accumulation, and 8-bit sRGB ImageData packing. All 48 calibrated
  checkpoints matched Chrome pixel-for-pixel, so the default model-to-browser
  safety margin is zero; optional cross-version overrides remain available.
- ADR-003 now distinguishes the implemented artifact-gate foundation,
  retrospective simulation, and bounded online hard-target controller from
  still-proposed predictive allocation, selective 8k scaling, broader
  operators, and hybrid residuals.
- The adaptive simulator now rescores canonical raw stages at the exact ImageData
  byte boundary instead of depending on historical continuous manifest scores,
  and models only the online hard-target stop rather than offline plateau or
  regression rules. Both tested targets saved 1.3% of aggregate stage time,
  below the 5% gate, so further hard-target A/B expansion is not recommended.
- Pixel-runtime calibration now defaults to the active SplatThis interpreter instead
  of a Playwright environment in the sibling `svg2pptx` repository.
- The SVG recipe gate now evaluates browser-delivered SVG with a reused
  Chromium process instead of librsvg. One unmeasured filter warm-up precedes
  three required byte-identical captures, avoiding a first-draw one-level
  pixel wobble without charging browser startup to every recipe.
- Browser-delivered SVG is now governed by one shared Playwright Chromium
  renderer across core export grading, ADR-003 fidelity, corpus benchmarking,
  artifact calibration, and experimental runners. CairoSVG and librsvg are no
  longer implicit deployed-artifact fallbacks. If Chromium is unavailable, SVG
  acceptance now fails closed while retaining proxy metrics as diagnostics only.
- Repeated SVG evaluation now reuses one Chromium page and decodes governing
  frames in memory; per-artifact geometry is validated once before measured
  repeat captures.
- SVG gradient fidelity is now a policy (`standard` or adaptive `high`) rather
  than another primitive recipe. High uses a 0.005 opacity-curve error bound,
  up to nine stops, and four-decimal opacity.

### Fixed

- Score pixel-runtime checkpoints at the emitted 8-bit framebuffer boundary instead
  of comparing a continuous float proxy with Chrome. This removes up to
  0.001102 SSIM of systematic score overstatement in the calibration set.
- Preserve the virtual-environment Python symlink when launching Playwright
  capture, rather than resolving it to a system interpreter without the venv's
  packages; checkpoint calibration now also fails dependency checks once,
  before entering the corpus loop.
- Emit standard, browser-compatible, scripted-matrix, palette-quantized, and
  blur SVG splats in back-to-front painter order while leaving the exact
  front-to-back pixel runtime unchanged.

## 0.2.0 - 2026-07-30

### Added

- Target-aware pixel-runtime, SVG, and native DrawingML/PPTX export pipelines.
- MLX optimization with periodic tile-plan rebuilds and full geometry training.
- Actual-artifact SVG evaluation and real-PowerPoint corpus capture tooling.
- Monotonic pixel-runtime checkpoint and post-processing gates.
- Browser-pixel-buffer corpus metrics, per-picture comparisons, and budget history.
- Optional layered pixel-runtime parallax.
- Experimental Top-K distillation and mixed-primitive MVP tooling.

### Changed

- MLX now defaults to eight tiles per batch. Three real corpus checkpoints
  measured 12-34% lower median forward-and-backward latency than a 16-tile
  batch without changing render math; on the Chameleon converter-default
  comparison it was 23% lower than 128 tiles.
- The corpus overview renders canvases lazily near the viewport.
- Canvas splats are sorted once during export instead of at every page load.
- Internal PNGs are named and described as splat proxies, not screenshots of
  the emitted SVG or PPTX.
- Package metadata and documentation now report full-corpus, deployed-artifact
  results instead of extrapolating from a favorable single image.

### Fixed

- Correct rotated anisotropic footprint bounds in Canvas and NumPy rendering.
- Fall back to Torch when MLX imports but no Metal device is available.
- Publish SVG, PPTX, Canvas, JSON, manifest, and proxy-PNG outputs atomically.
- Escape Canvas document titles and reject invalid canvas dimensions.
- Report the real number of parallax layers instead of JSON-string length.
- Return concise CLI errors for invalid or unreadable input.

### Compatibility

- Python 3.13 or newer is required.
- MLX remains optional and requires an Apple-Silicon Metal session.
- PPTX quality claims refer to Microsoft PowerPoint slideshow captures;
  internal proxies and LibreOffice renders are labeled separately.
