# Changelog

All notable changes to SplatThis are documented here.

## Unreleased

### Added

- Versioned, target-specific repeat-render noise floors for Chrome Canvas,
  `rsvg-convert` SVG, and Microsoft PowerPoint slideshow captures.
- A resumable full-corpus calibration tool that retains repeated captures and
  records renderer, artifact, capture, timing, and metric provenance.
- An offline adaptive-Canvas simulator for replaying stage checkpoints and
  comparing existing 2k/4k artifacts under guarded SSIM and LPIPS policies.
- A default-off online Canvas controller that can stop before densification,
  later stages, and residual detail after the current run reaches an explicit
  margin-adjusted Chrome quality target.
- Content-addressed adaptive-policy options in the full-corpus benchmark.
- A resumable full-frame Canvas checkpoint parity calibrator and versioned
  48-checkpoint model-to-Chrome evidence.
- Versioned exact replay of all 84 raw Canvas stage checkpoints across the
  21-image corpus, including a predeclared compute go/no-go comparison for
  SSIM targets 0.98 and 0.979.
- A `capture` installation extra for exact Chrome Canvas capture from
  SplatThis's own virtual environment.

### Changed

- Canvas capture can retain every repeated browser rendering for calibration
  instead of keeping only the final PNG.
- Adaptive Canvas targets now denote desired Chrome quality and are evaluated
  with a deployed scorer that mirrors JavaScript double math, Float32Array
  accumulation, and 8-bit sRGB ImageData packing. All 48 calibrated
  checkpoints matched Chrome pixel-for-pixel, so the default model-to-browser
  safety margin is zero; optional cross-version overrides remain available.
- ADR-003 now distinguishes the implemented artifact-gate foundation,
  retrospective simulation, and bounded online hard-target controller from
  still-proposed predictive allocation, selective 8k scaling, broader
  operators, and hybrid residuals.
- The adaptive simulator now rescores canonical raw stages at the exact Canvas
  byte boundary instead of depending on historical continuous manifest scores,
  and models only the online hard-target stop rather than offline plateau or
  regression rules. Both tested targets saved 1.3% of aggregate stage time,
  below the 5% gate, so further hard-target A/B expansion is not recommended.
- Canvas calibration now defaults to the active SplatThis interpreter instead
  of a Playwright environment in the sibling `svg2pptx` repository.

### Fixed

- Score Canvas checkpoints at the emitted 8-bit framebuffer boundary instead
  of comparing a continuous float proxy with Chrome. This removes up to
  0.001102 SSIM of systematic score overstatement in the calibration set.
- Preserve the virtual-environment Python symlink when launching Playwright
  capture, rather than resolving it to a system interpreter without the venv's
  packages; checkpoint calibration now also fails dependency checks once,
  before entering the corpus loop.

## 0.2.0 - 2026-07-30

### Added

- Target-aware Canvas, SVG, and native DrawingML/PPTX export pipelines.
- MLX optimization with periodic tile-plan rebuilds and full geometry training.
- Actual-artifact SVG evaluation and real-PowerPoint corpus capture tooling.
- Monotonic Canvas checkpoint and post-processing gates.
- Browser-pixel-buffer corpus metrics, per-picture comparisons, and budget history.
- Optional layered Canvas parallax.
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
