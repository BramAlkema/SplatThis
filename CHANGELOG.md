# Changelog

All notable changes to SplatThis are documented here.

## 0.2.3 - 2026-08-01

### Changed

- The README showcase renders all four outputs at one size and adds scriptless
  CSS as a fourth column. The previous table sized each image by its own
  intrinsic pixels -- 476x502, 364x384 and 728x768 -- so a three-way comparison
  appeared at three different scales.

  Sized with explicit `width` attributes rather than by resampling the files:
  PyPI's `readme_renderer` allowlist keeps `width` and `height` on `<img>`, and
  GitHub does too, so every asset stays at native resolution and stays crisp on
  high-DPI displays.

  CSS appears as a Chromium capture rather than live. Both platforms sanitize
  README markup through an allowlist that omits the `<style>` tag and the
  `style` attribute -- `readme_renderer` runs `nh3.clean()` with no CSS
  sanitizer configured -- so a build made entirely of CSS radial gradients
  arrives as 1,615 unstyled `<div>` elements. Each cell is labelled as live
  vector or as a capture, and captures are noted as lower bounds: they carry
  screen-capture and rescaling losses on top of the compositor's own.

### Fixed

- The relative-link guard now covers `src="..."` as well as markdown `](...)`.
  The showcase moved to raw `<img>` tags for sizing, which had made the
  markdown-only check blind to every image on the page -- the same class of
  defect that put 18 dead links on the 0.2.0 project page.

## 0.2.2 - 2026-08-01

### Changed

- The README showcase now carries the current best artifact for each
  compositor, selected by measurement. The published demo scored 0.7270
  SSIM_sRGB in Chromium at its own native size while artifacts already on disk
  reached 0.8665; it was also 379x400 against a 476x502 source, which is why
  `evaluate_svg_export_quality` refused to score it at all — the geometry
  mismatch fails closed, so the published demo could not be measured without
  hitting that wall first.

      SVG   corrected-high    0.8665  1.51 MB  native vector, embedded
      CSS   exact9-fp2.875    0.8748  0.80 MB  native DOM, linked live
      PPTX  corrected-order   0.7885  161 KB   real PowerPoint capture

  SVG and CSS appear as themselves rather than as screenshots. GitHub and PyPI
  both strip `<style>` elements and `style` attributes from rendered markdown,
  so a scriptless CSS build cannot render inline in a README and is linked
  live instead; an SVG loaded as `<img>` is rendered as its own document and
  does apply its internal CSS, which is why that one embeds. PowerPoint is the
  single screenshot, because photographing a real slideshow is the only honest
  way to show what PowerPoint draws, and its score is measured through that
  capture and is therefore a lower bound.

  SVG and CSS share one 1,615-splat population, so their difference is purely
  compositor. The deck is a separate 1,674-splat run and is not a like-for-like
  comparison against them.

### Added

- `tools/refresh_showcase.py`, which re-scores every surviving artifact in its
  governing renderer and installs the winners, ignoring recorded metrics
  entirely. Recorded numbers are a record of what a past run saw, not of what
  is on disk: the CSS experiment reported a 0.8748 winner that nothing in the
  named directory could match, because the artifact was in a `stage2/`
  subdirectory under a different name. Selecting by directory name also pulled
  a canvas render into the CSS candidate set, where it outscored every real CSS
  build. Near-ties settle on file size, bucketed so a few hundred bytes cannot
  outrank a real quality difference.

## 0.2.1 - 2026-08-01

### Fixed

- Absolute URLs throughout the README. It doubles as the PyPI project
  description, and PyPI does not rewrite relative links the way GitHub does, so
  all 18 of them resolved against `pypi.org/project/splatthis/` and every image
  and document reference on the published 0.2.0 page pointed at nothing. A
  release description is immutable, which is why this needs a version of its
  own. Guarded by a test rejecting relative links, since the defect is
  invisible until after publishing.

### Added

- An end-to-end regression test that converts the tracked demo image and
  asserts a quality floor, because 88% line coverage did not notice that the
  packaged SVG templates were missing from every fresh checkout. The existing
  smoke test does run a conversion but only asserts the output parses, so it
  cannot see valid-but-wrong output: reverting the splat-orientation convention
  leaves it green while the new floor fails at SSIM_sRGB 0.460974 against
  0.510. Marked PROXY evidence — it detects that the pipeline moved and may
  never approve a browser artifact.

### Changed

- A platform-aware coverage floor in CI. MLX installs only on Apple Silicon, so
  roughly 705 statements across the five `mlx_*` modules are unreachable on
  Linux and Windows; measured 87.44% macOS against 79.64% ubuntu and 79.66%
  windows. One global floor would be vacuous on macOS or unmeetable elsewhere.
- The mypy exemption comment now describes the real 462 findings and the
  condition for lifting it, rather than a stale "~300".

## 0.2.0 - 2026-08-01

First publication to PyPI: `pip install splatthis`. Continues the 0.2.0
development recorded under the 2026-07-30 entry below.

### Added

- A PEP 561 `py.typed` marker, declared in both `package-data` and
  `MANIFEST.in`. The package exports 26 names as a library API and is annotated
  throughout, but without the marker every consumer type-checking against the
  installed distribution saw `Any`. Verified by installing the built wheel into
  a clean environment and resolving `save_svg` and `GaussianSplat` to their real
  signatures.
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

- Reworked the release workflow onto PyPI Trusted Publishing, matching the
  sibling `openxml-audit` repository, which has shipped five releases through
  the same shape. No API token is stored in the repository: PyPI mints a
  short-lived credential per workflow run via OIDC. The trigger also moves from
  `release: published` to `push: tags: v*` — the old trigger required a GitHub
  Release to exist before the workflow would run, so pushing a tag alone did
  nothing; the workflow now creates the release itself. Build, PyPI publish, and
  GitHub release are separate jobs sharing one built artifact, and the wheel
  smoke test runs in a throwaway venv so it cannot be satisfied by build
  dependencies present in the job environment.
- Declared the conversion engine's shared mixin surface in `engine_state.py`:
  35 state attributes and 32 cross-mixin method signatures, inherited by all
  seven mixins. Only 4 of those attributes previously carried any annotation.
  The declaration is inert — no runtime members, last in the MRO — so state
  ownership and method implementations are unchanged, and a seeded Torch
  conversion emits a byte-identical SVG before and after. Corrected the
  dependency inversion where artifact backends and pipeline phases annotated
  the public `PNG2SVGConverter` facade rather than the `ConversionEngine` base
  they actually receive. Together: mypy 778 → 462 errors, `attr-defined`
  340 → 1.
- Documented what a fixed `--seed` actually guarantees per optimizer backend.
  Torch reproduces the emitted artifact byte for byte; MLX does not. Metal
  orders float32 reductions nondeterministically, so repeated single-process
  seeded MLX runs differ by roughly one float32 ULP (~3e-8) in splat
  parameters, which is enough to tip a rounded SVG attribute across a
  formatting boundary. Reported metrics still agree to nine significant
  figures, so the quoted four-decimal corpus medians are unaffected. The
  previous note attributed MLX nondeterminism solely to concurrent processes
  sharing one Metal device; it also occurs in a single process. "Reproducible
  manifests" in the project status is now "provenance-complete manifests".
- Corrected package metadata that would have been immutable after the first
  release: the author contact was a placeholder for a nonexistent domain, and
  the summary and keywords claimed SVG-only conversion.
- Collapsed four project names onto one. The distribution is now `splatthis`
  (was `splat-this`), the import path is `splatthis` (was `png2svg_gs`), and the
  console entry point is `splatthis` (was `splatlify`). The repository, the
  GitHub Pages site, and the project name in prose are unchanged. `png2svg_gs`
  had also become inaccurate: the package emits SVG, PPTX, native Canvas, CSS,
  and pixel-runtime artifacts, not only SVG. No compatibility shim or alias
  entry point is provided — the package has never been published to PyPI, so
  there is no installed base to migrate.
- Removed the unreferenced root-level `png2svg` launcher. It exposed an older,
  divergent flag surface (`--max-splats`, `--output-format {svg,drawingml,pptx}`,
  `--backend {auto,torch,gsplat}`) that no longer matched the packaged CLI, and
  its `gsplat` backend has been retired and is now explicitly rejected. The
  packaged `splatthis` entry point is the only supported command-line surface.
- Replaced the state-restoring monolithic conversion run with an immutable
  `ConverterConfig`, per-call `ConversionRequest`/`RunContext`, and explicit
  prepare, fit, and deployment phases. The public converter no longer mutates
  and restores its own budget state, and its coordinator is now a thin facade.
- Reduced `converter.py` to a stable public facade and `conversion_engine.py`
  to a 123-line composition root. Configuration, initialization, optimization,
  densification, post-fit, artifact strategy, and regional guidance now have
  separate internal modules; constructor branches are isolated behind focused
  setup helpers.
- Added one artifact-backend registry for SVG, DrawingML, PPTX, native Canvas,
  CSS, and pixel-runtime emission, persistence, governing evaluation, and
  evidence provenance. PPTX persistence reuses its already-emitted DrawingML
  payload instead of generating the full shape tree twice.
- Split atomic storage, pure quality metrics, browser artifact evaluation,
  reporting, and round-trip validation into one-way dependency layers.
  `artifact_io` is now also a compatibility facade, and the side-by-side HTML
  document moved to a packaged template.
- Replaced the 3,541-line I/O monolith with focused artifact-I/O, shared export,
  browser, SVG, PPTX, pixel-runtime, and color modules. `splatthis.io` remains
  a 132-line compatibility facade, while production code imports the focused
  implementations directly. SVG, DrawingML, and PPTX package markup now lives
  in packaged templates rather than Python source; representative emitted
  artifacts remain byte-identical. NumPy color transforms now have one shared
  implementation.
- Torch/CPU corpus runs accept opt-in `--jobs N` subprocess parallelism while
  keeping JSONL writes and artifact scoring coordinated in one thread. Parallel
  MLX runs are rejected because shared-Metal processes caused seeded parameter
  drift. Resumability now reuses only successful records whose primary artifact
  still matches its recorded size and SHA-256, so failed, corrupted, and
  partially deleted runs are retried instead of becoming false cache hits.
- The explicit `fast` MLX profile now renders 32 tiles per batch. A 21-image,
  full-frame 512-splat sweep reduced median optimizer time by 42% and median
  wall time by 23%; balanced and max-fidelity retain the conservative 8-tile
  behavior because low-budget per-image fidelity moved in both directions.
- Conversion reuses its already-computed linear proxy framebuffer when writing
  the optional proxy PNG, eliminating a second full CPU splat render. Internal
  Python pixel-runtime helpers now use explicit names; the ambiguous historical
  `generate_canvas_html` aliases were removed.
- Internal `*_splat_proxy.png` files are now emitted only when explicitly
  requested or required by a side-by-side report, instead of appearing beside
  every normal conversion and being mistaken for the deployed artifact.
- Removed the unshipped gsplat adapter, which depended on private legacy 2D
  APIs and was neither packaged nor used by release benchmarks. Renderer
  `auto` now resolves deterministically to the Torch reference backend.
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

## 0.2.0 (earlier development) - 2026-07-30

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
