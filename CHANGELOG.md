# Changelog

All notable changes to SplatThis are documented here.

## 0.2.6 - 2026-08-02

### Changed

- **Defaults now match the corpus evidence.** Two defaults the August audit
  flagged as measured losers are flipped, and one API break is repaired:

  PPTX emits corrected back-to-front shape order by default -- it matches the
  renderer's transmittance model and won the 21-image real-PowerPoint corpus
  (median LPIPS 0.320 vs 0.375, 19 improvements / 1 regression).
  `--pptx-painter-order legacy` reproduces the historical stack.

  The SVG compositor gate's incumbent is now the default corrected-standard
  emitter instead of the legacy order. Under the old incumbent the gate
  medianed 0.7111 SSIM against 0.7404/0.7483 for either fixed corrected
  choice, because four corpus images kept legacy purely by incumbency; legacy
  stays in the race but now has to win an image outright, so the gate's
  floor is the default emitter's output.

  `CompositorFidelity` is restored as a deprecated alias of `Fidelity`, so
  0.2.5-era imports survive the rename (the field set still changed with the
  deployed/compositor split).

- **The PPTX post-fit stage was making decks worse; fixed and measured.**
  `--pptx-proxy-postfit-iters` refined colour and alpha against the
  soft-edge alpha law even for `gradient` decks -- the right constant with
  the wrong curve, under-modelling opacity by up to 27% at high alpha -- and
  because the stage also picks its best iterate with that same model, it
  reported a gain while degrading the artifact. Measured on chameleon with
  60 iterations, real PowerPoint captures, only the law changed:

      no post-fit          SSIM 0.8395  LPIPS 0.2075  dE 0.0568
      post-fit, before     SSIM 0.8198  LPIPS 0.2253  dE 0.0449
      post-fit, after      SSIM 0.8386  LPIPS 0.1935  dE 0.0322

  The stage now improves the deployed deck by 0.014 LPIPS instead of
  costing 0.018, and nearly halves colour error. Both laws live in one
  shared `proxies.pptx_effective_alpha` used by the training proxies and the
  post-fit alike, so they cannot drift again; a regression test pins each
  against its emitter. Defect predates the current naming; opt-in, so no
  released default changed.

- **`--training-export-target pptx-gradient`: fit the splats for the deck
  PowerPoint actually draws.** The DrawingML gradient emitter writes stops of
  `1 - exp(-PPTX_GRADIENT_ALPHA_SCALE * alpha * G(r))` and PowerPoint
  composites them alpha-over in display sRGB, while training defaulted to a
  linear-light true Gaussian -- a train/deploy split of exactly the kind that
  cost browser SVG half its quality until 0.2.x. The new target closes it,
  and it is a twenty-line parameter transform rather than a new renderer:
  scaling the alpha column by the constant the emitter already applies makes
  the base renderer reproduce the deployed opacity curve at every stop, with
  the piecewise-linear residual bounded by the emitter's own stop-placement
  error budget.

  Implemented in the existing proxy architecture -- `_PPTXGradientProxyRenderer`
  in `proxies.py` beside the soft-edge proxy, its MLX mirror in
  `mlx_renderer.py`, torch/MLX parity and emitter-curve fidelity pinned by
  tests. Measured on one image against a real PowerPoint capture: chameleon
  LPIPS 0.2075 -> 0.1895, SSIM 0.8395 -> 0.8557, and OKLab colour error
  halved at 0.0568 -> 0.0289, on the shipped primitive at 1,675 shapes and
  4 seconds to open -- beating the ring-stack line that cost 8x the shapes
  and 34 seconds. Opt-in, not the `auto` default: that needs a corpus pass.

  Known follow-up: `_postfit_splats_for_pptx_proxy` still refines against a
  plain Gaussian, so a `pptx-gradient` run that also enables
  `--pptx-proxy-postfit-iters` trains and post-fits on different curves.
  Left unchanged pending measurement rather than patched silently.

- **Ring-stack PPTX primitive: investigated, measured, closed.** Replacing
  DrawingML gradient splats with solid-alpha ring stacks -- plus a
  ring-aware differentiable fit -- did beat the gradient baseline on the
  hardest test image and nearly halved its OKLab colour error, but requires
  K=8 rings to render contour-free, which costs 8x the shapes and **34
  seconds to open one slide against 4 seconds**. Quality and cost share one
  knob, so the line is closed rather than parked; both the result and the
  flawed "exactly modelable" premise that motivated it are recorded in the
  report's negative results and `docs/pptx-ring-stack.md`.

- **PowerPoint captures are now color-managed, and the fix moved nothing it
  shouldn't.** The capture path converts every screenshot through its
  embedded ICC profile to sRGB (macOS records Display P3; the primaries of
  every prior PowerPoint score were desaturated). The corpus was re-captured
  and re-scored under the corrected protocol: headline medians held (SSIM
  0.6279, LPIPS 0.3212) because the bias lived in the color axis, and the
  sRGB-training experiment reproduced its split verdict on clean data -- so
  the linear training default stays, now chosen on unbiased evidence.

- **PowerPoint's color space is measured, and it is a hybrid.** A synthetic
  probe deck captured in real PowerPoint (`tools/pptx_colorspace_probe.py`)
  shows gradFill color ramps interpolating in linear light (opposite of
  browsers) while alpha compositing happens in display sRGB -- neither of the
  project's training models, which explains the sRGB-training MVP's split
  decision (`tools/pptx_srgb_training_mvp.py`). The probe's calibration
  swatches also exposed a capture-chain bias: macOS screenshots record
  Display P3 coordinates that the scoring pipeline reads as sRGB, so every
  PowerPoint capture is scored with desaturated primaries. Findings and
  implications in `docs/pptx-colorspace.md`; the capture-profile fix is a
  governing-protocol change and is deliberately not patched quietly here.

- **The corpus gallery shows every format for every image.** Five columns
  per corpus image at /corpus/: the original, the scripted pixel runtime
  rendering live (JS/WebGL evaluating the splat formula), the scriptless CSS
  build as live DOM, the corrected-exporter SVG as a live vector, and the
  deck as a real-PowerPoint slideshow capture with the editable .pptx one
  click away. Captions carry LPIPS, SSIM, splat count, and artifact size per
  cell, quoted from the freshest ledger for each format; splat counts are
  recorded at emit time in docs/corpus/stats.json and the build fails closed
  on any missing asset.

- **The home page shows the regenerated corpus, end to end.** A fourth
  generated region on the landing page publishes per-format deployed medians
  over all 21 governing images -- the fresh schema-v2 SVG rerun, the
  regenerated real-PowerPoint pass, and the registry's CSS and pixel-runtime
  blocks -- guarded in CI like the rest.

- **The schema-v2 ledger regeneration ran.** 21 fresh governing SVG rows
  (`run_tag v2-governing-aug2026`): seed-0 populations retrained by current
  code under current defaults, Chromium-captured, median LPIPS 0.2392 / SSIM
  0.7509 -- at the published expectations, with every image within noise of
  its registry row. `powerpoint_results.jsonl` fully regenerated by the
  attended pass (`tools/run_powerpoint_pass.py`): all 21 decks re-emitted
  under the corrected-order default and captured from real PowerPoint,
  reproducing the order-study medians exactly (LPIPS 0.3200 / SSIM 0.6279).
  Artifacts are on disk again; `result/README.md`'s do-not-aggregate warning
  now describes a regenerated ledger instead of a lost one.

- **The demo deck ships the corrected order it now defaults to.** The
  showcase `chameleon.pptx` and its real-PowerPoint capture were a
  legacy-order build predating the default flip; both are replaced with the
  order study's corrected-order artifacts for the same 1,674-splat
  population (SSIM 0.7931 -> 0.8379, LPIPS 0.2621 -> 0.2085 on the same
  real-PowerPoint protocol).

- **Both audit decisions are now measured, not just argued.** Re-running
  the gate study under the corrected-standard incumbent (same populations,
  same policy, fresh Chromium captures) selected `high` on 15 images and the
  default `standard` on 6 -- legacy never won an image outright -- lifting
  gate medians from 0.7111 to 0.7483 SSIM (`svg-compositor-corpus-v2.json`,
  `tools/validate_svg_gate_incumbent.py`). And the registry's last provenance
  caveat closed: re-emitting all 21 pixel-runtime artifacts from current code
  and capturing them in governing Chrome reproduced the historical ledger's
  deployed medians exactly (0.2443 LPIPS / 0.7751 SSIM), so
  `expected_fidelity("pixel-runtime")` now publishes deployed figures
  (`tools/measure_pixel_runtime_deployed.py`).

- **CSS now has its own calibrated repeat-render noise floor.** The
  calibration tool gained a `css` target that captures the committed
  corpus-gallery builds -- emitted by the shipped emitter from the exact
  populations the fidelity registry measured -- in governing Chromium. All 21
  artifacts at five repeats each measured a span of exactly zero on every
  metric, matching SVG and the pixel runtime, so CSS no longer borrows the
  seed-noise floor for its indistinguishability claims.
  `data/artifact-gates.json` carries the section and provenance; native
  Canvas is now the only browser target without its own floor.

### Fixed

- **The shipped CSS exporter now produces what the README showcases.** The demo
  scored 0.8748 while `splatthis --format css` produced 0.7017 on the same
  splats: the showcase had been generated by `tools/css_linear_compositor_mvp.py`,
  a research variant whose wins were never promoted into the emitter. Ported,
  and it reproduces the showcase figure exactly.

  Three changes, all measured on the same population: emit back-to-front, since
  CSS composites in DOM order permanently; emit colour in linear sRGB and apply
  the Gaussian falloff as an alpha mask over a solid fill, so the browser stops
  interpolating colour and opacity together and darkening every splat's skirt;
  and sample the exact opacity curve at nine evenly spaced stops instead of
  placing them adaptively, which preserves the tail that alpha-over accumulation
  depends on.

      compositor fidelity, median   0.7488 -> 0.9426
      p10                           0.6510 -> 0.9154
      improved on                   21 of 21 corpus images

### Changed

- **Published expectations now keep deployed and compositor fidelity apart.**
  `expected_fidelity()` replaces `compositor_fidelity()` as the entry point
  (the old function name remains as a deprecated alias; the
  `CompositorFidelity` dataclass is renamed `Fidelity` without one). Each
  measured format publishes *deployed* fidelity -- the artifact against the
  original image, what a user of this tool gets -- separately from
  *compositor* fidelity, the emitter-only term that matters to a caller
  supplying its own splats. Deployed, the declarative emitters are
  indistinguishable: median LPIPS 0.2433 / 0.2439 / 0.2429 for
  svg / svg-high / css, with an SSIM spread below the 0.029 seed noise floor.
  Compositor-only, from the shipped emitters:

      svg             median 0.9303 SSIM_sRGB   0.0310 LPIPS
      svg-high        median 0.9486             0.0181
      css             median 0.9426             0.0221
      pixel-runtime   median 0.9993             0.0001

  This corrects the svg compositor median published under 0.2.5 (0.7540),
  which was measured on artifacts that did not come from the shipped emitter,
  and retires the conclusion drawn from it ("CSS overtakes SVG by 0.19"):
  with the corrected number the three declarative emitters sit within 0.018
  SSIM of each other. The exact-stop sampling half of the CSS technique
  already transfers to SVG's evenly spaced gradient stops; the alpha-mask
  half cannot, and no longer needs to carry the difference it was thought to.

- **The README quality tables are generated, not maintained.**
  `tools/update_readme.py` renders the corpus-results block from the
  committed ledgers, cross-checking every published median against its
  per-image evidence -- and the compositor SSIM medians against the
  independent `result/svg-quality/measurements.json` -- before writing
  anything. `tests/unit/test_readme_results.py` runs its `--check` mode, so a
  ledger change that is not reflected in the README fails CI. The Pages
  landing page gets the same treatment: its hero SVG card and both
  number-bearing tables are generated by `tools/update_landing.py` and
  guarded by `tests/unit/test_landing_page.py`, and the hero now showcases
  the corrected exporter instead of apologizing for a historical artifact.

- **The Pages demo is live everywhere a browser allows.** Fixed-pixel demo
  documents no longer stretch wrongly inside their iframes -- an iframe never
  scales its document, so a scale-to-fit wrapper now does. The landing page
  gains a PowerPoint section: a real slideshow capture with its ledger scores
  and the editable deck, the one target that cannot render live. And
  `/corpus/` shows the whole 21-image governing corpus side by side --
  source, corrected-exporter SVG, and scriptless CSS builds emitted by the
  shipped emitter from the exact populations the registry measured -- all
  rendered live, generated by `tools/build_corpus_gallery.py` and guarded by
  `tests/unit/test_corpus_gallery.py`.

- `--profile` validates its choices at argparse (`m4-fast-loop`, `fast`,
  `balanced`, `max-fidelity`, from `profiles.PROFILE_NAMES`); an unknown
  profile used to surface as a late `ValueError` and the valid names were
  documented nowhere. `--svg-recipe`'s help no longer claims its default
  comes from the quality profile -- no profile sets one; the default is
  `standard` unconditionally.

- **The technical report is published, and it is living.** `paper/report.md`
  gained its missing sections (introduction, OOXML findings, conclusion,
  references), withdrew the renderer-independence claim that later artifacts
  falsified, and dated its historical eras. Its §5.4 -- the corrected-emitter
  results and the fit/emitter decomposition -- is regenerated from the
  versioned ledgers by `tools/update_paper.py`, which also renders the report
  to GitHub Pages at `/paper/`; `tests/unit/test_paper_page.py` fails CI when
  either copy goes stale. `CITATION.cff` now carries a `preferred-citation`
  for the report instead of a comment explaining why it could not.

## 0.2.5 - 2026-08-02

### Added

- `compositor_fidelity("pixel-runtime")`. The runtime evaluates the splat
  formula directly rather than approximating it, and measures effectively
  lossless against the reference renderer -- median 0.9993 SSIM_sRGB against
  SVG's 0.7540. For a consumer supplying its own splats that choice dominates
  every other tuning decision. Published under the honest label: the corpus
  records these rows as `canvas`, but they are the ImageData renderer, not the
  native Canvas 2D gradient emitter, which still has no measurement and still
  raises.

### Changed

- The README leads with what the corpus actually shows. "Seed-0 medians, none
  reached 0.99" reads as a weak result; the stronger and equally honest
  statement is that fidelity is predicted by content, not by format. Mean
  gradient magnitude correlates with SSIM at r=-0.84 across the 21-image
  governing corpus, and splitting at the median gradient gives 0.72 for the
  smooth half against 0.53 for the textured half -- a gap larger than any
  between two output formats. Quoted as full-population figures, with the LPIPS
  relationship stated separately at r=+0.53 rather than implied equally strong.
- Scriptless CSS renders live on the GitHub Pages site. It cannot appear in a
  README because both GitHub and PyPI strip the `style` attribute and `<style>`
  tag, which is the entire substance of that build.

### Fixed

- `paper/report.md` has no unmeasured sections. The seed noise floor is 0.029
  SSIM_sRGB worst-case over six images and three seeds, taken as the worst case
  rather than the median so a claim survives the least favourable image.
- The corpus-wide SVG-versus-OOXML comparison contradicts a claim this project
  had been carrying. PowerPoint leads by 0.012 SSIM and 0.013 LPIPS at the
  median, both below the seed noise floor, and wins 12 of 21 images on LPIPS --
  a coin flip. The quality difference is not claimable; the size difference is:
  5.3x smaller at indistinguishable quality. The earlier "PPTX wins
  perceptually" finding came from a single image that happens to favour it.
- Corpus result rows are explicitly labelled `schema_version: 1`, so the mixed
  evidence-level state is visible in the data rather than inferable from which
  keys are absent.

## 0.2.4 - 2026-08-01

### Added

- `splatthis.compositor_fidelity()`, publishing how faithfully a deployed
  compositor reproduces the *same* splat population the internal reference
  renderer draws. A consumer that supplies its own splats -- projecting 3D
  Gaussians through EWA rather than fitting a bitmap -- inherits only this
  term, not the fitting loss that dominates the headline source-fidelity
  numbers, and previously had to re-derive it from the corpus.

      compositor SSIM_sRGB   min 0.5965   median 0.7540   max 0.8991   (n=21)
      r(content gradient)    -0.470

  The correlation is the substantive result. End-to-end fidelity correlates
  with content gradient at -0.84, which invites the conclusion that a splat
  view's quality is predictable from what it depicts. It is not: that
  dependence is almost entirely the fitting stage, and compositor loss is
  broadly uniform. `is_content_predictable()` returns that judgement so it
  cannot be re-inferred wrongly.

- `--save-json` on the CLI. `ConversionRequest.save_json` was honoured but
  unreachable from the command line.

### Changed

- Acceptance no longer gates on wall-clock by default. `max_runtime_sec: 60.0`
  failed correct runs against presets that train for minutes -- 8 of 70 stored
  manifests failed on runtime alone, including full-quality runs at SSIM 0.91.
  Runtime remains measured under `acceptance.measured` and honoured when set
  explicitly or by `--time-budget`.
- Run manifests record `artifact_hash_stable` (false under MLX, with a note)
  and `memory_guard`, so reproducibility and host-memory reductions travel with
  the data instead of living in prose.
- Corpus result rows carry `schema_version`, and resume re-scores anything
  older, so an evidence level can no longer vanish across a resumed run.

### Fixed

- A failing SSIM silently became an inflated one: `except Exception` around the
  skimage call fell through to a global SSIM reading ~0.10 higher, far above
  the 0.50 acceptance floor. Only an absent dependency or an input too small to
  carry a window may fall back now.
- Colour transforms no longer raise a negative base to a fractional exponent on
  the discarded branch of `np.where`, which warned during ordinary compositing.
- The splat-orientation convention is pinned by direct tests, and
  `mixed_primitives.py` no longer open-codes it.

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
