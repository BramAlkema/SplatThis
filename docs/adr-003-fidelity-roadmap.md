# ADR-003: Artifact-Gated Fidelity Optimization

- **Status:** Accepted; roadmap active
- **Date:** 2026-07-28
- **Accepted on:** 2026-07-31
- **Revision:** 4
- **Authors:** SplatThis Development Team
- **Supersedes:** Revision 3 of this ADR

## Decision summary

SplatThis adopts artifact-gated, accept-or-revert optimization as the control
plane for fidelity work after ordinary splat training.

A candidate may be promoted only when it:

1. is evaluated with the compositor that users will actually receive, or with
   an exact model of that compositor whose parity is separately verified;
2. produces a meaningful measured gain;
3. stays inside explicit quality, shape-count, file-size, and runtime gates;
4. preserves the pre-fidelity baseline on protected metrics and regions; and
5. records enough provenance to reproduce the decision.

The ordinary converter remains the baseline and the fallback. An experiment
that loses its gate is reverted rather than becoming a new default.

This ADR accepts that architecture and the currently implemented bounded
slices. It does **not** claim that the entire fidelity roadmap has shipped.
In particular, predictive adaptive allocation and selective scaling, the
broader operator portfolio, hybrid raster residuals, and full artifact search
remain proposed. A narrower, default-off Canvas controller that stops on an
absolute observed quality target is implemented, but its exact full-corpus
replay missed the compute gate and will not be expanded in its current form.

## Context

ADR-002 established a deterministic initialize, optimize, refine, and export
pipeline over a shared 2D Gaussian representation. The same splat population
is then deployed through three materially different compositors:

- Canvas executes the project's linear-light alpha-over runtime.
- SVG uses browser or library gradient, filter, and source-over semantics.
- PowerPoint uses DrawingML primitives rendered by a specific Office viewer.

Optimizing a Gaussian proxy is therefore not sufficient evidence that the
emitted artifact improved. A better loss value can produce a worse browser
canvas, SVG rasterization, or PowerPoint slide. More splats can also widen the
train-to-deploy gap when the export primitive does not match the trained
primitive.

The full-corpus work also showed that SSIM alone is not a safe objective.
Smoothing can raise SSIM while damaging foreground detail, edges, local
structure, or perceptual similarity. Fidelity decisions need a guarded metric
vector and fixed local regions, not one global score.

## Evidence behind the decision

The current reference corpus contains 21 complete images at a maximum edge of
roughly 384 px. Seed-0 results are measured from deployed artifacts: the exact
Chrome canvas pixel buffer, rasterized emitted SVG, and Microsoft PowerPoint
slideshow captures.

| Deployed artifact | Budget | Median final splats | Median SSIM | Median LPIPS | Median size | Median training |
|---|---:|---:|---:|---:|---:|---:|
| Canvas HTML | requested 2k | 1,395 | 0.7751 | 0.2443 | 226 KB | 3.6 min |
| Canvas HTML | effective 4k | 2,382 | 0.8406 | 0.1612 | 391 KB | 9.9 min |
| SVG | requested 2k | 1,389 | 0.6022 | 0.4002 | 765 KB | 4.2 min |
| PowerPoint | requested 2k | 1,374 | 0.6091 | 0.3843 | 127 KB | 6.6 min |

All 21 Canvas images improved in both SSIM and LPIPS from requested 2k to
effective 4k, but none reached 0.99 SSIM. Chameleon reached 0.9631 only at an
effective 8k point. These results support scaling when the measured curve
justifies it; they do not support a general near-0.99 promise at small budgets.

The experiments also bound what may be promoted:

- normalized Top-K teacher/student distillation showed positive median SVG
  SSIM in some low-budget screens, but repeated image-level, edge, perceptual,
  and size regressions; it is not a default converter stage;
- residual native paths improved the selected SVG candidate on 18 of 21
  images, but the translated PowerPoint candidates did not preserve the SVG
  SSIM gain; mixed primitives remain target-specific experimental work; and
- the combined Chameleon and Coffee portfolio demonstrated that artifact
  search can recover better candidates, but also that the winning population,
  recipe, and primitive family are image- and target-dependent.

## Decision

### 1. The deployed artifact decides

The canonical post-training flow is:

```text
ordinary fitted baseline
    -> deterministic candidate proposal
    -> optional cheap proxy rejection
    -> emit candidate through the real export path
    -> render with the target compositor
    -> guarded metric comparison
    -> accept or revert
    -> record decision and provenance
```

Candidate operators propose changes. They do not own evaluation thresholds and
cannot redefine success in their favor.

The untouched ordinary output is always retained until another candidate
passes. A failure, unavailable renderer, time limit, or exhausted search budget
must return the best already accepted candidate, never an unverified partial
state.

### 2. Evidence levels are explicit

Not every target can be evaluated at the same cost during conversion. Reports
must label the evidence used:

| Evidence level | Meaning | Allowed use |
|---|---|---|
| Deployed artifact | The emitted file is rendered by its target runtime or viewer | Final acceptance and release claims |
| Parity-verified deployed model | The same equations and ordering as the shipped runtime, with separate artifact parity checks | In-run safety and checkpoint selection |
| Proxy | An approximation used for training or cheap rejection | Diagnostics and early rejection only |

A proxy-only result may not be described as a rendered SVG, browser Canvas, or
PowerPoint result.

Current target handling is:

| Target | In-converter gate | Deployed-artifact verification |
|---|---|---|
| Canvas | Exact NumPy counterpart of the emitted Canvas runtime | Chrome reads the emitted canvas pixel buffer in the corpus harness |
| SVG | Emit the candidate SVG and rasterize it; proxy fallback is rejected | `rsvg-convert` or CairoSVG, with renderer identity recorded |
| PowerPoint | No ADR-003 in-converter fidelity stage yet | Offline tooling captures the slideshow in Microsoft PowerPoint |

PowerPoint capture is authoritative but currently too external and expensive
to run inside every conversion. Internal PowerPoint previews remain proxies.

### 3. Use a guarded metric vector

The implemented fidelity evaluator records:

- LPIPS and salient-region LPIPS when the optional dependency is available;
- display-sRGB SSIM and PSNR;
- multiscale windowed luma SSIM;
- mean and p95 OKLab error;
- edge chamfer and edge-gradient error;
- the worst error over fixed residual ROIs;
- splat count, emitted bytes, and renderer identity.

ROIs are selected from the baseline residual and then frozen. A candidate
cannot improve its result by moving the measurement region.

Hard regression gates compare against the original pre-fidelity baseline.
Meaningful gains compare against the current incumbent. This permits several
small accepted improvements without allowing cumulative drift beyond the
baseline contract.

The current SVG defaults reject:

- proxy-only renders;
- candidates beyond the configured file-size limit;
- any added splat when the added-splat budget is zero;
- SSIM, edge, or worst-ROI regressions beyond their configured tolerance; and
- candidates without a sufficient LPIPS, salient-LPIPS, or OKLab-p95 gain.

These numerical defaults are benchmark parameters, not universal perceptual
constants. Repeat-render noise floors are now calibrated for the current
renderer versions. Promotion thresholds remain policy parameters, and
cross-version calibration is still required when a renderer changes.

### 4. Pure vector output remains the default contract

SVG and PowerPoint outputs remain native, editable shapes unless the user
explicitly selects a future hybrid mode. A bitmap must never be embedded
silently to make a fidelity number look better.

Native residual paths are mixed vector primitives, not hybrid raster
residuals. They have their own visual-plausibility and target-renderer gates.

### 5. Claims are full-frame and corpus-scoped

A crop, foreground window, or selected image may diagnose a mechanism but may
not stand in for the complete image or corpus.

Promotion evidence must state:

- source set and whether every full frame was scored;
- budget, initialized population, peak population, and final population;
- seed count and optimization schedule;
- artifact renderer and capture method;
- whole-frame and fixed-region metrics;
- artifact bytes, training time, and deployed render time; and
- failures and reverted candidates, not only winners.

No output mode in the current corpus supports a general 0.99 SSIM claim.

## Current implementation

### Released control-plane pieces

The following are implemented and are part of the accepted architecture:

- a typed `FidelityConfig` and `off`, `balanced`, and `max` modes;
- a bounded `FidelityStage` with baseline-versus-incumbent acceptance;
- deterministic fixed-ROI residual analysis;
- the guarded metric vector;
- emitted-and-rasterized SVG candidate evaluation;
- rejection when only an SVG proxy is available;
- JSON decision traces, baseline/final metrics, and manifest provenance;
- byte-identical fallback when no candidate is accepted;
- Canvas stage-checkpoint selection against the deployed runtime model;
- a Canvas postprocess gate that reverts destructive pruning; and
- default-off Canvas early stopping after at least two observed checkpoints
  meet an explicit absolute quality target.

The Canvas gates are separate from the `--fidelity-stage` CLI stage. The
monotonic checkpoint and postprocess gates are enabled by default; adaptive
hard-target stopping remains default-off.

### Current CLI contract

`--fidelity-stage` defaults to `off` and currently applies only to SVG:

| Mode | Current behavior | Status |
|---|---|---|
| `off` | Skip the ADR-003 post-optimization stage | Released default |
| `balanced` | Run the evaluator/reporting shell with no proposal operators | Implemented no-op scaffold |
| `max` | Try bounded single-splat recolor proposals, with zero added splats by default | Implemented opt-in; not a broad optimizer |

For Canvas or PowerPoint, selecting this flag records an unsupported-target
reason rather than pretending that SVG evaluation applies.

`max` does not currently move, resize, rotate, split, merge, reorder, or add
splats. It does not search export recipes. Its only proposal operator is
bounded recoloring of a strong splat in a fixed high-error ROI.

Canvas has a separate `--adaptive-compute` switch. It is off by default,
Canvas-only, and currently stops only when the best observed deployed-model
checkpoint reaches `--adaptive-target-ssim-srgb` after at least
`--adaptive-min-checkpoints` stages. The user target denotes desired Chrome
artifact quality. The scorer reproduces JavaScript double math, Float32Array
accumulation, and the final 8-bit sRGB ImageData buffer. It does not stop on
plateau or regression and does not predict an unseen higher-budget result.

### Implemented experiments, not release paths

The repository also contains working MVP code for:

- normalized Top-K teacher/student training in Torch and MLX;
- residual native SVG paths and editable DrawingML segments;
- a combined population, post-fit, recipe, recolor, and residual-path
  portfolio runner;
- full-corpus actual-artifact scoring; and
- real Microsoft PowerPoint slideshow capture.

These tools generate evidence. They are not implicitly part of `splatlify`,
and their presence does not make their algorithms production defaults.

## Roadmap

The roadmap closes cheap evidence and allocation questions before making the
search space larger. The hard-target allocation question is now closed as a
no-go; the next active slice is the broader deterministic operator portfolio.

### A. Calibrate artifact gates

**Status:** Baseline implemented; cross-version calibration remains

1. Measure repeat-render noise floors for Chrome, each supported SVG
   rasterizer, and Microsoft PowerPoint.
2. Separate deterministic conversion variance from viewer capture variance.
3. Calibrate hard regression and meaningful-gain thresholds per target.
4. Add cross-rasterizer SVG checks where a candidate is sensitive to filter or
   color semantics.

The first full-corpus run is complete: every Canvas, SVG, and PowerPoint
artifact was captured five times. Canvas and SVG were pixel-deterministic.
The largest additional PowerPoint warm-up SSIM span was below `0.000001`, far
below the current policy gates. The versioned result is
`data/artifact-gates.json`; methodology and results are documented in
`docs/artifact-gates-and-adaptive-compute.md`.

Remaining exit gate: repeat the calibration when target renderer versions
change and add cross-rasterizer SVG checks for sensitive candidates.

### B. Adaptive compute and selective scaling

**Status:** Bounded online slice implemented; hard-target expansion rejected;
predictive allocation and selective scaling remain proposed

The existing Canvas checkpoint gate can now stop before densification, later
stages, and residual detail when an absolute observed quality target is met.
Every decision records its policy, observed checkpoints, selected checkpoint,
skipped stages and iterations, requested Chrome target, runtime-scorer identity,
effective in-process threshold, and that no future evidence was used.

The runtime scorer is calibrated against Chrome on 48 compatible historical
full-frame checkpoints: four stages for each of 12 images. Every framebuffer
matched byte-for-byte. The previous continuous scorer overstated SSIM by as
much as `0.001102`; reproducing the deployed 8-bit boundary eliminates that
gap, so the calibrated default safety margin is zero. The other nine images do
not have matching historical Chrome captures, but their raw checkpoints do
exist.

All 84 raw checkpoints across all 21 images were therefore rescored with that
byte-exact model. Targets `0.98` and `0.979` both stopped only Brick and
Colorwheel, saved 162.3 of 12,841.7 recorded stage seconds (1.3%), and had zero
observed SSIM and PSNR opportunity cost on those two curves. This misses the
predeclared 5% compute gate, so the present hard-target controller remains
default-off and will not receive fresh multi-seed A/B expansion. The governing
record is `data/adaptive-exact-replay.json`.

This is not the broader controller that decides from a measured quality
slope whether to continue, densify, or move to an effective 4k or 8k budget.
That future decision should also use predicted training time, tile overlap,
browser time, and artifact bytes. It should be revisited only if a richer safe
signal clears the retrospective 5% compute gate before expensive variance
testing.

The controller must distinguish requested splats from initialized, peak, and
final populations. A nominal 4k cap with a 1,200 initialization ceiling is not
an effective 4k experiment.

Exit gate: on the full corpus, adaptive allocation matches or beats the fixed
budget quality envelope while using less aggregate compute, with no protected
image regression beyond calibrated noise.

### C. Broader operator portfolio

**Status:** Proposed, except bounded recolor

Add one deterministic operator at a time:

1. smaller recolor and alpha adjustment;
2. bounded center movement;
3. scale and rotation adjustment;
4. split and merge;
5. local draw-order repair;
6. target-specific recipe or primitive substitution; and
7. visually constrained residual native paths.

Every operator must have its own budget, invariants, unit tests, artifact
ablation, and accept-or-revert trace. Operators may not be enabled as a bundle
until each one has shown an independent corpus benefit.

Exit gate: positive actual-artifact benefit on the complete corpus and selected
multi-seed reruns, without material per-image, ROI, edge, size, or runtime
regression.

### D. Full artifact search

**Status:** Proposed; bounded MVP exists

Promote the combined-portfolio idea into a reusable, bounded search over:

- accepted training checkpoints or populations;
- target-specific post-fit variants;
- export recipes and safe optimizers;
- operator sequences;
- mixed native primitive candidates; and
- explicit quality, size, shape-count, and time budgets.

The search should retain a small lineage-diverse beam rather than selecting a
single proxy winner too early. Cheap proxies may prune obvious losers, but the
final rank must use the deployed artifact.

SVG, Canvas, and PowerPoint require separate winners. A candidate selected by
SVG may not be translated to DrawingML and called the PowerPoint winner
without an actual PowerPoint gate.

Exit gate: a deterministic converter integration with bounded cost, atomic
fallback, complete decision traces, and target-specific corpus wins.

### E. Optional hybrid residual atlas

**Status:** Proposed

Only after the pure-vector rate-distortion curve is measured, test an explicit
hybrid mode that stores a sparse raster correction for residuals that are
inefficient to express as native shapes.

Requirements:

- opt-in mode and visible manifest disclosure;
- bounded residual area, bytes, and opacity;
- deterministic atlas packing;
- no bitmap in the pure-vector modes;
- comparisons against an ordinary PNG or JPEG at matched bytes; and
- actual SVG and PowerPoint rendering tests.

Exit gate: a material fidelity-per-byte advantage over both pure vector and a
normal matched-size bitmap for a documented use case.

### F. Learned proposal policy

**Status:** Deferred

Only after the deterministic operators and full artifact search have produced
enough accepted and rejected traces should the project consider learning which
operator, region, recipe, or budget to try next. The learned component may
rank proposals; it may not bypass the artifact gate.

## Promotion and release gates

A proposed fidelity feature may enter the normal converter only when:

1. tests cover determinism, bounds, failure fallback, and output validity;
2. a no-op or rejected run preserves the baseline output;
3. every quality claim comes from a labeled deployed artifact or
   parity-verified deployed model;
4. the full 21-image frames are evaluated at seed 0;
5. the best configuration is repeated across at least three seeds where
   stochasticity matters;
6. per-image results accompany medians and threshold counts;
7. protected whole-frame, worst-ROI, edge, perceptual, and color metrics hold;
8. shape count, bytes, training time, and render time are reported;
9. target-specific viewer behavior is tested rather than inferred from another
   format; and
10. the feature is default-off until its corpus gates pass.

Mixed primitives additionally require inspection at normal scale and enlarged
scale so disconnected or implausible marks cannot win only through aggregate
metrics.

## Consequences

### Positive

- Fidelity work becomes monotonic within stated tolerances.
- Proxy improvements cannot silently replace a better deployed artifact.
- Canvas, SVG, and PowerPoint can evolve independently where their compositors
  require different solutions.
- Failed experiments remain useful because the decision traces expose which
  metric, region, size, or target gate rejected them.
- Future operators can reuse one evaluation and provenance contract.

### Negative

- Actual-artifact evaluation is slower than proxy-only optimization.
- PowerPoint verification requires an installed external viewer and cannot yet
  be a cheap in-process loop.
- A metric vector and fixed ROIs make tuning more complex than maximizing one
  score.
- Keeping the ordinary baseline and candidate artifacts increases temporary
  disk use.

### Risks and mitigations

- **Metric gaming:** use hard protected metrics, fixed ROIs, and visual
  inspection for new primitive families.
- **Renderer overfitting:** identify the renderer and run cross-renderer checks
  for sensitive SVG changes.
- **Search explosion:** bound operators, candidates, passes, beam width, and
  wall time; implement adaptive allocation before broad search.
- **Cumulative drift:** keep hard floors anchored to the original baseline,
  not only the latest incumbent.
- **Hidden rasterization:** keep hybrid residuals explicit and opt-in.
- **Misleading aggregate wins:** publish per-image results and reverted cases.

## Non-goals

- promising universal or paper-level 0.99 fidelity;
- treating a selected crop as a whole-image result;
- forcing one splat population or export recipe to win for every target;
- replacing real SVGs or native PowerPoint shapes with undisclosed PNGs;
- enabling every experimental algorithm merely because an MVP exists; or
- making Microsoft PowerPoint automation a mandatory runtime dependency.

## Implementation map

- Fidelity stage: `src/png2svg_gs/fidelity/`
- Converter integration and Canvas gates: `src/png2svg_gs/converter.py`
- Top-K student/teacher experiment: `src/png2svg_gs/distillation.py`
- Mixed native primitives: `src/png2svg_gs/mixed_primitives.py`
- Corpus and capture tooling: `tools/corpus_benchmark.py`
- Combined artifact portfolio: `tools/combined_portfolio_mvp.py`
- Versioned artifact gates: `data/artifact-gates.json`
- Repeat-render calibration: `src/png2svg_gs/artifact_gates.py` and
  `tools/calibrate_artifact_noise.py`
- Online and retrospective adaptive compute:
  `src/png2svg_gs/adaptive_compute.py`, `src/png2svg_gs/converter.py`, and
  `tools/simulate_adaptive_canvas.py`
- Online adaptive MVP evidence: `data/adaptive-online-mvp.json`
- Exact full-corpus hard-target replay: `data/adaptive-exact-replay.json`
- Canvas checkpoint parity calibration: `src/png2svg_gs/canvas_parity.py`,
  `tools/calibrate_canvas_checkpoint_parity.py`, and
  `data/canvas-checkpoint-parity.json`
- Fidelity tests: `tests/unit/test_fidelity_stage.py`
- Canvas gate tests: `tests/unit/test_png2svg_export_pipeline.py`

## Current checklist

- [x] Establish a full-frame, deployed-artifact corpus baseline.
- [x] Implement the bounded accept-or-revert stage contract.
- [x] Implement actual emitted-SVG evaluation with honest proxy fallback.
- [x] Freeze residual ROIs and use a guarded metric vector.
- [x] Record candidate decisions and manifest provenance.
- [x] Add default-on monotonic Canvas checkpoint and postprocess gates.
- [x] Test Top-K distillation on full frames and keep the losing configuration
      out of the default converter.
- [x] Test mixed native paths against SVG and real PowerPoint and keep the
      target-inconsistent result experimental.
- [x] Calibrate current Chrome, `rsvg-convert`, and PowerPoint repeat-render
      noise floors over the full corpus.
- [x] Implement default-off Canvas hard-target stopping before densification
      and residual detail.
- [x] Calibrate the Canvas checkpoint scorer against 48 unchanged full-frame
      Chrome captures and reproduce every deployed framebuffer byte-for-byte.
- [x] Replay the exact hard-target policy over all 84 raw checkpoints and stop
      expansion after its 1.3% saving missed the 5% compute gate.
- [ ] Validate predictive adaptive compute and selective 8k allocation over the
      full corpus and multiple seeds, only after a richer policy clears the
      retrospective compute gate.
- [ ] Expand the operator portfolio beyond bounded recolor.
- [ ] Integrate deterministic, bounded full artifact search.
- [ ] Decide on an explicit hybrid residual mode from matched-byte evidence.
- [ ] Add an in-converter PowerPoint artifact gate if a robust automation
      boundary becomes practical.
- [ ] Consider learned proposal ranking only after enough traces exist.

## References

- [ADR-002: PNG to Gaussian Splat Pipeline in Python](adr-002-png2splat-python-pipeline.md)
- [Canvas Scaling MVP](canvas-scaling-mvp.md)
- [Top-K Teacher to Native Vector Student MVP](topk-distillation-mvp.md)
- [Mixed-Primitives Fidelity MVP](mixed-primitives-mvp.md)
- [Combined Artifact-Portfolio MVP](combined-portfolio-mvp.md)
- [Artifact Gates and Adaptive Compute](artifact-gates-and-adaptive-compute.md)
- [SVG and PowerPoint Gaussian Tricks](SVG_PPTX_GAUSSIAN_TRICKS.md)
- [Provenance and Benchmarks](PROVENANCE_AND_BENCHMARKS.md)
