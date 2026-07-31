# Artifact Gates and Adaptive Compute

**Status:** calibration and bounded online controller implemented; hard-target
expansion rejected by the corpus compute gate
**Date:** 2026-07-31

## What shipped

The ADR-003 measurement foundation now has:

- one shared target-specific calibration schema in
  `src/png2svg_gs/artifact_gates.py`;
- a versioned current calibration in `data/artifact-gates.json`;
- repeated pixel-runtime, SVG, and PowerPoint capture in
  `tools/calibrate_artifact_noise.py`;
- optional retention of every Chrome canvas repeat in
  `tools/capture_canvas_html.py`;
- a reusable retrospective policy model in
  `src/png2svg_gs/adaptive_compute.py`;
- a full-corpus replay tool in `tools/simulate_adaptive_canvas.py`;
- a resumable full-frame checkpoint parity calibrator in
  `tools/calibrate_canvas_checkpoint_parity.py`;
- a default-off, pixel-runtime-only online hard-target controller in the converter;
  and
- content-addressed adaptive options in `tools/corpus_benchmark.py`, with the
  first deployed-artifact evidence versioned in
  `data/adaptive-online-mvp.json`; and
- exact full-corpus replay evidence in `data/adaptive-exact-replay.json`.

Calibration is a lower bound on meaningful deltas. Algorithm-specific policy
thresholds remain in force when they are stricter.

## Calibration contract

Noise is measured only within repeated renders of one unchanged artifact.
Differences between images, seeds, algorithms, or splat budgets are never
classified as renderer noise.

Each observation records:

- target and artifact identity plus hashes;
- source and capture paths plus hashes;
- repeat index;
- renderer and renderer version;
- capture method and render duration; and
- the full guarded metric vector.

The recommended minimum delta is the greater of twice the cross-artifact p95
repeat span and the largest observed span. The maximum term prevents a rare
but real viewer variation from disappearing when most artifacts are perfectly
deterministic.

## Full-corpus calibration result

The full 21-image corpus was captured five times per target. An additional
five-capture PowerPoint warm-up session is included conservatively because one
of those captures differed by a few pixel values.

| Target | Artifacts | Main observations | Repeat result |
|---|---:|---:|---|
| Chrome pixel runtime (`ImageData` via Canvas) | 21 | 105 | Byte-identical within every artifact |
| Playwright Chromium SVG | 21 | 105 | Byte-identical within every artifact |
| Microsoft PowerPoint | 21 | 105 | Byte-identical within every artifact |

The extra PowerPoint warm-up produced these largest observed metric spans:

| Metric | Maximum observed span |
|---|---:|
| SSIM | 0.0000007173 |
| MS-SSIM | 0.0000005739 |
| LPIPS | 0.0000002682 |
| PSNR | 0.000008363 dB |
| Edge-gradient L1 | 0.0000004340 |

These floors are orders of magnitude below the existing algorithm gates. The
current `0.0005` pixel-runtime SSIM checkpoint tolerance and `0.002` SVG SSIM
regression tolerance therefore remain policy choices, not accommodations for
repeat-render noise.

This run does not establish cross-version or cross-machine stability. The
recorded environment is Chrome pixel-runtime canvas capture, Playwright Chromium
140.0.7339.81, PowerPoint 16.89.1, and macOS 26.5.2 on arm64.

## Adaptive pixel-runtime replay

The simulator consumes two evidence sets and keeps them separate:

1. canonical raw pixel-runtime stage checkpoints, rescored with the byte-exact
   deployed runtime model;
2. final 2k and 4k pixel buffers captured from Chrome.

Raw artifacts exist for all four checkpoints of all 21 current 4k curves. The
simulator no longer depends on the old continuous scores stored in only 12
manifests. It exactly rescored all 84 raw artifacts and applied the online
controller's real contract: at least two observed checkpoints, an absolute
SSIM target, and no plateau, regression, or future-evidence stop.

Targets `0.98` and `0.979` produced the same result:

- 2 early stops over 21 images: Colorwheel after stage 2 and Brick before the
  residual pass;
- 162.3 seconds saved from 12,841.7 recorded stage seconds, or **1.3%**;
- zero SSIM and PSNR opportunity cost on both stopped observed curves; and
- a `do-not-expand` verdict because 1.3% is below the predeclared 5% aggregate
  compute gate.

This supersedes the earlier 12-curve, 2.1% replay. The exact result is
versioned in `data/adaptive-exact-replay.json`. It does not justify fresh
multi-seed A/B expansion of this hard-target policy or making the controller
default-on.

The 2k-to-4k retrospective oracle covers all 21 images. With both a 0.95 SSIM
and 0.15 LPIPS target, it:

- continues 19 images to 4k;
- stops Brick and Colorwheel at 2k;
- saves 774.7 seconds, or 6.1% versus fixed 4k;
- has mean SSIM opportunity cost 0.00080; and
- has maximum SSIM opportunity cost 0.00858.

This second result uses the already-known 4k outcome. It describes available
rate-distortion headroom and is **not** evidence that a predictor can make the
same choice from the 2k state.

## Online hard-target controller

`--adaptive-compute` evaluates the deployed pixel-runtime model after each completed
main stage, at the existing monotonic checkpoint boundary. After at least two
checkpoints, it stops only when the selected checkpoint reaches the explicit
SSIM target. The stop occurs before the next densification step and skips all
later stages plus residual detail.

The controller deliberately does not:

- stop on a plateau or one regressing checkpoint;
- infer an unseen 4k or 8k result;
- use a future checkpoint to justify a decision; or
- run for native Canvas, CSS, SVG, or PowerPoint.

The run manifest records the policy, every observed decision, selected
checkpoint, skipped stage count, skipped scheduled iterations, and whether
residual detail was skipped.

### Fresh deployed-artifact MVP

The first hard-target implementation was run end to end on Colorwheel and
Brick with seed 0, MLX, the max-fidelity profile, a genuine 4k initial cap, and
Chrome `canvas.toDataURL` scoring. These runs predate the interim `0.0012`
continuous-to-browser safety margin and the later byte-exact runtime scorer,
so the table is historical mechanism evidence rather than a validation of the
current threshold. Each arm is one run and is not a stable performance
estimate.

| Image and policy | Stop | Wall time | Versus fixed | Chrome SSIM | Chrome LPIPS |
|---|---|---:|---:|---:|---:|
| Colorwheel fixed | none | 614.5 s | — | 0.983286 | 0.006373 |
| Colorwheel target 0.98 | none; target missed | 777.3 s | +26.5% | 0.979540 | 0.007975 |
| Brick fixed | none | 429.9 s | — | 0.979724 | 0.030867 |
| Brick target 0.98 | after stage 3; residual only | 447.1 s | +4.0% | 0.980027 | 0.030733 |
| Colorwheel target 0.979 | after stage 2 | 449.1 s | **-26.9%** | 0.983924 | 0.006738 |

The 0.979 run skipped the final 250 scheduled iterations and residual detail.
Its deployed-artifact metrics are mixed relative to fixed: SSIM and worst-ROI
error improved slightly, while LPIPS, MS-SSIM, PSNR, OKLab p95, and
edge-gradient error moved slightly in the other direction. The runs also
showed substantial seeded MLX variation: the two Colorwheel 0.98 arms followed
different checkpoint curves even before any stop occurred. Sequential wall
time was additionally affected by likely thermal variation.

The result therefore demonstrates a real speed mechanism and an explicit
quality/speed control, not a general 27% speedup. The later exact full-corpus
replay is the governing speed result: both tested targets saved only 1.3% of
aggregate recorded stage time.

## Checkpoint model-to-Chrome calibration

The parity calibrator reconstructed unchanged, self-contained pixel-runtime
HTML for every available full-frame stage artifact and captured its exact
canvas pixel buffer with Chrome. It compared both the former continuous NumPy renderer and
the deployed runtime scorer with those pixels. No checkpoint was retrained, so
optimizer variance is outside this measurement.

The browser-parity capture set covers 12 images and four checkpoints each: 48
full-frame comparisons with zero capture failures. Nine corpus images do not
yet have matching historical Chrome captures. Their raw stage artifacts are
nevertheless present and were included in the separate 84-checkpoint exact
policy replay above.

The old scorer retained continuous float pixels. Chrome instead computes
Gaussian terms as JavaScript doubles, rounds accumulator writes through
`Float32Array`, then packs the final display-space result into 8-bit
`ImageData`. Reproducing those three boundaries removes the apparent browser
gap:

| Scorer and quantity | Median | p95 | Worst relevant value |
|---|---:|---:|---:|
| Old continuous SSIM overstatement | +0.000468 | +0.001064 | +0.001102 |
| Old continuous direct parity SSIM | 0.999344 | 0.999869 | minimum 0.998810 |
| Byte-exact SSIM overstatement | 0 | 0 | 0 |
| Byte-exact direct parity SSIM | 1.0 | 1.0 | minimum 1.0 |
| Byte-exact pixel MAE | 0 | 0 | maximum 0 |

The online controller therefore interprets its user target as a desired
Chrome artifact score and evaluates the exact deployed framebuffer. All 48
calibration checkpoints matched Chrome pixel-for-pixel, so the calibrated
default margin is zero: target `0.98` means an effective threshold of `0.98`.

The requested target, runtime-scorer identity, pixel-exact calibration count,
optional cross-version margins, and effective thresholds are all recorded in
the run manifest. A future browser/runtime change still requires
recalibration; advanced nonzero margin overrides remain available for that
case. The versioned result is
`data/canvas-checkpoint-parity.json`; the full local capture record is written
to `./tmp/canvas-checkpoint-parity/summary.json`.

## Reproduction

Capture and calibrate the complete corpus:

```bash
PYTHONPATH=src venv/bin/python tools/calibrate_artifact_noise.py \
  --capture --targets pixel-runtime,svg,pptx --repeats 5 \
  --output-dir ./tmp/artifact-noise
```

PowerPoint capture opens the real application and controls its slideshow UI.
Pixel-runtime capture requires a Python environment containing Playwright.

Calibrate the available pixel-runtime checkpoint curves against Chrome:

```bash
PYTHONPATH=src venv/bin/python tools/calibrate_canvas_checkpoint_parity.py \
  --corpus-root result/corpus \
  --output-dir ./tmp/canvas-checkpoint-parity
```

Both tools use the current interpreter by default. Install the `capture` extra
in SplatThis's own virtual environment. An explicit interpreter override is
kept as a virtual-environment symlink rather than canonicalized to a system
Python, because resolution would discard that environment's Playwright package.

Replay the current adaptive policies:

```bash
PYTHONPATH=src venv/bin/python tools/simulate_adaptive_canvas.py \
  --corpus-root result/corpus \
  --artifact-gates data/artifact-gates.json \
  --target-ssim 0.98 --compare-target-ssim 0.979 \
  --minimum-useful-saving-fraction 0.05 \
  --output-dir ./tmp/adaptive-canvas-simulation
```

These calibration and replay tools write machine-readable JSON plus a
Markdown report.

Run a fresh default-off online experiment:

```bash
PYTHONPATH=src venv/bin/python tools/corpus_benchmark.py \
  --root ./tmp/adaptive-online-mvp --materialize

PYTHONPATH=src venv/bin/python tools/corpus_benchmark.py \
  --root ./tmp/adaptive-online-mvp --run --formats pixel-runtime --seeds 0 \
  --splats 4000 --initial-splat-cap 4000 --only colorwheel,brick \
  --profile max-fidelity --optimizer-backend mlx \
  --canvas-capture-python venv/bin/python \
  --adaptive-compute --adaptive-target-ssim-srgb 0.98
```

## Next implementation gate

Do not make adaptive stopping default-on or spend a fresh multi-seed benchmark
on the current hard-target rule. Its 1.3% retrospective saving misses the 5%
compute gate before optimizer-variance validation is even charged.

The first target-specific follow-up is now complete. A full-corpus
native-dimension Playwright Chromium gate accepted palette-quantized on Brick,
Cell, Chameleon, Hubble deep field, Immunohistochemistry, Retina, and Rocket.
Its 7/21 wins clear the predeclared minimum of five; automatic browser recipe
selection is eligible for a separate default-off integration slice but is not
yet in `splatlify`. See `docs/svg-recipe-gate-mvp.md` and
`data/svg-recipe-gate-mvp.json`.

That selector integration is the next slice. Bounded center adjustment remains
the next independent geometry operator. Revisit adaptive allocation only when
a richer safe signal, such as a calibrated perceptual proxy or quality-slope
model, clears the same retrospective compute gate. Only then measure same-seed
MLX variation and run an interleaved, multi-seed adaptive-versus-fixed
benchmark.
