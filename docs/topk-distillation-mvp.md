# Top-K Teacher to Native Vector Student MVP

**Status:** experiment, not a production output mode\
**Targets:** browser-rendered SVG and real Microsoft PowerPoint
**Default:** disabled

## Question

Can a paper-style normalized top-K Gaussian renderer provide a better geometric
and color solution, then be distilled into ordinary source-over shapes that SVG
and DrawingML can render?

The MVP must answer this before the converter gains another permanent pipeline
stage.

## Representations

The experiment compares three arms with the same initialization, splat budget,
resolution, seed, and total optimization iterations:

1. **Direct student:** the current export-target-aware alpha-over training.
2. **Teacher ceiling:** normalized top-K rendering with `K=10`.
3. **Distilled student:** initialize from the trained teacher, switch to the
   target's alpha-over proxy, and optimize against a combined source/teacher
   objective before the existing artifact post-fit.

The normalized teacher uses, per pixel:

```text
S(x) = indices of the K strongest Gaussian responses G_i(x)
C(x) = sum(i in S(x), G_i(x) * color_i) / sum(i in S(x), G_i(x))
```

It ignores alpha and drawing order. The exported student continues to use the
current source-over equation and therefore remains representable as native SVG
or DrawingML shapes.

## MVP slices

### Slice 1: renderer proof

- Add an isolated Torch `normalized-topk` render mode.
- Pin `K=10` by default, while tests may use smaller K values.
- Verify the per-pixel equation, order and alpha invariance, finite gradients,
  and parity between reference and batched Torch renderers.
- Do not expose the mode as an SVG/PPTX export promise.

### Slice 2: controlled distillation benchmark

Add an experimental runner that:

1. builds one shared initialization;
2. trains the direct alpha-over arm;
3. trains the normalized teacher;
4. copies teacher geometry and color into the student;
5. initializes the student with a coverage-safe handoff: teacher geometry and
   color, scales no smaller than the shared initialization, and full native
   opacity before target-aware adaptation;
6. compares that full handoff with a color-only handoff that retains the
   initialization's coverage geometry;
7. optionally constrains teacher geometry with an auxiliary opaque alpha-over
   exportability loss;
8. trains the student with teacher guidance decaying from 25% to zero;
9. applies the same existing SVG proxy post-fit to the direct and distilled
   arms so exporter mismatch is not mistaken for a representation gain;
10. emits the actual SVG and captures it at native dimensions in Chromium;
11. emits PPTX and captures a real PowerPoint slide image;
12. writes one comparison record and visual panel per source image.

The teacher term is guidance, not the acceptance target. A student is successful
only when the real deployed artifact improves against the original source.

### Slice 3: native-vector refinements

Only after Slice 2 succeeds:

- use teacher responsibility maps to initialize local drawing order and alpha;
- allow residual-driven split/prune operations during distillation;
- compare Gaussian-only output with a bounded mixed-primitives arm;
- add embedded-SVG-in-PPTX as a separate fidelity product, not as native
  DrawingML editability.

## Original staging plan

Start with six deliberately different images:

- `logo`: flat regions and hard edges;
- `text`: thin high-contrast structure;
- `chameleon`: salient subject and complex color;
- `rocket`: dark background and small bright structures;
- `coffee`: photographic texture;
- `checkerboard`: adversarial high-frequency geometry.

Run at 128 px maximum edge first, with 256 and 1,024 splats. Use three fixed
seeds. Promote to the full corpus only if the small run is directionally
positive.

The initial bounded test was not conclusive because it used only three images
at 64 px. The implementation was therefore taken through a full-frame,
full-corpus run before making a pipeline decision.

## Measurements

Record separately for teacher, student proxy, actual SVG, and actual PPTX:

- SSIM and multiscale/local SSIM;
- PSNR;
- LPIPS when available;
- edge error and worst fixed-ROI error;
- splat/shape count;
- compressed and uncompressed artifact bytes;
- training, export, and deployed-render time;
- teacher-to-student imitation error.

Never label an internal proxy as an SVG or PowerPoint render.

## MVP decision gates

Proceed to converter integration only if, against the direct student:

- median actual-artifact SSIM improves by at least `0.01`;
- at least four of six images improve for both actual SVG and actual PPTX;
- no image regresses by more than `0.005` SSIM;
- worst-ROI and edge metrics do not regress;
- shape count is identical and file size grows by no more than 5%;
- the gain remains positive across at least two of three seeds.

If the teacher improves strongly but the student does not, the bottleneck is
representational and the next experiment is mixed primitives or embedded SVG.
If the teacher itself does not improve, stop: top-K distillation is not the
answer for this pipeline and budget.

## Bounded result

At 64 px, 64 splats, seed 0, and 40 teacher plus 40 student iterations:

| Variant | Logo SVG delta | Chameleon SVG delta | Rocket SVG delta |
|---|---:|---:|---:|
| full handoff, decayed guide | +0.0051 | -0.0087 | -0.0416 |
| color-only handoff | -0.0489 | +0.0054 | -0.0035 |
| full plus 0.2 exportability | +0.0091 | -0.0129 | -0.0448 |
| full plus equal 40-iteration SVG post-fit | -0.0009 | +0.0098 | -0.0643 |

No variant clears the predefined deployed-SVG gate across the small corpus.
Real PowerPoint capture therefore remains deferred. The runner records an
explicit accept-or-revert decision and requires a positive teacher ceiling,
at least +0.01 actual SVG SSIM, and no material PSNR regression before a
student is promoted to PowerPoint testing.

## Full-corpus low-budget screen

The follow-up used all 21 corpus images at their stored corpus dimensions (up
to 384 px), not crops. Deployed direct baselines were the existing MLX-trained
artifacts. The experimental Top-K teacher/student arms ran in Torch on MPS with
the tiled batched renderer. Each arm used 20 teacher plus 20 student
iterations. The 1,024-splat arm ran seeds 0, 1, and 2; the 256-splat arm ran
seed 0. The recorded SVG metrics in the table below were produced by
`rsvg-convert` before Chromium became the governing browser target. They are
retained as historical evidence and are not sufficient for promotion. The
runner now uses native-dimension Playwright Chromium capture; this experiment
must be rerun before making a new browser-SVG decision.

This is a screening configuration, not a paper-level quality ceiling. The
deployed corpus contains 959-1,714 splats after pruning (median 1,389) from a
2,000-splat cap, so the 1,024-splat experiment is below the deployed median.
It is also heavily under-optimized: the deployed maximum-fidelity schedule
uses roughly 1,800 optimization iterations plus residual fitting, compared
with only 40 total direct-arm iterations here.

| Splats | Seed | Median ΔSSIM | SSIM improved | Median ΔMS-SSIM | Median ΔLPIPS | LPIPS improved | Median size growth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0 | +0.0030 | 12/21 | +0.0119 | -0.0005 | 12/21 | +6.2 KB |
| 1,024 | 0 | +0.0112 | 14/21 | +0.0067 | +0.0073 | 9/21 | +37.3 KB |
| 1,024 | 1 | +0.0085 | 13/21 | +0.0174 | -0.0044 | 11/21 | +36.6 KB |
| 1,024 | 2 | +0.0124 | 12/21 | +0.0111 | +0.0008 | 10/21 | +36.5 KB |

The median can look positive while the method is unsafe. At 1,024 splats,
`moon` regressed by roughly 0.10 SSIM on every seed, `cell` by roughly 0.02,
and `checkerboard`, `colorwheel`, `retina`, `rocket`, and `text` also
repeatedly regressed. Only 12 of 21 images improved SSIM on every seed, and
only nine improved both SSIM and LPIPS on every seed. Edge-gradient error
worsened on the median image in all three seeds. Package growth was far above
the 5% gate.

**Decision:** reject only this 1,024-splat, 40-iteration configuration as a
default or monotonic fidelity stage. It misses the no-regression, edge,
perceptual, and size gates even though two seeds clear the median +0.01 SSIM
target. This result does not reject Top-K at an adequate budget or at
convergence.

The next fair scaling test should keep the full 21-image frames and use:

1. 2,000 splats as the deployed-budget anchor;
2. 4,096 and 8,192 splats to measure the fidelity/size curve;
3. enough iterations for direct, teacher, and student losses to plateau,
   rather than a fixed 20-step screen;
4. seed 0 across the full corpus at every budget, followed by three seeds at
   the best budget;
5. actual SVG gates before any PowerPoint capture;
6. a 16,384-splat arm only if the 8,192 curve is still improving enough to
   justify its runtime and artifact size.

## Full-frame Chameleon MLX port

The normalized Top-K renderer and all three distillation arms now also run on
MLX. The MLX path preserves the experiment's important contracts:

- the direct and student arms use sRGB alpha-over;
- the teacher uses normalized Top-K with alpha/order invariance;
- teacher guidance decays from 25% to zero;
- the optional opaque alpha-over exportability loss is supported;
- Adam uses the same per-parameter-group learning rates;
- checkpoints are selected by source loss, not by the changing guided loss;
- every final decision still uses the rasterized SVG.

The first matched test starts from the converged 1,615-splat Chameleon SVG
checkpoint and runs 200 teacher plus 200 student iterations. The direct arm
receives the same 400 total iterations.

| Backend | Direct proxy SSIM | Teacher proxy SSIM | Student proxy SSIM | Direct SVG SSIM | Student SVG SSIM | Distillation time |
|---|---:|---:|---:|---:|---:|---:|
| Torch/MPS | 0.92635 | 0.90150 | 0.91186 | 0.73586 | 0.73256 | 56.3 min wall |
| MLX | 0.92077 | 0.90144 | 0.91262 | 0.72261 | 0.71589 | 8.1 min runner / 11.5 min wall |

The teacher and student proxy results are close across backends and lead to
the same rejection: the teacher ceiling is below the direct arm and the
student also loses on the actual SVG. The observed end-to-end wall-clock is
roughly 4.9x faster than Torch; the MLX runner's instrumented distillation
phase itself takes 8.1 minutes. The direct arm does not yet converge to the
same local solution as Torch. That optimizer-path difference must be understood
before MLX results replace Torch reference results in a corpus claim.

The MLX port therefore removes the main runtime blocker for further bounded
tests, but it does not change the Chameleon decision and is not yet a reason to
promote Top-K distillation into the production converter.
