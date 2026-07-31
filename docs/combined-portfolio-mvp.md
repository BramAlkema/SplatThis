# Combined artifact-portfolio MVP

This MVP tests the Chameleon image first. It does not merge every objective
into one optimizer loss. Instead, it keeps a small beam of exportable
candidates and lets the deployed artifact decide.

## Candidate graph

1. Train a 4k-cap SVG-target-aware MLX population with the existing
   residual-guided densify/split/prune schedule.
2. Continue that exact checkpoint with equal-budget direct alpha-over,
   normalized Top-K teacher, and exportable teacher/student arms.
3. Build a region-conditioned indexed hybrid when requested: student
   geometry/color/alpha inside the foreground mask, direct splats outside it,
   while preserving the direct expert's global draw-order slots.
4. Apply both SVG-gradient-aware and native-blur-aware color/alpha post-fit as
   separate proposals.
5. Run one bounded deployed-artifact recolor pass on each population.
6. Emit standard, browser-compatible, palette-quantized, and blur SVG
   recipes. Standard and palette outputs also get a safe precision-2 SVGO
   branch.
7. Keep a lineage-diverse beam and add residual native paths to each retained
   candidate.
8. Score every retained SVG through an actual `rsvg-convert` render. Build a
   separate native-PPTX shortlist, capture it in real Microsoft PowerPoint,
   and choose its winner independently.

The normalized Top-K teacher is intentionally not emitted: it is a
non-alpha-over optimization ceiling. Its student is the exportable proposal.
The scripted-matrix SVG is also excluded from the artifact gate because it
needs JavaScript and `rsvg-convert` cannot execute it.

## MVP gate

SSIM is not sufficient. Relative to the baseline, a candidate is rejected if
it loses more than:

- 0.002 SSIM;
- 0.003 MS-SSIM;
- 0.005 LPIPS;
- 0.5 px edge chamfer; or
- 3% worst-ROI error.

Guarded candidates are ranked with a balanced delta score over SSIM,
MS-SSIM, LPIPS, OKLab p95, edge chamfer, edge-gradient error, and worst-ROI
error. File size is only a tie-breaker. These thresholds are MVP parameters,
not corpus-calibrated constants.

## Reproduction

The combined runner is `tools/combined_portfolio_mvp.py`. Its report contains
all candidates, full-frame metrics, operations, guard decisions, exact artifact
paths, and a browser overview:

```bash
PYTHONPATH=src:. python tools/combined_portfolio_mvp.py \
  result/corpus/images/chameleon.png \
  --baseline-raw ./tmp/chameleon-combined-mvp/base4k/artifacts/final.raw.json \
  --population direct=./tmp/chameleon-combined-mvp/topk/chameleon_direct.raw.json \
  --population student=./tmp/chameleon-combined-mvp/topk/chameleon_student.raw.json \
  --population legacy-svg2k=result/corpus/runs/chameleon_svg_s0_art/final.raw.json \
  --population canvas4k=result/corpus/runs/chameleon_canvas_s0_b0bccdc1f3d2_art/final.raw.json \
  --population pptx2k=result/corpus/runs/chameleon_pptx_s0_art/final.raw.json \
  --foreground-expert student --background-expert direct \
  --hybrid-name foreground-hybrid \
  --svg-postfit-iters 40 --blur-postfit-iters 40 \
  --postfit-population base4k --postfit-population direct \
  --postfit-population student \
  --recolor --capture-powerpoint --postfit-device mps \
  --pptx-include-population pptx2k \
  --pptx-include-population student \
  --pptx-include-population foreground-hybrid --pptx-raw-limit 4 \
  --output-dir ./tmp/chameleon-combined-mvp/portfolio
```

The resulting comparison is written to
`./tmp/chameleon-combined-mvp/portfolio/comparison.json`, with the visual
overview at `./tmp/chameleon-combined-mvp/portfolio/index.html`.

## Early Chameleon observations

The fresh SVG-target-aware run reached the full 4,096-splat optimization cap,
then the existing postprocess pruned it to 3,036 exported splats. Training took
876 seconds. Its internal alpha-over proxy reached 0.94004 SSIM, while the
actual standard SVG reached only 0.64789 SSIM. The old 1,615-splat SVG control
had reached 0.71730, so this run is direct evidence that a larger optimizer
budget can widen the current SVG train-to-deploy gap.

From the exact 3,036-splat checkpoint, 200 iterations of direct continuation
raised proxy SSIM to 0.94872 and actual SVG SSIM to 0.65321. A 100-iteration
normalized Top-K teacher reached only 0.92149 proxy SSIM. Its 100-iteration
exportable student reached 0.93287 proxy SSIM and 0.59136 actual SVG SSIM.
On this checkpoint the Top-K ceiling did not beat direct alpha-over and its
student is therefore a measured losing proposal, not an accepted improvement.

The artifact portfolio still recovered a substantially better SVG from the
direct branch. The winning deployed candidate combines SVG post-fit,
palette quantization, and 64 residual native paths. Its actual `rsvg-convert`
render scores 0.76184 SSIM, 0.78508 MS-SSIM, and 0.25279 LPIPS at 3,100 shapes
and 468,150 bytes. This is a real improvement over both new 4k standard
exports, but remains far below the paper-level 0.99 expectation.

## Foreground/background finding

The PowerPoint evaluation reports full-frame metrics plus:

- exact foreground- and background-mask color errors;
- metrics over the foreground bounding box; and
- metrics over the densest 255×255 foreground focus window.

The user's visual observation is reproduced by the measurements. The student
has a much more explicit chameleon surface, but full-frame SSIM drops to
0.53309 and background L1 rises to 0.10598. The old 2k blur candidate has the
lowest background L1 at 0.09461, but destroys detail: focus LPIPS is 0.58056
and focus edge chamfer is 4.41979.

The indexed foreground hybrid substitutes student splats only in the
foreground while retaining the direct expert's background and global draw
order. It uses 2,167 student and 869 direct splats. The hybrid improves over
the full student on full-frame SSIM (0.56084 versus 0.53309), focus LPIPS
(0.35725 versus 0.37098), and focus edge chamfer (2.20676 versus 2.82558).
It does not yet recover the old smooth background: background L1 is 0.10703.

The old 2k gradient candidate remains the SSIM winner at 0.60622 full-frame
and 0.43506 in the focus window. However, the direct post-fit candidate has
better focus LPIPS (0.35249), and the hybrid has better focus edge chamfer.
This is concrete evidence that selecting on SSIM alone favors smoothing and
does not represent the perceived foreground-detail gain.

## Coffee follow-up

The same portfolio was run on Coffee from the existing 4,000-requested
max-fidelity MLX checkpoint, which contains 2,303 splats after pruning. A
200-step direct continuation reached 0.82694 proxy SSIM and 0.62599 actual SVG
SSIM. The normalized Top-K teacher and exportable student reached 0.81048 and
0.80978 proxy SSIM respectively; the student's actual SVG SSIM was 0.59673.
Top-K therefore did not provide a global ceiling on this full-resolution
checkpoint either.

The winning SVG uses direct continuation, native-blur post-fit, lossless SVGO,
and 64 residual paths. It reaches 0.63156 SSIM, 0.71767 MS-SSIM, and 0.26911
LPIPS with 2,367 shapes at 1,177,703 bytes. The old 2k SVG scored 0.58227 SSIM
and 0.38279 LPIPS at 668,899 bytes after SVGO. A smaller guarded alternative,
the hybrid palette-quantized SVG, scores 0.60722 SSIM and 0.27666 LPIPS at
313,346 bytes.

Coffee does not reproduce the Chameleon hybrid improvement in PowerPoint:
the unhybridized student is the guarded full-frame winner. Relative to the old
gradient PPTX, it raises full-frame SSIM from 0.57362 to 0.58853, lowers
full-frame LPIPS from 0.32440 to 0.25131, lowers focus LPIPS from 0.22800 to
0.13548, and lowers focus edge chamfer from 1.42450 to 0.89195. Background L1
does regress slightly, from 0.06958 to 0.07376.

The indexed hybrid is close but worse on the important focus measures:
0.58315 full-frame SSIM, 0.14329 focus LPIPS, 1.22724 focus edge chamfer, and
0.07662 background L1. The direct blur-post-fit gradient has the best
full-frame LPIPS of the sharp variants at 0.24314, but its SSIM, focus LPIPS,
and focus edge accuracy all trail the student. A native-blur variant obtains
the lowest background L1 at 0.06153, while severely regressing LPIPS to
0.48252.

The automatically derived foreground mask covers 26,641 of 98,304 pixels, but
selects 1,150 of 2,303 splats because detailed table regions attract many
centers. Its bounding box spans the full image height and 318 of 384 columns.
For Coffee, saliency is therefore not an adequate semantic cup/plate mask.
This failure supports replacing the binary center-based switch with either a
semantic/interactive mask or a soft per-splat expert blend with an explicit
background-quality term.

The PowerPoint captures use the real slideshow window ID obtained from
CoreGraphics, followed by a guarded trim of PowerPoint's thin black
presentation matte. This prevents desktop notifications and slideshow chrome
from contaminating the artifact metrics.
