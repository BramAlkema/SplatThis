# Mixed-Primitives Fidelity MVP

**Status:** promising experiment; not yet enabled in the converter\
**Primitive:** residual-guided connected edge paths
**Outputs:** native SVG paths and editable DrawingML rounded segments

## Hypothesis

Gaussian splats are efficient for smooth color fields but spend too much of
their budget approximating thin edges. A very small number of native vector
paths should repair high-value residual edges more efficiently than adding or
shrinking more Gaussians.

## Method

1. Rasterize the deployed baseline SVG.
2. Compare it with the source in display sRGB.
3. Build a priority map from target-edge magnitude times color residual.
4. Trace iso-luma contours through high-priority regions.
5. Keep short connected fragments and simplify their vertices.
6. Solve each path's source color for alpha-over compositing.
7. Sweep bounded path count, length, width, and opacity.
8. Capture and score every emitted SVG at native dimensions in Chromium.
9. Select the lowest-complexity candidate that gains at least 0.005 SSIM with
   no material PSNR regression.
10. Translate accepted paths to editable rounded DrawingML segment shapes.
11. Open baseline and mixed decks in Microsoft PowerPoint, run its full-screen
    slideshow renderer with the pointer hidden, capture the queried slide
    surface, and score equal-size rasters.

Isolated short strokes were tested first. They produced large metric gains but
looked like disconnected sticks at 4x inspection, so they are not the selected
primitive despite their scores. Connected paths plus a minimum-complexity
selection rule reduce that failure, but some fragments remain visibly
disconnected at 4x. They are acceptable for measuring the representation idea,
not yet visually clean enough for production.

## First results

The baseline is the same 64-splat, 64-pixel, SVG-post-fitted direct arm used by
the top-K experiment.

| Image | Paths | Native PPTX segment shapes | SVG SSIM gain | PowerPoint SSIM gain |
|---|---:|---:|---:|---:|
| logo | 2 | 5 | +0.0051 | +0.0006 |
| chameleon | 4 | 6 | +0.0051 | +0.0054 |
| rocket | 4 | 11 | +0.0073 | +0.0067 |

PSNR also improved for all six deployed-artifact comparisons. SVG file growth
was approximately 0.5-0.9 KB for the selected candidates. The PPTX packages
remain native vector packages without embedded correction bitmaps.

## Decision

Proceed to a six-image, three-seed experiment and add local edge/ROI gates.
Do not yet enable mixed primitives by default. Promotion requires:

- positive actual SVG and PowerPoint medians;
- no material image-level, worst-ROI, or edge regression;
- visual inspection at 1x and 4x;
- a bounded native shape and file-size increase;
- deterministic primitive selection.

Before production, contour fragments must also gain continuation/curvature and
color-consistency constraints so the metric cannot be improved with an
implausible isolated mark.

If that experiment holds, integrate the operator into the ADR-003
accept-or-revert fidelity stage rather than the normal Gaussian training loop.

## Full-corpus result

**Historical renderer note:** the measurements below used `rsvg-convert`
before Chromium became the governing browser target. They remain useful for
explaining the earlier decision but cannot promote a browser-SVG operator. The
runner now uses Playwright Chromium and requires a fresh run for new claims.

The follow-up evaluated all 21 full corpus frames against the existing
MLX-trained seed-0 SVG and PPTX artifacts. Candidate selection used the actual
`rsvg-convert` result, a +0.001 SVG SSIM threshold, and a bounded sweep of 16,
32, or 64 connected paths. Eighteen images accepted a candidate; `cell`,
`colorwheel`, and `moon` reverted.

For the 18 accepted SVGs:

- SSIM, MS-SSIM, LPIPS, PSNR, and both OKLab error statistics improved on all
  18 images.
- Median ΔSSIM was +0.00178 and median ΔLPIPS was -0.00604.
- Edge chamfer improved on 17/18 and worst-tile error improved on 12/18.
- Median SVG growth was 3.3 KB.

The actual Microsoft PowerPoint result did not preserve the SVG SSIM gain:

| Metric | Median delta | Images improved |
|---|---:|---:|
| SSIM | -0.00017 | 5/18 |
| MS-SSIM | -0.00019 | 7/18 |
| LPIPS | -0.00420 | 15/18 |
| PSNR | -0.00339 dB | 2/18 |
| Edge chamfer | -0.00621 | 16/18 |
| Worst-tile error | 0.00000 | 3/18 |
| PPTX package size | +1.4 KB | n/a |

This disagreement is useful: native rounded DrawingML segments often help
perceptual and edge metrics, but their PowerPoint rasterization is not
pixel-equivalent to the selected SVG path. The rocket capture also makes a
small implausible residual mark visible that is easy to overlook in aggregate
metrics.

**Decision:** keep mixed paths experimental and disabled. The next MVP should
select or revert using both actual SVG and actual PowerPoint artifacts, and
should add continuation, curvature, and minimum-length constraints before
testing larger path budgets. The current implementation is evidence for a
mixed representation, not yet a shippable operator.
