# Circumventing PowerPoint's gradient rendering with solid-alpha rings

Measured 2026-08-02 with `tools/pptx_ring_stack_mvp.py`, following the
color-space probe (`docs/pptx-colorspace.md`). All captures are real
Microsoft PowerPoint slideshows under the ICC-corrected protocol.

## The idea

PowerPoint renders this project's alpha-ramp gradient splats with a median
**0.10 LPIPS of emitter loss** against the reference render of the same
splats — three to five times the SVG (0.031) or CSS (0.022) emitters. But
the probe showed PowerPoint composites *solid-alpha* shapes in clean,
predictable display sRGB, within 0.006 of the model. So: stop using the
primitive it mangles. Approximate each Gaussian as K concentric solid-alpha
ellipses whose per-ring alphas are solved so the cumulative sRGB alpha-over
composite matches the Gaussian profile at each ring midpoint — a stepwise
falloff built from the one primitive PowerPoint provably renders exactly.

## Results (unfitted populations, equal-fraction radii)

Deployed LPIPS against the source, rings versus the gradient baseline, same
populations, same painter order:

| image | gradient | rings K=4 | rings K=8 |
|---|---:|---:|---:|
| chameleon | 0.2075 | 0.2786 | 0.2306 |
| colorwheel | 0.2138 | 0.2465 | **0.1924** |
| text | 0.2653 | **0.2426** | 0.2630 |
| hubble_deep_field | 0.4167 | **0.3725** | **0.3773** |

At K=4 the stepwise falloff bands visibly on smooth content. At K=8 the
picture is two wins, one tie, one small deficit — achieved with populations
**trained for the gradient model**, a first-guess ring layout, and no
fitting for the new primitive at all. Deck sizes grow from ~160 KB to
250–450 KB, still a fraction of the SVG artifact.

## Why this is the right direction anyway

The strategic property is not the four scores; it is that the ring
compositor is **exactly modelable**. The gradient emitter's 0.10 LPIPS loss
is unfixable by training because no proxy reproduces PowerPoint's ramp
rendering; the ring emitter's loss is a computable function of K and the
radii, and the fit can train directly against the true deployed composite.
That converts PPTX from the one format with an unmodelable compositor into
one with a closed loop, the same move that took browser SVG from 0.40 to
0.24 deployed LPIPS.

## Visual verdict: not shippable yet

The eyeball rule overrides the per-image metric wins. Plain rings leave
visible concentric contours on large smooth splats (observed directly on
the deployed deck), which no headline metric fully punishes. Feathering
every ring with the calibrated DrawingML blur (sigma = rad/3.25) melts the
contours but trades them for softness: chameleon at K=6 blurred scores
0.2606 against plain K=8's 0.2306 and the gradient's 0.2075. The banding
concentrates in the outer low-alpha steps, because equal-radius rings make
unequal alpha jumps.

So the primitive is promising and exactly modelable, but the current
construction is not shippable. The gradient style remains the default and
the deployed baseline.

## Next steps

1. Alpha-quantile radii: place rings at equal alpha steps, so the banding
   energy is spread evenly instead of concentrated in the outer contours.
2. Selective feathering: blur only the outer one or two rings, keeping the
   inner rings crisp for detail.
3. Ring-aware training — the decisive step: a differentiable stepwise proxy
   (the ring construction is a deterministic function of the Gaussian
   parameters), so populations are fitted for the primitive that ships and
   the fit itself learns to hide the steps.
4. Full-corpus comparison at the winning configuration, judged by the
   standard real-PowerPoint pass **and a side-by-side eyeball**, before any
   `--pptx-splat-style rings` default discussion.
