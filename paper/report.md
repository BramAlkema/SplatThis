# Deploying 2D Gaussian Splats to Vector Document Formats

**Bram Alkema** · SplatThis v0.2.6 · August 2026 ·
[code](https://github.com/BramAlkema/SplatThis) ·
[project page](https://bramalkema.github.io/SplatThis/)

*Technical report. Every number below is measured; none are pending. Numbers
in §5.4 are regenerated from the repository's versioned ledgers by
`tools/update_paper.py`; the historical sections are dated and frozen.*

## Abstract

We fit 2D anisotropic Gaussian splats to single images and export them to
declarative document formats that ship no rasterizer of their own: SVG radial
gradients, scriptless CSS, and PowerPoint OOXML/DrawingML shapes. The
interesting problem is not the fit but the *export*: the optimizer's
differentiable forward model and the renderer that finally draws the artifact
are different programs, and the gap between them dominated deployed quality
for most of this project's life.

We report four things. First, a negative result stated up front: on a
21-image corpus, JPEG at an equal byte budget beats our SVG output on
**21 of 21 images** while using 96% less than the budget. This approach is
not competitive on rate–distortion and should not be presented as
compression. Second, the measures that actually closed the deployment gap —
training in the deployment compositor rather than linear light, emitting in
the compositor's fixed paint order, and gating every change on the emitted,
rasterized artifact rather than the internal renderer. Together these roughly
halved deployed error (median LPIPS 0.4023 → 0.2433 on the same
populations). Third, that once the gap is closed, *the emitter stops
mattering*: SVG, high-precision SVG, and scriptless CSS land within 0.001
median LPIPS of one another, below the corpus's own seed noise, so format
choice is a question of embedding constraints and bytes, not fidelity.
Fourth, a content-class applicability map: graphics and structured content
reconstruct well (deployed LPIPS 0.02–0.15) while broadband texture bottoms
out around 0.51–0.54, which tells a practitioner when *not* to use this.

We also describe, to our knowledge for the first time, a working
splat-to-OOXML pipeline, including an empirical calibration of DrawingML's
blur primitive against real Microsoft PowerPoint.

## 1. Introduction

3D Gaussian splatting [1] owes part of its practicality to a closed loop: the
trained model and the renderer ship together, so what the optimizer sees is
what the viewer gets. Recent 2D variants such as GaussianImage [5] keep that
loop closed and compete on rate–distortion. This report studies what happens
when the loop is deliberately opened: the splats are handed to a renderer we
do not control — a browser's SVG and CSS compositors, or Microsoft
PowerPoint — because the *deliverable* is a document. A slide whose splats
are native editable shapes, an image that survives a no-JavaScript content
policy, a vector that scales without resampling: these are constraints, not
fidelity plays, and the correct comparison inside them is not against a
codec but against not shipping the format at all.

Opening the loop moves the difficulty. Fitting Gaussians to one image is a
solved, stable optimization; §5.2 measures its seed-to-seed noise at 0.029
SSIM worst-case. What is not solved is that the deployed file is drawn by a
program the optimizer can only approximate. The recurring theme of this work
is that a splat optimizer reports a number the deployed file does not
honour. Our internal renderer and the rasterized SVG disagreed by a wide
margin for most of this project's life, and closing that gap — rather than
improving the fit — produced nearly all of the real quality gains.

Concretely, this report contributes:

- a deployment discipline — train in the target's compositing space, emit in
  its paint order, and accept changes only on the rasterized artifact in its
  governing renderer (§3);
- a measured decomposition of deployed error into fitting error and emitter
  error, showing the fit dominates and corrected emitters are mutually
  indistinguishable (§5.4);
- a splat-to-OOXML pipeline with an empirical calibration of DrawingML's
  blur primitive against real PowerPoint (§6);
- negative results, including the rate–distortion comparison every
  image-to-vector paper should publish and most do not (§2, §7).

## 2. What this is not

A reviewer's first question about any image-to-vector pipeline is whether it
beats simply shipping a raster. It does not, and the margin is not close.

For each corpus image we encoded the best JPEG that fits inside the byte
budget of the corresponding SVG (binary search on quality, capped at q=95;
ledger: `result/corpus/baselines.jsonl`, July 2026 emitter):

| | median |
|---|---|
| SVG, deployed LPIPS | 0.5103 |
| JPEG at matched bytes, LPIPS | **0.0034** |
| Images where SVG wins | **0 / 21** |
| JPEG's size vs the budget | **96% smaller** |

JPEG reached the quality cap on every image with roughly 25× headroom
remaining. The SVG side of this table predates the corrected emitter; the
correction (§5.4) roughly halves it, which narrows a two-orders-of-magnitude
gap without changing the outcome. Any claim of compression or fidelity value
would be false.

What the vector output buys is categorical rather than quantitative: the
result is *shapes*, not pixels — editable in a vector tool, scalable without
resampling, and in the OOXML case, native objects inside a PowerPoint slide.
Where that is a hard requirement, a raster is not a substitute at any
quality. Where it is not a requirement, this pipeline is the wrong tool, and
we would rather say so than let a reader discover it.

## 3. Method

### 3.1 Representation

Each splat carries nine trained values: position (2), scale (2), rotation,
RGB (3), and alpha, held as an `[N, 11]` tensor with a frozen importance
channel that fixes compositing order. There is no view dependence — the
target is a single image, so the spherical-harmonic colour of 3D Gaussian
splatting [1] has nothing to represent and is omitted.

### 3.2 The deployment gap

3D Gaussian splatting ships its own rasterizer, so the trained model and the
renderer are one artifact and cannot disagree. Exporting to a document format
inverts this: the renderer belongs to a browser or to PowerPoint, and the
optimizer can only approximate it. Three consequences drove the design.

**Train in the deployment compositor.** Browsers composite SVG in display
space, not linear light. Training with sRGB compositing when the target is
SVG roughly halved the train/deploy gap on both metrics (measured in
`docs/PROVENANCE_AND_BENCHMARKS.md`); the same change applied *after*
training as a fixed-splat conversion does not pay (§7).

**Order is fixed at emit time.** SVG document order and CSS DOM order *are*
composite order, permanently, whereas a 3D splat renderer re-sorts by depth
every frame. Emitting back-to-front so the painter's order matches the
optimizer's front-to-back transmittance model was the single largest
correction in the project (§5.4); all sorts in the pipeline are stable, and
ties resolve identically across backends.

**Sample the falloff where the compositor interpolates.** Declarative
gradients are piecewise-linear in the stop list. Sampling the exact opacity
curve at evenly spaced stops — rather than placing stops adaptively —
preserves the Gaussian tail that alpha-over accumulation depends on. In CSS
the same idea goes further: colour is emitted as a linear-sRGB solid fill and
the Gaussian is applied as a 9-stop alpha *mask*, so the browser never
interpolates colour and opacity together. The mask half does not transfer to
SVG (`radialGradient` cannot express it; SVG masks failed their quality/cost
gate), but the stop-sampling half already had: the two emitters end §5.4
indistinguishable.

### 3.3 Evaluating the artifact, not the model

Every number reported here comes from the emitted file, never from the
internal renderer: internal metrics are recorded as diagnostics but cannot
accept a change.

Which rasterizer scores the file turned out to matter. Early in the project
one SVG scored identically through Chrome and `rsvg-convert` to four decimal
places on LPIPS (0.4152 vs 0.4151), which briefly justified using rsvg as a
browser stand-in. Later artifacts broke the agreement — the same file scored
0.7618 SSIM in librsvg against 0.7193 in Chrome — so the claim of
renderer-independence did not survive and is withdrawn. All browser-target
figures in this report are therefore governed by native-size Playwright
Chromium capture; rsvg-scored rows are labelled historical. PPTX is rendered
by real Microsoft PowerPoint (§5.3); LibreOffice is not used anywhere, as it
renders DrawingML incorrectly for these shapes.

### 3.4 Metric choice

We gate on LPIPS [2]. SSIM [3] and LPIPS rank our corpus differently and the
disagreement is not marginal: `cell` has the second-best SSIM in the corpus
(0.853) with a middling LPIPS (0.510), while `grass` has the worst SSIM
(0.153) and a mid-table LPIPS (0.656). SSIM systematically rewards the
smoothness that splat renderings produce in abundance, a known failure mode
[4]. The starkest measurement: against the reference render of one splat
population, a 2-pixel Gaussian blur scores 0.9290 SSIM and the *source
photograph* — a different image entirely — scores 0.9053, against 0.9336 for
the actual SVG; LPIPS separates the same three cases as 0.1268 / 0.1456 /
0.0411. SSIM figures are still published for continuity, and no SSIM
difference below the seed noise floor (§5.2) is claimed as a result.

## 4. Reference corpus

Twenty-one images from `scikit-image`'s standard sample data [6], normalized
to a 384 px maximum edge, spanning portrait, fur, landscape, natural,
graphic, transparency, smooth-gradient, hard-edge, text-like, texture,
grayscale and dark-sparse content, plus the project's standing test image.

The corpus is content-hashed and duplicate-checked on materialization; this
immediately caught `scikit-image`'s `cat` being an alias for `chelsea`, a
duplicate that would have double-weighted one image in every aggregate.

Using a standard corpus matters here beyond reproducibility. Our own standing
test image ranks **3rd of 21** (LPIPS 0.397 against a corpus median of
0.510) — it is a favourable image, and every single-image number this project
produced before the corpus was optimistic by roughly 0.11 LPIPS.

## 5. Results

### 5.1 Fidelity by content class (initial emitter, July 2026)

Scored through `rsvg-convert` on the pre-correction emitter; frozen here as
the baseline that §5.4 improves on.

| content class | n | median LPIPS |
|---|---:|---:|
| texture | 4 | 0.598 |
| text-like | 2 | 0.563 |
| natural | 2 | 0.557 |
| dark-sparse | 1 | 0.524 |
| grayscale | 3 | 0.510 |
| portrait | 1 | 0.506 |
| smooth-gradient | 2 | 0.473 |
| transparency | 1 | 0.445 |
| graphic | 1 | 0.443 |
| hard-edges | 1 | 0.432 |

Corpus: median LPIPS 0.510, mean 0.509, sd 0.090; median SSIM(sRGB) 0.513;
median 765 KB; median 60 s on Apple Silicon.

Best and worst are instructive. `brick` (0.381) and `moon` (0.388) are
smooth or regularly structured. `page` (0.714) and `grass` (0.656) are dense
high-entropy detail. We expected hard edges to be the worst case for soft
primitives and were wrong: `checkerboard` scored 0.432, better than the
corpus median. At a sufficient budget, splats resolve step edges; what
defeats them is texture whose entropy exceeds the splat count. The ordering
survives the emitter correction (§5.4): correcting the compositor lifts
every class, but texture stays last.

### 5.2 Seed noise floor

Measured on six corpus images spanning the content-gradient range, three seeds
each, at 600 splats / 256 px on the Torch backend (the MLX backend is not
byte-reproducible and is excluded deliberately):

| image | seed 0 | seed 1 | seed 2 | spread |
|---|---:|---:|---:|---:|
| moon | 0.8089 | 0.8075 | 0.8085 | 0.0014 |
| cell | 0.8274 | 0.8291 | 0.8253 | 0.0039 |
| chameleon | 0.6358 | 0.6494 | 0.6207 | **0.0287** |
| astronaut | 0.3098 | 0.3214 | 0.3113 | 0.0116 |
| gravel | 0.1206 | 0.1184 | 0.1102 | 0.0105 |
| grass | 0.1186 | 0.1174 | 0.1188 | 0.0014 |

Worst-case spread is **0.029 SSIM_sRGB**; the median is 0.007 and the mean
per-image standard deviation is 0.005. The floor is therefore taken as
**0.029**, the worst case rather than the median, because a claim must survive
the least favourable image rather than the typical one.

Note that the floor is not uniform across content: `chameleon` is twenty times
noisier across seeds than `moon` or `grass`. Seeding is content-adaptive, so
images with many near-equal candidate placements admit more variation than
either very smooth or very uniformly textured ones.

No difference smaller than this floor is claimed anywhere in this report.

### 5.3 SVG versus OOXML (July 2026 configurations)

Measured across all 21 corpus images, each deck captured from a real Microsoft
PowerPoint slideshow, each SVG in governing Chromium, both sides on the
pre-correction emitters (ledgers: `result/corpus/powerpoint_results.jsonl`,
`data/svg-compositor-corpus.json`).

| | SVG (Chromium) | PowerPoint |
|---|---:|---:|
| median SSIM_sRGB | 0.5973 | 0.6091 |
| median LPIPS | 0.4023 | 0.3843 |
| median size | 670 KB | **127 KB** |
| images won, LPIPS | 9 | 12 |
| images won, SSIM | 11 | 10 |

**The quality difference is not claimable.** PowerPoint leads by 0.012 SSIM
and 0.018 LPIPS at the median, both of which sit below the 0.029 seed noise
floor established in §5.2, and the per-image win count is a coin flip in each
metric. The honest statement is that the two formats reconstruct equally
well.

**The size difference is claimable and large.** PowerPoint output is 5.3x
smaller at the median. Since quality is indistinguishable, that is the whole
result: for equal fidelity, the OOXML path costs a fifth of the bytes.

Content class splits them where the medians do not. Text-like content strongly
favours SVG — `text` scores 0.7363 against PowerPoint's 0.3785, and `page`
0.5572 against 0.3423 — because DrawingML's soft-edged shapes cannot hold a
glyph edge that an SVG gradient stop can. Smooth and structured content leans
the other way: `moon` 0.8987 against 0.8442, `retina` 0.6784 against 0.5725,
`cell` 0.9054 against 0.8560. A practitioner choosing between the two should
select on content and byte budget, not on an expected quality difference that
this corpus cannot demonstrate.

On the standing test image alone, PPTX scored better than SVG (LPIPS 0.406
vs 0.415, SSIM 0.734 vs 0.701) at one sixth the file size, rendered by real
PowerPoint. Whether that survives the corpus is exactly the kind of
single-image claim this section exists to test.

<!-- current-emitters:begin -- generated by tools/update_paper.py from the versioned ledgers; edit the ledgers, not this block -->

### 5.4 Corrected emitters and the fit/emitter split (August 2026)

This section is regenerated from the repository's versioned ledgers (`compositor-fidelity.json`, `svg-compositor-corpus.json`, `svg-quality/measurements.json`); every published median is re-derived from its per-image evidence before rendering, and the build fails on any mismatch.

Applying §3.2 -- paint order and exact stop sampling -- to the same populations, deployed against the original image in governing Chromium:

| emitter | LPIPS ↓ median | LPIPS p90 | SSIM median | median size |
|---|---:|---:|---:|---:|
| legacy order (baseline, §5.3) | 0.4023 | — | 0.5973 | — |
| SVG, `standard` gradients | 0.2433 | 0.4619 | 0.7404 | 765 KB |
| SVG, `high` gradients | 0.2439 | 0.4532 | 0.7483 | 1,331 KB |
| scriptless CSS | 0.2429 | 0.4586 | 0.7517 | — |

The correction roughly halves deployed error (0.4023 → 0.2433 median LPIPS) and collapses the emitters: the three declarative targets span 0.001 median LPIPS, below the §5.2 noise floor. Isolating the emitter term -- the same splats through the deployed compositor versus the internal reference render -- shows why:

| emitter | LPIPS median | SSIM median |
|---|---:|---:|
| SVG, `standard` | 0.0310 | 0.9303 |
| SVG, `high` | 0.0181 | 0.9486 |
| scriptless CSS | 0.0221 | 0.9426 |
| pixel runtime | 0.0001 | 0.9993 |

Deployed error is 8x the emitter's own loss for standard gradients and 13x for high: the fit, not the format, is the limit. Against the original, the declarative emitters are indistinguishable: median LPIPS 0.2433 / 0.2439 / 0.2429 for svg / svg-high / css, a spread of 0.001, and an SSIM spread of 0.011 that sits below the 0.029 seed noise floor. Emitter choice is not where deployed quality comes from; the fit is. Compositor-only figures make the emitters look far more different than they are, and must not be quoted as quality.

Per content class, deployed median LPIPS on the corrected emitters:

| content class | n | SVG | SVG high | CSS |
|---|---:|---:|---:|---:|
| texture | 4 | 0.4596 | 0.4696 | 0.4724 |
| dark-sparse | 1 | 0.4402 | 0.4276 | 0.4271 |
| smooth-gradient | 2 | 0.3054 | 0.2984 | 0.2883 |
| grayscale | 3 | 0.2990 | 0.2645 | 0.2568 |
| text-like | 2 | 0.2953 | 0.2899 | 0.2962 |
| natural | 2 | 0.2720 | 0.2732 | 0.2767 |
| fur | 1 | 0.2459 | 0.2447 | 0.2520 |
| portrait | 1 | 0.2422 | 0.2326 | 0.2351 |
| landscape | 1 | 0.2136 | 0.1896 | 0.1856 |
| hard-edges | 1 | 0.1733 | 0.1557 | 0.1614 |
| reference | 1 | 0.1672 | 0.1491 | 0.1456 |
| transparency | 1 | 0.0863 | 0.0789 | 0.0823 |
| graphic | 1 | 0.0155 | 0.0127 | 0.0122 |

The default per-image artifact gate originally raced each candidate against a legacy-order incumbent; in the July study it kept `high` gradients on 17 of 21 images, legacy order on 4, and `standard` on 0, for gate medians of 0.7111 SSIM / 0.2439 LPIPS -- below either fixed corrected choice, because incumbency let legacy persist without winning. As of 0.2.6 the incumbent is the default corrected-standard emitter, bounding the gate's floor at the default output; the PPTX shape-order default of §6 flipped to corrected in the same release.

<!-- current-emitters:end -->

## 6. Splats as OOXML

DrawingML was not designed to draw Gaussians, and the pipeline's job is to
find the encoding whose failure modes cost the least. Four findings, all
verified by real-PowerPoint capture:

**Encoding.** Each splat is one ellipse with a radial `gradFill` whose stop
list samples the Gaussian falloff — the same stop-sampling principle as the
SVG and CSS emitters (§3.2), expressed in DrawingML's vocabulary. The deck
contains native shapes only; there is no embedded preview bitmap pretending
to be a slide.

**Blur is usable but must be calibrated.** `<a:blur rad="...">` is
isotropic-only and its radius parameter is not a standard deviation. Fitting
an error-function edge response against real PowerPoint captures gives
σ ≈ rad / 3.25; anisotropy is approximated by the ellipse's aspect ratio
under an isotropic blur. `<a:softEdge>`, the more obvious primitive, is a
feathered hard shape rather than a Gaussian — confirmed by capture, and the
reason the gradient style is the default.

**The schema has physical limits.** Slide geometry must respect OOXML's
one-inch minimum: canvases below 96 px otherwise emit a schema-invalid deck.
Negative offsets are legitimate and must survive — splats overlapping the
slide edge carry negative `a:off`, and clamping them displaces every border
splat inward.

**Paint order transfers.** The back-to-front correction of §5.4 applies to
DrawingML shape order too. On the same 21 populations, corrected order
improves the median (LPIPS 0.3200 vs 0.3750, real-PowerPoint captures;
`data/pptx-order-compositor-corpus.json`) but regresses individual images,
so a strict per-image artifact gate kept corrected order on 14 of 21 and
legacy on 7. Corrected order is the default as of v0.2.6, since it matches
the renderer's transmittance model and wins the corpus median; the legacy
stack remains available for the rare regressing image, and the external gate
can still choose per image.

## 7. Negative results

Reported because they cost real time and are absent from the literature.

- **`feGaussianBlur` as the SVG primitive.** A true Gaussian via filter
  primitives loses to radial-gradient stops: rsvg's filter pipeline is 8-bit,
  so a point source underflows before amplification, and enlarging the source
  to compensate flattens it into a disc. ~6× slower for worse output.
- **SVG masks as the CSS technique's port.** The alpha-mask-over-solid-fill
  construction that lifted CSS (§3.2) does not pay in SVG: mask and
  linearRGB-filter variants failed their quality-or-cost gates on the
  Chameleon MVP and were excluded from the corpus study.
- **torch on Apple MPS.** 4.5× *slower* than CPU on the same workload
  (269 s vs 59 s, 400 px / 2000 splats). MLX, not MPS, was the answer.
- **sRGB compositing as a fixed-splat change.** Matching the browser's blend
  improves a fixed splat set but makes optimization harder by roughly the
  same amount; it only pays when training in that space from the start.
- **Post-hoc SVG optimization at low precision.** Running `svgo` at
  precision 1 collapses the stop-opacity vocabulary from 64 distinct values
  to 7 and truncates Gaussian tails, costing 0.0063 LPIPS. At precision 2 the
  optimization is free and still 15% smaller.

## 8. Limitations

- Single optimizer configuration; no ablation over splat budget or schedule.
- Real-PowerPoint captures cover the full corpus for §5.3 and the
  paint-order study, but at one capture repeat; PowerPoint is not scriptable
  enough for repeat-render noise calibration at corpus scale.
- The CSS emitter has no calibrated repeat-render noise floor of its own; its
  indistinguishability claims borrow the SVG/pixel-runtime floor of §5.2.
- 384 px maximum edge; behaviour at print resolution is unmeasured.
- No human study. Perceptual claims rest on LPIPS.
- One machine, one hardware generation for all timings.
- The MLX backend is not byte-reproducible (Metal reorders float32
  reductions); metrics agree to nine significant figures, but artifact
  hashes are only stable on the Torch backend.

## 9. Conclusion

The lesson of this project is a displacement of difficulty. Fitting 2D
Gaussians to an image is stable and well-behaved; its seed noise is
measurable and small. Deploying those Gaussians through a renderer the
optimizer does not control is where quality is won or lost, and the wins came
from alignment, not cleverness: composite in the space the target composites
in, emit in the order the target paints in, sample the falloff where the
target interpolates, and let only the rasterized artifact accept a change.
Under that discipline the emitters converge — SVG, high-precision SVG, and
scriptless CSS become indistinguishable against the original, and even
PowerPoint's DrawingML lands within the seed noise floor of SVG — so format
choice reduces to what it always should have been: constraints. Bytes favour
OOXML at 5.3× smaller; no-script policies force CSS; editability and
single-file delivery argue for SVG. Fidelity does not vote.

The honest boundaries stand. Against a codec this is not compression, by two
orders of magnitude. Against broadband texture the representation runs out of
entropy. Within its constraints — documents that must remain shapes — the
pipeline delivers a measured, reproducible quality level, and the fit, not
the format, is what limits it.

## References

1. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting
   for Real-Time Radiance Field Rendering.* ACM Transactions on Graphics
   42(4), SIGGRAPH 2023.
2. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. *The Unreasonable
   Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
3. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image Quality
   Assessment: From Error Visibility to Structural Similarity.* IEEE
   Transactions on Image Processing 13(4), 2004.
4. J. Nilsson, T. Akenine-Möller. *Understanding SSIM.* arXiv:2006.13846,
   2020.
5. X. Zhang et al. *GaussianImage: 1000 FPS Image Representation and
   Compression by 2D Gaussian Splatting.* ECCV 2024.
6. S. van der Walt et al. *scikit-image: image processing in Python.* PeerJ
   2:e453, 2014.
