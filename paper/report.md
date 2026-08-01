# Deploying 2D Gaussian Splats to Vector Document Formats

*Draft. The seed noise floor (§5.2) is measured; the corpus-wide PPTX
comparison (§5.3) is not, and says so in place.*

## Abstract

We fit 2D anisotropic Gaussian splats to single images and export them to two
vector document formats: SVG, as radial-gradient ellipses, and PowerPoint
OOXML/DrawingML, as native shapes. The interesting problem is not the fit but
the *export*: the optimizer's differentiable forward model and the renderer
that finally draws the artifact are different programs, and the gap between
them dominates deployed quality.

We report three things. First, a negative result stated up front: on a
21-image corpus, JPEG at an equal byte budget beats our SVG output on
**21 of 21 images**, using 96% less than the budget. This approach is not
competitive on rate-distortion and should not be presented as compression.
Second, the measures that do help — training in the deployment compositor
rather than linear light, and gating every change on the emitted, rasterized
artifact rather than the internal renderer. Third, a content-class
applicability map: splats reconstruct smooth and structured content
acceptably (LPIPS 0.38) and fine stochastic texture and text poorly
(LPIPS 0.60–0.71), which tells a practitioner when *not* to use this.

We also describe, to our knowledge for the first time, a working
splat-to-OOXML pipeline including an empirical calibration of DrawingML's
blur primitive against real PowerPoint.

## 1. Introduction

[TODO: framing — vector-native output as a constraint, not a fidelity play]

The recurring theme of this work is that a splat optimizer reports a number
that the deployed file does not honour. Our internal renderer and the
rasterized SVG disagreed by a wide margin for most of this project's life,
and closing that gap — rather than improving the fit — produced nearly all of
the real quality gains.

## 2. What this is not

A reviewer's first question about any image-to-vector pipeline is whether it
beats simply shipping a raster. It does not, and the margin is not close.

For each corpus image we encoded the best JPEG that fits inside the byte
budget of the corresponding SVG (binary search on quality, capped at q=95):

| | median |
|---|---|
| SVG, deployed LPIPS | 0.5103 |
| JPEG at matched bytes, LPIPS | **0.0034** |
| Images where SVG wins | **0 / 21** |
| JPEG's size vs the budget | **96% smaller** |

JPEG reached the quality cap on every image with roughly 25× headroom
remaining. Any claim of compression or fidelity value would be false.

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
splatting has nothing to represent and is omitted.

### 3.2 The deployment gap

3D Gaussian splatting ships its own rasterizer, so the trained model and the
renderer are one artifact and cannot disagree. Exporting to a document format
inverts this: the renderer belongs to a browser or to PowerPoint, and the
optimizer can only approximate it. Two consequences drove the design.

**Train in the deployment compositor.** Browsers composite SVG in display
space, not linear light. Training with sRGB compositing when the target is
SVG closed roughly half the train/deploy gap. [TODO: cite the measured table]

**Order is fixed at emit time.** SVG document order *is* composite order,
permanently, whereas a 3D splat renderer re-sorts by depth every frame. All
sorts in the pipeline are therefore stable, and ties are resolved
identically across backends.

### 3.3 Evaluating the artifact, not the model

Every number reported here comes from the emitted file: the SVG rasterized by
`rsvg-convert`, the PPTX either rendered by real PowerPoint or scored through
a calibrated proxy (§5.3). Internal-renderer metrics are recorded but never
used to accept a change.

We verified that this measurement is renderer-independent: the same SVG
scored through Chrome and through `rsvg-convert` agrees to four decimal
places on LPIPS (0.4152 vs 0.4151). The figure is a property of the format,
not of the rasterizer, so a browser is not required in the measurement loop.

### 3.4 Metric choice

We gate on LPIPS. SSIM and LPIPS rank our corpus differently and the
disagreement is not marginal: `cell` has the second-best SSIM in the corpus
(0.853) with a middling LPIPS (0.510), while `grass` has the worst SSIM
(0.153) and a mid-table LPIPS (0.656). SSIM systematically rewards the
smoothness that splat renderings produce in abundance. [TODO: cite LPIPS,
SSIM blur-bias work]

## 4. Reference corpus

Twenty-one images from `scikit-image`'s standard sample data, normalized to a
384 px maximum edge, spanning portrait, fur, landscape, natural, graphic,
transparency, smooth-gradient, hard-edge, text-like, texture, grayscale and
dark-sparse content, plus the project's standing test image.

The corpus is content-hashed and duplicate-checked on materialization; this
immediately caught `scikit-image`'s `cat` being an alias for `chelsea`, a
duplicate that would have double-weighted one image in every aggregate.

Using a standard corpus matters here beyond reproducibility. Our own standing
test image ranks **3rd of 21** (LPIPS 0.397 against a corpus median of
0.510) — it is a favourable image, and every single-image number this project
produced before the corpus was optimistic by roughly 0.11 LPIPS.

## 5. Results

### 5.1 Fidelity by content class

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
defeats them is texture whose entropy exceeds the splat count.

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

### 5.3 SVG versus OOXML

*Still to be measured.* A corpus-wide SVG-versus-OOXML comparison requires a
real-PowerPoint slideshow capture of all 21 decks. LibreOffice is not an
acceptable substitute — it renders DrawingML incorrectly for these shapes — and
the capture drives a live PowerPoint window, so the pass is attended rather than
batchable. The single-image result below stands until it is done, and is
labelled as such rather than generalized.

On the standing test image alone, PPTX scored better than SVG (LPIPS 0.406
vs 0.415, SSIM 0.734 vs 0.701) at one sixth the file size, rendered by real
PowerPoint. Whether that survives the corpus is exactly the kind of
single-image claim this section exists to test.

## 6. Splats as OOXML

[TODO: DrawingML specifics]

- Ellipse + radial `gradFill` per splat; `<a:blur rad>` calibrated at
  σ = rad / 3.25 by erf-fit edge response against real PowerPoint.
- `<a:softEdge>` is a feathered hard shape, not a Gaussian — confirmed by
  capture, and the reason the gradient style is the default.
- Slide geometry must respect the OOXML one-inch minimum; canvases below
  96 px otherwise emit a schema-invalid deck.
- Negative offsets are legitimate and must survive: splats overlapping the
  slide edge carry negative `a:off`, and clamping them displaces every
  border splat inward.

## 7. Negative results

Reported because they cost real time and are absent from the literature.

- **`feGaussianBlur` as the SVG primitive.** A true Gaussian via filter
  primitives loses to radial-gradient stops: rsvg's filter pipeline is 8-bit,
  so a point source underflows before amplification, and enlarging the source
  to compensate flattens it into a disc. ~6× slower for worse output.
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
- PPTX corpus numbers use a calibrated proxy; real PowerPoint validation is
  on a subset only, because PowerPoint is not scriptable at corpus scale.
- 384 px maximum edge; behaviour at print resolution is unmeasured.
- No human study. Perceptual claims rest on LPIPS.
- One machine, one hardware generation for all timings.

## 9. Conclusion

[TODO]

## References

[TODO — from the survey]
