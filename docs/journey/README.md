# How we got here

Ten months of fitting 2D Gaussians to a photograph, told through the
fourteen renders that survived. Everything else — 187 scratch files,
112 MB of one-off comparison pages — has been binned.

Open [`index.html`](index.html) for the full gallery with captions.

## 01 · Sept 2025 — First light

A splat renderer that drew, then a splat renderer that fit a photo — badly. Everything here is the retired `splat_this` pipeline learning what the primitive could and couldn't do.

- **Four splats on black** — The first thing the renderer ever produced: a handful of isotropic Gaussians, flat-coloured, composited over an empty black canvas. No image fitting yet — just proof the primitive drew. [[still](stills/01-first-splats.jpg)] · [svg](svg/01-first-splats.svg)
- **The hole in the middle** — First real photo fit. Splats clustered on high-gradient detail and left the smooth centre of the subject completely uncovered — the black is canvas showing through, not paint. [[still](stills/02-coverage-hole.jpg)] · [svg](svg/02-coverage-hole.svg)
- **Halftone collapse** — Size-varied placement without overlap control: splats landed on a near-regular lattice and the gaps between them read as a printed dot screen. [[still](stills/03-halftone.jpg)]
- **First full frame** — Depth/spacing rework: the subject and most of the frame finally get paint, though the bottom edge still drops to bare canvas. Soft and muddy, but for the first time the output is a picture rather than a scatter plot. [[still](stills/04-full-frame.jpg)] · [svg](svg/04-full-frame.svg)

## 02 · Sept 2025 — More splats won't save you

The chameleon becomes the standing test image and the budget goes up 8×. It doesn't help. The background is black because nothing paints it — a coverage bug that no amount of density can outrun.

- **100 splats** — The chameleon becomes the standing test image. At 100 splats it is pure confetti — and critically, the background is still black. [[still](stills/05-density-100.jpg)] · [svg](svg/05-density-100.svg)
- **400 splats** — 4× the budget. More confetti, denser clusters, same black ground. [[still](stills/06-density-400.jpg)] · [svg](svg/06-density-400.svg)
- **800 splats** — 8× the budget and the picture is no closer to legible. The lesson that shaped everything after: coverage was the bug, not density. Throwing splats at it could never fix a missing background. [[still](stills/07-density-800.jpg)] · [svg](svg/07-density-800.svg)
- **Uniform placement** — Placement-strategy bake-off on a synthetic target. Uniform seeding spends budget evenly and blurs every edge. [[still](stills/08-placement-uniform.jpg)] · [svg](svg/08-placement-uniform.svg)
- **Structure-guided placement** — Same budget, seeded from the structure tensor. Edges survive as oriented strokes — the ancestor of today's anisotropic init. [[still](stills/09-placement-structure.jpg)] · [svg](svg/09-placement-structure.svg)

## 03 · Feb 2026 — The Python rewrite

`png2splat` rebuilds the pipeline and climbs the resolution ladder. Sharper, denser, and carrying the same structural hole as six months before.

- **256px** — The png2splat rewrite. A resolution ladder to find where detail starts paying for itself — still splats-on-black. [[still](stills/10-res-256.jpg)] · [svg](svg/10-res-256.svg)
- **768px** — 3× the resolution, far more splats, still no ground. Same structural gap as six months earlier, now at higher fidelity. [[still](stills/11-res-768.jpg)] · [svg](svg/11-res-768.svg)

## 04 · Feb–May 2026 — Paint the canvas

An explicit background plate plus radial gradients that follow the renderer's real opacity curve. The moment the output stops being a scatter plot and starts being an image.

- **Background and true Gaussian falloff** — The turn. An explicit background plate plus per-splat radial gradients following the renderer's real opacity curve, 1&nbsp;−&nbsp;e<sup>−&alpha;G</sup>. The canvas is painted, the falloff is smooth, and the subject reads at a glance for the first time. [[still](stills/12-breakthrough.jpg)]

## 05 · Jul 2026 — Along the edge, not across it

Ten months in, with honest metrics, OKLab loss, an MLX backend and four SVG recipes shipping — the render was still speckled. The cause was three lines old: splats elongated across edges instead of along them.

- **Before the orientation fix** — July 2026, 2,000 splats, full modern pipeline — and still speckled. Two of four splat-creation sites fed the structure tensor's gradient direction straight in as the major axis, elongating splats across edges instead of along them. LPIPS 0.449. [[still](stills/13-orientation-before.jpg)]
- **After the orientation fix** — Identical seed and budget, one π/2 rotation applied at every creation site via features.edge_tangent_angle(). Eyes, casque and mouth resolve. LPIPS 0.415, SSIM(sRGB) 0.701. [[still](stills/14-orientation-after.jpg)]

---

Stills are JPEG renders at 420 px; linked SVGs are the original
artifacts, unmodified. Early-era images come from the retired
`splat_this` pipeline and several use different source photographs,
so they are grouped chronologically rather than compared directly.
Only the final pair is a controlled before/after.
