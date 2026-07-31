# Pixel-Runtime Scaling MVP

**Status:** full-corpus 2k→4k comparison complete\
**Date:** 2026-07-29
**Scope:** emitted ImageData pixel-runtime HTML, full source frames, seed 0

## Question

Before adapting paper-style Top-K training to SVG or PowerPoint, how much of
the current fidelity gap is simply an inadequate Gaussian budget or incomplete
optimization?

The target was historically called Canvas, but it is the JavaScript software
renderer now named `pixel-runtime`: it implements the same linear-light
alpha-over equation used during MLX training and writes the result to
`ImageData`. It removes SVG, native Canvas-gradient, and DrawingML primitive
mismatch from the experiment.

## Measurement contract

- Train every arm at the stored corpus dimensions, up to a 384 px edge.
- Never score a crop as if it represented the image.
- Capture the emitted HTML in Google Chrome.
- Read the exact canvas pixel buffer through `canvas.toDataURL("image/png")`.
- Score the captured PNG, not the Python splat proxy.
- Report requested budget, initialized population, peak stage population, and
  final population after pruning separately.
- Report whole-frame SSIM, MS-SSIM, PSNR, LPIPS, worst 64 px ROI error, edge
  error, HTML bytes, training time, and browser render time.

## Correctness fix before scaling

The JavaScript and NumPy renderers clipped rotated anisotropic splats because
their raster bounds used `3*sx` by `3*sy` without rotation. MLX and Torch used
a conservative maximum-axis tile footprint and did not have that clipping
error.

The deployed and proxy renderers now use the rotated ellipse extents:

```text
extent_x = 3 * sqrt((sx*cos(theta))^2 + (sy*sin(theta))^2)
extent_y = 3 * sqrt((sx*sin(theta))^2 + (sy*cos(theta))^2)
```

After the fix, the full-frame chameleon effective-4k browser SSIM is 0.9438;
the final MLX acceptance SSIM is 0.9447. The remaining gap is about 0.0009.

## Existing full-corpus anchor

The requested-2k run uses the maximum-fidelity `1000,500,250` schedule and a
historical initialization cap of 1,200. Across the latest corrected browser
captures for all 21 images:

| Statistic | Result |
|---|---:|
| Median final splats | 1,395 |
| Median SSIM | 0.7751 |
| Median MS-SSIM | 0.8904 |
| Median LPIPS | 0.2443 |
| Images at SSIM >= 0.90 | 6 / 21 |
| Images at SSIM >= 0.95 | 3 / 21 |
| Images at SSIM >= 0.99 | 0 / 21 |
| Median HTML size | 226 KB |
| Median browser render | 59.2 ms |
| Total training time | 79.6 min |

This is a whole-corpus anchor, not evidence for a universal 2k ceiling. The
effective budget differs per image after densification and pruning.

## Full-frame chameleon calibration

| Arm | Initial | Peak | Final | SSIM | MS-SSIM | LPIPS | PSNR | Worst ROI | HTML | Train | Browser |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| requested 2k | 1,000 | — | 1,682 | 0.9140 | 0.9616 | 0.1284 | 30.46 | 0.0464 | 287 KB | 272 s | 65.8 ms |
| effective 4k | 2,000 | 3,680 | 3,025 | 0.9438 | 0.9776 | 0.0830 | 32.39 | 0.0368 | 524 KB | 519 s | 91.6 ms |
| effective 8k | 4,000 | 7,998 | 5,300 | 0.9631 | 0.9864 | 0.0582 | 34.34 | 0.0289 | 932 KB | 870 s | 136.3 ms |

The 4k row uses `--initial-splat-cap 4000`. An earlier requested-4k run retained
the profile's 1,200 initialization cap and finished with only 2,267 splats; it
is not an effective 4k density point. Browser times are medians of three fresh
page renders; the overview also shows the observed minimum and maximum.

## Scaling result and next gate

From effective 4k to 8k, chameleon improves by 0.0193 SSIM, 0.0088 MS-SSIM,
1.95 dB PSNR, 0.0248 LPIPS, and 0.0079 worst-ROI error. Final splats grow 75%,
HTML grows 78%, training grows 68%, and browser rendering grows 49%. The curve
is still improving, but 8k reaches 0.9631 rather than approaching 0.99.

The chameleon result justified an effective-4k run across the full 21-image
corpus. That run is now complete.

Do not claim a path to 0.99 from the chameleon curve alone; the high-frequency
corpus members remain the decisive test.

## Full-corpus 2k to effective-4k result

Every row uses the full stored frame, seed 0, the maximum-fidelity schedule,
and the exact Chrome canvas pixel buffer. The 4k arm initializes 2,000 splats
and permits growth to 4,000; final counts reflect checkpoint selection and
monotonic pruning.

| Statistic | Requested 2k | Effective 4k | Paired change |
|---|---:|---:|---:|
| Images | 21 | 21 | — |
| Median final splats | 1,395 | 2,382 | 1.74× |
| Median SSIM | 0.7751 | 0.8406 | +0.0498 |
| Median MS-SSIM | 0.8904 | 0.9282 | +0.0364 |
| Median LPIPS | 0.2443 | 0.1612 | -0.0850 |
| Median worst-ROI error | — | — | -0.0118 |
| SSIM improved | — | — | 21 / 21 |
| LPIPS improved | — | — | 21 / 21 |
| MS-SSIM improved | — | — | 21 / 21 |
| Worst ROI improved | — | — | 21 / 21 |
| Edge distance improved | — | — | 20 / 21 |
| Images at SSIM >= 0.90 | 6 | 10 | +4 |
| Images at SSIM >= 0.95 | 3 | 4 | +1 |
| Images at SSIM >= 0.99 | 0 | 0 | 0 |
| Median HTML size | 226 KB | 391 KB | 1.76× |
| Median browser render | 59 ms | 105 ms | 2.00× paired ratio |
| Median training | 3.6 min | 9.9 min | 2.77× paired ratio |
| Selected-run training total | 79.6 min | 212.5 min | 2.67× total |

The 4k browser-render p95 is 251 ms; Retina is the maximum at 344 ms. Training
p95 is 14.3 min; Cell is the maximum at 23.5 min. Moon is the only image whose
edge-chamfer score regresses, despite improving in SSIM, MS-SSIM, LPIPS, and
worst ROI.

### Per-image paired result

| Image | Final splats | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Browser | Training |
|---|---:|---:|---:|---:|---:|---:|---:|
| Checkerboard | 2,705 | 0.7784 | +0.2031 | 0.0630 | -0.1719 | 92 ms | 4.5 min |
| Gravel | 2,162 | 0.6615 | +0.1439 | 0.4382 | -0.0929 | 144 ms | 9.2 min |
| Grass | 2,212 | 0.5916 | +0.1292 | 0.4290 | -0.0997 | 123 ms | 9.9 min |
| Hubble deep field | 3,999 | 0.7039 | +0.1253 | 0.2575 | -0.1360 | 251 ms | 11.2 min |
| Astronaut | 2,812 | 0.8406 | +0.1246 | 0.1385 | -0.1149 | 76 ms | 7.2 min |
| Immunohistochemistry | 2,872 | 0.7552 | +0.1085 | 0.3541 | -0.0870 | 171 ms | 9.4 min |
| Stereo motorcycle | 2,382 | 0.8099 | +0.0897 | 0.2199 | -0.0850 | 105 ms | 10.0 min |
| Page | 1,756 | 0.7782 | +0.0743 | 0.3623 | -0.1295 | 68 ms | 9.5 min |
| Retina | 4,000 | 0.9292 | +0.0693 | 0.1316 | -0.1040 | 344 ms | 11.6 min |
| Coins | 1,926 | 0.8041 | +0.0632 | 0.1881 | -0.0636 | 80 ms | 9.9 min |
| Coffee | 2,303 | 0.8249 | +0.0498 | 0.1698 | -0.0529 | 71 ms | 6.4 min |
| Camera | 2,225 | 0.8112 | +0.0485 | 0.2724 | -0.0948 | 96 ms | 14.3 min |
| Chelsea | 3,057 | 0.9045 | +0.0456 | 0.1612 | -0.0830 | 63 ms | 5.8 min |
| Logo | 2,297 | 0.9583 | +0.0379 | 0.0424 | -0.0547 | 190 ms | 11.8 min |
| Chameleon | 3,025 | 0.9438 | +0.0298 | 0.0830 | -0.0454 | 92 ms | 8.6 min |
| Text | 1,649 | 0.9199 | +0.0297 | 0.0980 | -0.0457 | 63 ms | 6.2 min |
| Rocket | 2,423 | 0.9156 | +0.0283 | 0.1227 | -0.0651 | 127 ms | 12.3 min |
| Cell | 2,558 | 0.9699 | +0.0154 | 0.1477 | -0.0884 | 143 ms | 23.5 min |
| Moon | 2,149 | 0.9455 | +0.0117 | 0.2087 | -0.0566 | 71 ms | 11.0 min |
| Colorwheel | 2,874 | 0.9837 | +0.0086 | 0.0059 | -0.0057 | 211 ms | 12.2 min |
| Brick | 2,113 | 0.9804 | +0.0081 | 0.0311 | -0.0161 | 117 ms | 7.8 min |

## Monotonicity failures found by the corpus run

Two post-training operations could destroy a better canvas:

1. The fixed `alpha <= 0.03` postprocess cutoff removed collectively important
   low-alpha splats. Hubble fell from roughly 0.705 to 0.544 SSIM and Retina
   from roughly 0.930 to 0.880 in the internal deployed model.
2. A later densification stage could improve the combined training loss while
   degrading deployed SSIM/PSNR. An ungated Colorwheel run fell from a
   stage-1 SSIM around 0.979 to a browser SSIM of 0.9726.

Pixel runtime now has two default-on artifact-in-the-loop gates:

- score every stage checkpoint with the exact deployed canvas model and retain
  the best material SSIM/PSNR checkpoint, preferring fewer splats at equivalent
  quality;
- score the optimized and postprocessed populations with that same model and
  revert postprocessing when it exceeds the SSIM or PSNR tolerance.

The corrected Hubble output keeps 3,999 splats and reaches browser SSIM 0.7039
instead of 0.5443. The corrected Colorwheel output chooses stage 2, rejects
destructive pruning, and reaches browser SSIM 0.9837 instead of 0.9726.

## Decision

Effective 4k is a real, corpus-wide improvement, not a selected-image effect.
It is also not a route to automatic 0.99: no corpus image reaches 0.99, and the
difficult texture images remain between 0.59 and 0.70 SSIM.

Do not run 8k blindly across the corpus. The next canvas MVP should make the
checkpoint gate an early-stop controller and allocate 8k only to images whose
4k deployed curve is still improving enough to justify predicted tile-overlap,
browser, HTML, and training costs. Smooth/high-scoring images such as
Colorwheel and Brick should stop earlier; Grass, Gravel, Hubble, Checkerboard,
Page, and other difficult images are the candidates for selective 8k or mixed
primitives.
