# Schedule length is not saturated at the default

Measured 2026-08-03 on two corpus images, SVG output, seed 0, governing
native-size Playwright Chromium captures, run through
`tools/corpus_benchmark.py` (tags `stages-2x`, `stages-4x`) against the
existing `v2-governing-aug2026` default-schedule rows.

The `max-fidelity` profile's `1000,500,250` is already the "much longer run"
an older note in `result/README.md` contrasted against a shorter default.
The open question was therefore not long-versus-default but whether the
default has converged. It has not.

| image | schedule | splats | artifact | LPIPS | Δ | SSIM | Δ | training |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| astronaut | `1000,500,250` | 1,617 | 1,583 KB | 0.2304 | — | 0.7030 | — | 6.6 min |
| astronaut | `2000,1000,500` | 1,645 | 1,615 KB | 0.1888 | −0.0416 | 0.7512 | +0.0482 | 14.5 min |
| astronaut | `4000,2000,1000` | 1,623 | 1,598 KB | **0.1708** | **−0.0596** | **0.7796** | **+0.0766** | 40.1 min |
| stereo_motorcycle | `1000,500,250` | 1,333 | 1,295 KB | 0.3144 | — | 0.6765 | — | 5.6 min |
| stereo_motorcycle | `2000,1000,500` | 1,333 | 1,301 KB | 0.2914 | −0.0230 | 0.7024 | +0.0259 | 14.4 min |
| stereo_motorcycle | `4000,2000,1000` | 1,341 | 1,308 KB | **0.2709** | **−0.0435** | **0.7185** | **+0.0420** | 27.1 min |

## What makes this different from the splat-count lever

**The artifact does not grow.** Splat counts move by under 2% and file sizes
by under 2%; stereo_motorcycle's 2x run has the identical 1,333 splats as
its default. This was the first thing checked, because a longer schedule
running more densification cycles would have made this a measurement of
splat count in disguise -- already the known big lever. It is not. The same
population simply lands in better places.

So this is quality at zero delivery cost: −0.060 LPIPS on astronaut for the
same bytes, the same shape count, and the same render time in the browser.
Splat count, by contrast, buys quality by spending size and render cost.

## Cost and the sensible operating point

Gains are sublinear and the cost is worse than linear. The first doubling
buys roughly two-thirds of the total gain for about a third of the extra
wall clock, which makes **2x the sensible operating point** and 4x a
deliberate choice for a hero artifact.

Both metrics move together and both far exceed the 0.029 SSIM seed noise
floor, so the direction is not in question on these two images.

## What this does not establish

Two images, one format, one seed. Texture-limited content (`grass`,
`gravel`, `page`) may be entropy-bound rather than schedule-bound and could
well flatten immediately -- they are the images that would decide whether a
profile change is worth it. Nothing here has been tried on PPTX or CSS. And
the 97-99% of wall clock that sits in the optimizer loop is what makes this
expensive to explore: repeated training on one image currently re-fits from
scratch because there is no warm-start path, which is the obvious enabler
for a corpus-wide schedule sweep.
