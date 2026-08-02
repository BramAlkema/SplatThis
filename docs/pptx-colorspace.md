# What color space does PowerPoint composite in?

Measured 2026-08-02 on real Microsoft PowerPoint (macOS slideshow capture),
with `tools/pptx_colorspace_probe.py`: one synthetic probe deck whose patch
midpoints distinguish linear-light from display-sRGB math, classified
against calibration swatches from the same capture, plus a corpus-wide tally
of both internal compositing models against the 21 real captures. Raw data:
`tmp/pptx-colorspace-probe/results.json` (regenerable in one 20-second
capture).

## Verdicts

**1. Gradient color ramps interpolate in linear light.** The red-to-green
`gradFill` midpoint measured within one 8-bit step of the linear-light
prediction (distance 0.010 vs 0.129 for sRGB interpolation). This is the
opposite of browsers, which interpolate gradient stops in display sRGB.

**2. Alpha compositing happens in display sRGB.** A 50%-alpha shape over a
backdrop and a black-to-transparent alpha ramp over white both landed on the
sRGB-blend prediction (the alpha ramp within a thousandth). This matches
browsers.

**3. PowerPoint is therefore a hybrid compositor** — linear gradient ramps
feeding sRGB alpha-over — that neither of this project's training models
describes. The corpus tally agrees: rendering the same populations under
pure-linear vs pure-sRGB compositing and comparing with the real captures
splits 9 / 12 across the 21 images. This also explains the sRGB-training
MVP's mixed result (`tools/pptx_srgb_training_mvp.py`: chameleon clearly
worse, text clearly better, ΔE-p95 better on all four): each pure model
matches one half of the real compositor.

**4. The capture chain desaturates every PowerPoint score.** The probe's
opaque calibration swatches came back as (234, 51, 35) for `FF0000` and
(117, 251, 76) for `00FF00` — sRGB primaries re-expressed in Display P3
coordinates. The macOS screenshot records the P3 framebuffer and the scoring
pipeline reads those numbers as sRGB, so every PowerPoint capture is scored
with systematically desaturated primaries (grays are unaffected). Part of
"PowerPoint renders less vibrant" is this measurement bias, not PowerPoint.
The probe's verdicts are unaffected because its predictions were computed
from the captured swatches themselves.

## Implications

- **Measurement first:** the PowerPoint capture path should convert the
  screenshot through its embedded ICC profile to sRGB before scoring. That
  is a governing-protocol change — all PPTX fidelity numbers (ΔE above all)
  shift when it lands, so it must be applied once, re-captured once, and
  republished through the ledgers, not patched quietly.
- **Training second:** a faithful PPTX training proxy is the measured
  hybrid — interpolate the per-splat gradient ramp in linear light, apply
  alpha-over in sRGB. Neither the current linear default nor the
  `--training-export-target svg` sRGB proxy is right, which the MVP's split
  decision already suggested.
- **The DrawingML skirt-washing remains real** regardless: `gradFill` cannot
  separate color from alpha the way the CSS mask does, so a residual gap to
  the browser targets is expected even after both fixes.

## Postscript: the fix landed (same day)

The capture path now converts every screenshot through its embedded ICC
profile to sRGB before cropping (`tools/full_corpus_mvp._screen_to_srgb`),
and the whole corpus was re-captured and re-scored under the corrected
protocol. Probe calibration swatches read pure (255, 0, 0) / (0, 255, 0)
afterwards, and all three compositor verdicts held with sharper separation.

Two sobering follow-ups. The headline PPTX medians barely moved (SSIM
0.6279 unchanged, LPIPS 0.3200 -> 0.3212): the bias lived in the color axis
that SSIM and LPIPS mostly ignore, so the published quality story survives,
and color metrics are now fair. And the sRGB-training MVP, re-scored on
unbiased captures, reproduced its split verdict almost exactly -- so the
linear training default stays, chosen on clean evidence. Note the probe's
linear-gradient verdict concerns *color* ramps; this project's splats are
constant-color with alpha ramps, so the compositing-space story does not by
itself explain the remaining per-image differences between proxies. That
residue -- stop quantization, rasterization, and fit interaction -- is the
open question a future PPTX proxy should target.

## Postscript 2: the residual is not the compositing space either

Follow-up measurement (2026-08-02) puts these findings in proportion.
Against the same real captures, the plain Gaussian model scores LPIPS
0.0959 in linear and 0.1082 in sRGB, and the exact feathered ring-stack
model scores 0.0843 — so choosing the right space, or the right primitive,
moves deployed agreement by hundredths while the floor stays near 0.08.
Misregistration (~0.002) and 8-bit per-step accumulation (~0.005) are
excluded as major causes. Whatever remains is the dominant term in every
PowerPoint fidelity number this project publishes, and no training-target
or primitive change addresses it. Identifying it — most plausibly
PowerPoint's shape rasterization, or a remaining capture-chain loss —
should precede further PPTX proxy work.
