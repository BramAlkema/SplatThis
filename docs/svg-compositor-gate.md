# SVG compositor gate

SVG and the mathematical splat renderer require opposite iteration orders.
The renderer consumes splats front-to-back and lets early layers claim
transmittance. SVG paints later elements on top, so an equivalent document
must emit the splat elements back-to-front while keeping every ellipse paired
with its own gradient.

SplatThis now makes that boundary explicit for the standard,
browser-compatible, scripted-matrix, palette-quantized, and blur recipes. The
pixel runtime is unchanged: it continues to evaluate front-to-back
transmittance directly.

## Fidelity policies

`standard` uses the existing density-aware gradient approximation with up to
eight stops and two-decimal opacity. `high` lowers the maximum opacity-curve
error to 0.005, permits up to nine stops, and writes four-decimal opacity. It
remains adaptive: simple and low-alpha splats do not automatically receive all
nine stops. SVG masks and global `linearRGB` interpolation are deliberately
absent; the Chameleon MVP found no defensible quality/cost trade-off for them.

The CLI exposes both policies:

```bash
splatthis input.png --format svg -o output.svg \
  --svg-gradient-quality high
```

Correct back-to-front order is the static default. `--svg-painter-order
legacy` exists only for reproducibility and compatibility checks.

## Max-fidelity accept-or-revert

The `max-fidelity` profile enables an artifact gate. It emits three bounded
candidates from the final unchanged splat population:

1. corrected back-to-front SVG with standard stops (the incumbent);
2. historical forward-order SVG with standard stops;
3. corrected back-to-front SVG with high stops.

The incumbent is what the default emitter ships. It was originally the
historical forward order, for monotonicity with the pre-correction era; on
the July 2026 corpus that biased the gate below either fixed corrected
choice (gate median 0.7111 SSIM against 0.7404/0.7483), because four images
kept legacy purely by incumbency. Legacy remains a candidate and now has to
win an image outright.

Chromium captures every candidate at native dimensions. The gate freezes
worst-error ROIs from the incumbent and compares SSIM, MS-SSIM, LPIPS, OKLab
delta-E, edge metrics, compressed size, and browser latency. A candidate that
violates any hard guard is reverted even if one headline metric improves. The
selected candidate and all rejected decisions are recorded under
`svg_compositor_gate` in the run manifest. If Chromium is unavailable, the
export falls back to the requested corrected policy and records that the gate
was unavailable.

Use `--no-svg-compositor-gate` for a deterministic policy without candidate
search, or `--svg-compositor-gate` to enable it on another profile.

## Corpus result

The full stored SVG corpus contains 21 images and 33 available seed
populations. Correct order with standard gradients improved median SSIM by
0.08513 and median LPIPS by 0.09097 at essentially unchanged gzip size. The
high policy added median SSIM 0.01019 and improved median LPIPS by 0.00441,
with 56.8% median gzip growth and 4.3% median capture-time growth.

Neither change was universally monotonic: standard order regressed Gravel and
Hubble by more than 0.002 SSIM, while high stops regressed two Colorwheel
seeds. This is why max-fidelity selects per artifact instead of making high
stops universal. On the 21 seed-0 populations, the simulated gate — run under the
original legacy incumbent — selected corrected-high 17 times and retained
legacy order four times. Its median moved from 0.5973 to 0.7111 SSIM and from
0.4023 to 0.2439 LPIPS; the four legacy retentions are what motivated moving
the incumbent to corrected-standard, which bounds the gate's floor at the
default emitter's 0.7404/0.2433.

The versioned summary is in
[`data/svg-compositor-corpus.json`](../data/svg-compositor-corpus.json). The
individual live SVGs, browser screenshots, and metric vectors are generated
under `./tmp/svg-compositor-corpus/` by
`tools/svg_order_compositor_mvp.py`.
