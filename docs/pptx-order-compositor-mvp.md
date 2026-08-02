# PowerPoint painter-order MVP

**Status:** full 21-image corpus complete; explicit production emission and
external artifact selection implemented. Superseded on the default question:
corrected back-to-front order became the CLI and library default in 0.2.6;
`--pptx-painter-order legacy` reproduces the historical stack.

The mathematical renderer consumes splats front-to-back. DrawingML uses a
painter stack in which later shapes sit above earlier shapes, so the equivalent
native-PPTX candidate reverses splats at the final shape-emission boundary.
When explicit splat layers are present, the emitter still writes the layer
groups background-to-foreground and reverses only the splats within each
group.

## Method

`tools/pptx_order_compositor_mvp.py` loads one stored final splat population
and emits two native, editable decks:

1. `legacy-order.pptx`, preserving the historical front-to-back shape order;
2. `corrected-order.pptx`, using back-to-front shape order.

Both decks use the same splat geometry, colors, opacity, gradient style,
background, and slide size. Microsoft PowerPoint 16.89.1 renders each deck in
slideshow mode twice. The complete native-size frames are then graded with
fixed worst-error ROIs from the legacy artifact and the full SSIM, MS-SSIM,
LPIPS, OKLab, edge, and worst-ROI metric vector. All four capture pairs were
byte-identical within each artifact.

## Results

| Image | Selected | Legacy SSIM | Corrected SSIM | Delta | Legacy LPIPS | Corrected LPIPS | Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Chameleon | corrected | 0.79311 | 0.83788 | +0.04476 | 0.26206 | 0.20847 | -0.05359 |
| Hubble Deep Field | legacy | 0.53768 | 0.52051 | -0.01717 | 0.37499 | 0.41717 | +0.04218 |

Chameleon also improved MS-SSIM, delta-E p95, edge-gradient error, and the
fixed worst ROI. Hubble regressed on every principal guarded metric; its edge
chamfer increased from 6.25 to 10.36. Reordering changed deck size by less than
30 bytes in either case.

## Full corpus result

The complete seed-0 PPTX corpus contains 21 images. Every legacy and corrected
native deck was freshly captured once in Microsoft PowerPoint. The two-image
pilot above used two repeats and established byte-identical captures for both
variants; the corpus timing pass does not claim repeat stability from a single
sample.

| Policy | Median SSIM | Median MS-SSIM | Median LPIPS | Median size |
|---|---:|---:|---:|---:|
| Legacy order | 0.60186 | 0.67135 | 0.37499 | 129,797 bytes |
| Corrected order | 0.62790 | 0.72106 | 0.32004 | 129,662 bytes |
| Per-image gate | 0.62790 | 0.72106 | 0.36351 | 129,662 bytes |

Corrected order produced a median SSIM delta of +0.02662 and a median LPIPS
delta of -0.03346. It improved SSIM by more than 0.002 on 19 images, regressed
by more than 0.002 only on Hubble, and was nearly neutral on Text. Median
delta-E p95, edge chamfer, edge-gradient error, and fixed worst-ROI error also
improved. Reordering changed median deck size by -9 bytes.

The strict full-vector gate selected corrected order for 14 images and retained
legacy for seven. Six of those seven still improved on SSIM and LPIPS but
violated a local edge or worst-ROI guard; Hubble was the only broad regression.
Consequently, the conservative gated median LPIPS is worse than making
corrected order universal, but it preserves the predeclared monotonic artifact
contract.

The result mirrors the SVG finding: correct painter semantics are a strong
candidate, but a trained population can be coupled to the historical artifact
compositor. Production integration should therefore retain the per-artifact
gate rather than silently switch every deck.

## Integrated boundary

Ordinary conversion now exposes both native DrawingML stacks without launching
desktop software:

```bash
splatthis input.png --format pptx -o legacy.pptx \
  --pptx-painter-order legacy
splatthis input.png --format pptx -o corrected.pptx \
  --pptx-painter-order back-to-front
```

The default remains `legacy`. The selected value is recorded as
`config.pptx_painter_order` in the run manifest. Corrected order reverses
splats within each explicit layer while preserving background-to-foreground
layer-group order.

The external real-PowerPoint runner emits both candidates, captures and grades
them, then atomically copies the accepted native deck to `selected.pptx`. A
resumed corpus run reuses completed metric reports and rematerializes a missing
or stale selected deck. This keeps GUI automation out of ordinary headless
`splatthis` while making the evidence-backed winner directly consumable.

## Reproduce

```bash
./venv/bin/python tools/pptx_order_compositor_mvp.py \
  result/corpus/images/chameleon.png \
  --splats-json result/corpus/runs/chameleon_pptx_s0_art/final.raw.json \
  --manifest result/corpus/runs/chameleon_pptx_s0_art/run_manifest.json \
  --output-dir ./tmp/pptx-order-compositor/chameleon \
  --repeats 2
```

The generated decks, captures, logs, metric report, and visual overview live
under `./tmp/pptx-order-compositor/<image>/`, including the accepted
`selected.pptx`. PowerPoint—not LibreOffice or an internal proxy—is the
governing renderer.

Run the complete corpus with:

```bash
./venv/bin/python tools/pptx_order_compositor_corpus.py \
  --output-root ./tmp/pptx-order-compositor-corpus \
  --summary-output data/pptx-order-compositor-corpus.json \
  --repeats 1 --force
```

The versioned result vector is in
[`data/pptx-order-compositor-mvp.json`](../data/pptx-order-compositor-mvp.json).
The complete corpus vector is in
[`data/pptx-order-compositor-corpus.json`](../data/pptx-order-compositor-corpus.json).
