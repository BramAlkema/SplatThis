# Layered Saliency Passes Specification

**Status:** Draft  
**Date:** 2026-05-23  
**Related:** `docs/ADAPTIVE_SPLATS_SPEC.md`, `docs/adr-002-png2splat-python-pipeline.md`

## Summary

The converter should reconstruct images as ordered layers of splats instead of one undifferentiated splat cloud. A low-frequency base layer covers the full canvas first. Salient object, detail, and edge layers are then trained on top, using saliency and residual error to spend splats where they matter.

This turns the "low-grade background pass" from a smoke-test artifact into an explicit architecture:

```text
background/base -> object mass -> salient detail -> edge/specular detail
```

The output remains one selectable artifact in PPTX, but internally it can expose nested groups for each layer.

## Problem

Single-cloud optimization makes the same splat budget compete across unrelated jobs:

- broad background coverage
- foreground object mass
- facial and animal detail
- sharp edges and highlights
- export-specific constraints such as SVG/DrawingML gradient fidelity

The native-size smoke test showed the failure mode clearly. Sparse detail-biased splats on a black/default background looked like a star field. Once broad non-salient coverage was made explicit, the same low-budget run became recognizable.

The next step is to make this layered behavior deliberate and configurable.

## Goals

- Allocate splats by image role, not only by global error.
- Keep backgrounds covered with broad, low-cost splats.
- Reserve small/high-cost splats for salient foreground and residual detail.
- Train layers in a stable order so later detail layers do not destabilize base coverage.
- Export layer structure to PPTX as real DrawingML groups, not raster images.
- Record per-layer budgets, metrics, and counts in the run manifest.

## Non-Goals

- Perfect semantic segmentation.
- Replacing current profiles in one rewrite.
- Making PPTX visually identical to the canvas renderer at high overlap counts.
- Making every splat individually pleasant to edit by hand. The useful editing unit is the layer group.

## Layer Model

### L0: Base / Non-Salient Layer

Purpose:
- cover the entire canvas
- approximate low-frequency color fields, shadows, and background gradients
- prevent transparent/black/white fallback artifacts

Placement:
- mostly stratified full-canvas positions
- optional downweighting of salient foreground centers
- very broad `sigma`, computed from native pixel area and base splat count

Training target:
- blurred or low-frequency target
- background-safe and low-saliency regions weighted highest
- foreground detail explicitly downweighted

Typical settings:
- `base_layer_fraction`: high for short budgets
- large `coverage_sigma_max`
- moderate alpha
- no residual micro-splats

### L1: Object Mass Layer

Purpose:
- recover large foreground masses and silhouettes
- add medium-scale color fields for people, animals, products, etc.

Placement:
- foreground/saliency mask sampling
- color-distance and dog/blur saliency sampling
- medium isotropic or mildly anisotropic splats

Training target:
- residual after L0 render
- foreground-weighted but not edge-only

### L2: Salient Detail Layer

Purpose:
- recover faces, eyes, glasses, dog markings, hands, text-like features, and other high-value detail

Placement:
- residual error multiplied by saliency
- edge band and foreground masks are weighted up
- smaller splats, stronger anisotropy from local structure

Training target:
- residual after L0 + L1
- full-resolution target
- foreground and edge weights high

### L3: Edge / Specular Layer

Purpose:
- add hard contours, highlights, tiny high-contrast marks, and final corrective detail

Placement:
- high residual, high gradient, high local-contrast patches
- small anisotropic splats aligned with structure tensor

Training target:
- final residual
- optional and skipped for smoke/short budgets

## Prepass

The prepass should produce a `LayerGuidance` bundle:

```python
LayerGuidance:
    background_linear_rgb: np.ndarray[3]
    foreground_mask: bool[H, W]
    background_safe_mask: bool[H, W]
    edge_band_mask: bool[H, W]
    saliency: float32[H, W]
    low_frequency: float32[H, W, 3]
    high_frequency: float32[H, W, 3]
    summary: dict
```

Suggested saliency terms:

```text
saliency = 0.40 * color_distance_from_border_background
         + 0.30 * difference_of_gaussians_lightness
         + 0.20 * edge_density
         + 0.10 * local_chroma_or_contrast
```

The existing region guidance masks are the first version of this prepass. Layered mode should extend them rather than introduce a parallel segmentation system.

## Budget Allocation

Total budget comes from the existing budget resolver:

```text
total_splats = megapixels * splats_per_mp * saliency_multiplier
```

Layer allocation then splits the total:

| Budget | L0 Base | L1 Mass | L2 Detail | L3 Edge |
| --- | ---: | ---: | ---: | ---: |
| `1m/smoke` | 80% | 20% | 0% | 0% |
| `5m` | 55% | 25% | 20% | 0% |
| `10m` | 40% | 25% | 25% | 10% |
| `30m` | 30% | 25% | 30% | 15% |

These are starting points. The implementation should allow profile overrides and manifest the resolved counts.

## Training Schedule

Layered training should be sequential with optional final joint polish:

1. Build guidance maps.
2. Initialize L0 from stratified/base coverage.
3. Train L0 against low-frequency target.
4. Freeze or heavily damp L0.
5. Render current stack and compute residual.
6. Initialize L1 from foreground/mass saliency.
7. Train L1 against residual-weighted target.
8. Initialize L2 from residual * saliency * edge weights.
9. Train L2.
10. Optionally initialize/train L3.
11. Optional short joint polish with low learning rate.

Layer-specific renderer order is fixed:

```text
L0 -> L1 -> L2 -> L3
```

Within a layer, existing importance ordering can remain.

## Data Model

Current splats should gain optional metadata without breaking raw schema compatibility:

```python
SplatLayer:
    id: int                # 0, 1, 2, 3
    name: str              # "base", "mass", "detail", "edge"
    order: int             # render order
    locked: bool           # trainable in current phase
```

Raw JSON should include a `layers` block and per-splat optional `layer` field:

```json
{
  "layers": [
    {"id": 0, "name": "base", "order": 0, "count": 152},
    {"id": 1, "name": "mass", "order": 1, "count": 45}
  ],
  "splats": [
    {"x": 1.0, "y": 2.0, "sx": 10.0, "sy": 10.0, "theta": 0.0, "r": 0.1, "g": 0.2, "b": 0.3, "a": 0.5, "layer": 0}
  ]
}
```

If a splat has no layer metadata, treat it as `layer=0` for backwards compatibility.

## Export Contract

### PPTX / DrawingML

Default PPTX export should remain native DrawingML, not raster media.

Group hierarchy:

```text
Splat Group
  Background Layer
    Splat Background
    L0 splat ellipses
  Mass Layer
    L1 splat ellipses
  Detail Layer
    L2 splat ellipses
  Edge Layer
    L3 splat ellipses
```

Acceptance:
- no `ppt/media/image1.png`
- no `<p:pic>` in slide XML
- one top-level `Splat Group`
- one child group per non-empty layer
- group order matches render order

### SVG

SVG should mirror the layer hierarchy with `<g id="layer-base">`, etc. This makes browser inspection and debugging simpler.

### Canvas

Canvas export should sort by `(layer_order, importance)` before rendering.

## Manifest

The run manifest should include:

```json
{
  "layered_saliency": {
    "enabled": true,
    "layers": [
      {
        "id": 0,
        "name": "base",
        "target": "low_frequency",
        "allocated_splats": 152,
        "final_splats": 152,
        "iterations": 2,
        "coverage": 1.0
      }
    ]
  }
}
```

Each stage artifact should include `layer_id` and `layer_name` when applicable.

## CLI / API

Initial CLI:

```bash
splatlify input.jpg --format pptx --time-budget 10m --layered-saliency
```

Flags:

- `--layered-saliency`: enable layered training/export.
- `--no-layered-saliency`: force legacy single-cloud behavior.
- `--layer-budget base=0.5,mass=0.25,detail=0.2,edge=0.05`: optional override.
- `--export-layer-groups/--no-export-layer-groups`: control nested layer groups for vector outputs.

API:

```python
PNG2SVGConverter(
    time_budget="10m",
    layered_saliency=True,
    layer_budget={"base": 0.40, "mass": 0.25, "detail": 0.25, "edge": 0.10},
)
```

## Acceptance Tests

### Unit

- Layer budget resolver normalizes fractions and respects total splat cap.
- Layered sort order is stable: base before mass before detail before edge.
- Raw JSON round-trips optional layer metadata.
- PPTX export contains top-level `Splat Group` and child layer groups.
- PPTX export contains no media PNG and no `<p:pic>`.

### Integration

Run native man/dog smoke:

```bash
splatlify tmp/test_harvard.jpg \
  --format pptx \
  --output tmp/man_dog_layered_smoke.pptx \
  --time-budget smoke \
  --layered-saliency \
  --artifacts-dir tmp/man_dog_layered_smoke_artifacts
```

Expected:
- runtime stays under the smoke target plus small packaging overhead
- coverage remains near `1.0`
- output is recognizable, not star-field sparse
- `Splat Group` exists
- `Background Layer` and at least one foreground/detail group exists when budget allows

### Regression

Compare layered against current single-cloud budgeted runs:

- smoke: should avoid black/empty fallback and maintain full coverage
- 5m/10m: should improve foreground detail at similar or lower splat count
- 30m: should not regress global SSIM/PSNR beyond tolerance while improving salient-region metrics

## Implementation Plan

### Slice 1: Metadata and Ordering

- Add optional `layer` metadata to splat serialization.
- Add layer-aware export ordering.
- Keep existing behavior for splats without metadata.

### Slice 2: PPTX/SVG Layer Groups

- Extend DrawingML export to emit child groups by layer.
- Extend SVG export to emit `<g>` by layer.
- Add package tests for group hierarchy.

### Slice 3: Layer Budget Resolver

- Split resolved total splats into layer budgets based on time budget/profile.
- Record budget plan in manifest.

### Slice 4: L0 Base Layer

- Formalize low-frequency target.
- Train base layer against low-frequency/background-weighted objective.
- Freeze/damp L0 before later layers.

### Slice 5: L1/L2 Residual Layers

- Add mass and detail layer initialization from saliency/residual maps.
- Train sequentially and record per-layer metrics.

### Slice 6: Optional L3 Edge Layer and Joint Polish

- Add edge/specular micro-layer.
- Add low-rate final polish with layer-specific learning-rate multipliers.

## Open Questions

- Should `layer` live on `GaussianSplat` directly or only in raw/export metadata?
- Should base layer use the same alpha-over renderer or a separate blurred target prefit?
- How much layer hierarchy should SVG expose by default?
- Should PowerPoint layer groups be locked/protected where supported, or simply grouped?
