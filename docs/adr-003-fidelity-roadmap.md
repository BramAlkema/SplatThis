# ADR-003: Maximum-Fidelity Reconstruction and Export Stage

**Status**: Proposed  
**Date**: 2026-07-28  
**Revision**: 2  
**Authors**: SplatThis Development Team

## Summary

Add a first-class, opt-in **fidelity stage** after normal splat optimization and
before final export. The stage generates bounded candidate corrections, renders
the real deployment artifact where possible, and keeps a candidate only when it
is measurably better and passes hard regression gates.

The stage combines:

1. multi-scale and local perceptual optimization;
2. residual topology analysis;
3. typed structural, photographic, and corrective splat proposals;
4. split, merge, recolor, reshape, and local reorder operations;
5. export-space post-fitting;
6. actual SVG rasterization in the acceptance loop;
7. calibrated PowerPoint proxy fitting plus offline real-PowerPoint calibration;
8. optional mixed primitives and a strictly bounded raster residual in an
   explicit hybrid mode.

The core rule is:

> Every fidelity operation is speculative. Evaluate the resulting deployed
> artifact, keep a proven gain, and otherwise restore the previous best state.

## Context

SplatThis already implements much of the foundation required for high fidelity:

- anisotropic `GaussianSplat` primitives;
- content-adaptive and edge-aware initialization;
- layered saliency and region-guidance maps;
- staged optimization and residual-detail passes;
- Torch and MLX renderers with parity tests;
- OKLab L1, SSIM, gradient, and spatially weighted losses;
- export-target-aware sRGB/source-over training;
- SVG recipes and native DrawingML output;
- actual SVG rasterization through CairoSVG or `rsvg-convert`;
- run manifests, time-budget profiles, and quality metrics.

Those capabilities remain the normal reconstruction path. This ADR does not
replace them.

The remaining fidelity ceiling comes from gaps between ordinary differentiable
training and the artifact a person finally sees:

1. The active structural term is global SSIM, which can miss local blur,
   edge displacement, ringing, and small high-value features.
2. Residual-detail logic distinguishes edges from general residuals, but does
   not yet model the full residual topology or choose among correction
   operators.
3. Training proxies approximate browser and PowerPoint behavior; only SVG is
   currently easy to rasterize in the normal pipeline.
4. Gaussian-only correction can spend many shapes on a feature better
   represented by a ridge, stroke, flat patch, or tiny residual tile.
5. Compositing order is discrete and can remain locally wrong even when all
   continuous splat parameters are well optimized.
6. A scalar average metric can hide severe regressions in a face, silhouette,
   text-like edge, or saturated accent.
7. Profiles contain many hand-tuned thresholds, but candidate changes are not
   governed by one monotonic, artifact-level accept-or-revert policy.

## Decision Drivers

In descending priority:

1. Fidelity of the **deployed artifact**, not merely the internal renderer.
2. No silent visual regression.
3. Preserve editable SVG and DrawingML as the default output contract.
4. Deterministic, inspectable decisions and artifacts.
5. Torch/MLX semantic parity.
6. Bounded splat count, file size, memory, and runtime.
7. Incremental integration with the current `PNG2SVGConverter`.

Maximum fidelity is allowed to be slower. Performance remains a measured
constraint, not the objective of this stage.

## Decision

### Pipeline placement

The canonical pipeline becomes:

```text
load and normalize
  -> region guidance
  -> initialize layers
  -> staged differentiable optimization
  -> densify and residual-detail passes
  -> fidelity stage
       1. establish deployed-artifact baseline
       2. analyze residual topology
       3. generate bounded candidate corrections
       4. optimize continuous parameters
       5. test discrete order and recipe candidates
       6. emit and rasterize actual artifact
       7. accept gain or restore previous best
  -> final export
  -> manifest and validation artifacts
```

The stage is disabled by default until its benchmark gates pass. It is enabled
through a profile or explicit configuration:

```text
--fidelity-stage off|balanced|max
```

`max` may consume the remaining time budget. It may not silently exceed explicit
splat-count, file-size, or hybrid-output constraints.

### Stage contract

The fidelity stage receives immutable run context and a copy of the best splat
state produced by the ordinary pipeline. It returns the winning state and a
complete decision trace.

```python
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np


FidelityMode = Literal["off", "balanced", "max"]
ExportTarget = Literal["canvas", "svg", "pptx-softedge"]


@dataclass(frozen=True)
class FidelityConfig:
    mode: FidelityMode = "off"
    export_target: ExportTarget = "svg"
    max_passes: int = 4
    max_candidates_per_pass: int = 12
    max_added_splats: int = 0
    max_file_size_bytes: int | None = None
    min_lpips_gain: float = 0.001
    max_ssim_regression: float = 0.002
    max_edge_regression: float = 0.002
    supersample: int = 2
    allow_mixed_primitives: bool = False
    allow_raster_residual: bool = False


@dataclass(frozen=True)
class FidelityContext:
    target_linear_rgb: np.ndarray
    guidance: dict[str, np.ndarray]
    fixed_rois: tuple[tuple[int, int, int, int], ...]
    output_format: str
    recipe: str
    seed: int
    work_dir: Path


@dataclass(frozen=True)
class FidelityCandidate:
    name: str
    splats: tuple["GaussianSplat", ...]
    recipe_overrides: dict[str, float] = field(default_factory=dict)
    auxiliary_layers: tuple[object, ...] = ()


@dataclass(frozen=True)
class FidelityResult:
    winner: FidelityCandidate
    baseline_metrics: "FidelityMetrics"
    final_metrics: "FidelityMetrics"
    decisions: tuple[dict[str, object], ...]
```

The stage must not mutate the converter's persistent configuration. This follows
the existing per-run state-isolation rule in `convert()`.

### Monotonic candidate loop

All operators use the same evaluator and gate:

```python
class CandidateOperator(Protocol):
    name: str

    def propose(
        self,
        best: FidelityCandidate,
        analysis: "ResidualAnalysis",
        context: FidelityContext,
        limit: int,
    ) -> Sequence[FidelityCandidate]: ...


class FidelityStage:
    def __init__(self, config, evaluator, operators):
        self.config = config
        self.evaluator = evaluator
        self.operators = tuple(operators)

    def run(self, baseline, context):
        best = baseline
        best_metrics = self.evaluator.evaluate(best, context)
        baseline_metrics = best_metrics
        decisions = []

        for pass_index in range(self.config.max_passes):
            analysis = analyze_residual(best, context, self.evaluator)
            improved = False

            for operator in self.operators:
                proposals = operator.propose(
                    best,
                    analysis,
                    context,
                    limit=self.config.max_candidates_per_pass,
                )
                for candidate in proposals:
                    metrics = self.evaluator.evaluate(candidate, context)
                    accepted, reason = accept_candidate(
                        baseline=baseline_metrics,
                        incumbent=best_metrics,
                        candidate=metrics,
                        config=self.config,
                    )
                    decisions.append(
                        {
                            "pass": pass_index,
                            "operator": operator.name,
                            "candidate": candidate.name,
                            "accepted": accepted,
                            "reason": reason,
                            "metrics": metrics.as_dict(),
                        }
                    )
                    if accepted:
                        best, best_metrics = candidate, metrics
                        improved = True

            if not improved:
                break

        return FidelityResult(
            winner=best,
            baseline_metrics=baseline_metrics,
            final_metrics=best_metrics,
            decisions=tuple(decisions),
        )
```

Evaluation is intentionally outside each operator. An operator cannot redefine
success to favor its own output.

## Quality Model

### Measure the right image

The evaluator produces two metric sets:

1. **proxy metrics** from the differentiable renderer, used for fast rejection;
2. **deployed metrics** from the emitted and rasterized artifact, used for final
   acceptance.

For SVG, deployed metrics must use an actual rasterizer. A proxy fallback may
be recorded, but a `max`-fidelity gain cannot be declared from a fallback.

For PPTX, the normal run uses the calibrated soft-edge proxy. Real PowerPoint
captures are maintained as offline calibration fixtures because PowerPoint is
not a suitable runtime dependency. LibreOffice is not a visual oracle for PPTX.

### Metric vector

No single metric is authoritative. Track:

```python
@dataclass(frozen=True)
class FidelityMetrics:
    lpips: float                 # lower is better
    psnr_srgb: float             # higher is better
    ssim_srgb: float             # higher is better
    ms_ssim_luma: float          # higher is better
    delta_e_ok_mean: float       # lower is better
    delta_e_ok_p95: float        # lower is better
    edge_chamfer: float          # lower is better
    edge_gradient_l1: float      # lower is better
    salient_lpips: float         # lower is better
    worst_roi_error: float       # lower is better
    splat_count: int
    file_size_bytes: int
    render_method: str

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()
```

Interpretation:

- LPIPS is the primary whole-image perceptual measure.
- `salient_lpips` prevents broad easy areas from hiding foreground regressions.
- local/windowed MS-SSIM catches blur that global SSIM may reward.
- OKLab delta-E catches chroma and lightness drift.
- edge chamfer catches displaced silhouettes even when pixel averages improve.
- `worst_roi_error` protects the single worst important region.
- file size and splat count are constraints, not fidelity rewards.

### Region-of-interest evaluation

Evaluate at least:

- full image;
- foreground;
- edge band;
- high-saliency regions;
- background-safe regions;
- the worst fixed-size residual windows.

The worst windows are selected deterministically from the baseline and remain
fixed while candidates are compared. Otherwise, each candidate could be judged
on a different set of easy or hard crops.

```python
def select_fixed_rois(error_map, saliency, size=64, count=8):
    priority = error_map * (1.0 + saliency)
    rois = []
    suppressed = priority.copy()
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(suppressed), suppressed.shape)
        rois.append(centered_crop(x=x, y=y, size=size, shape=priority.shape))
        suppress_neighborhood(suppressed, x=x, y=y, radius=size // 2)
    return tuple(rois)
```

### Acceptance gate

The gate is Pareto-like: require a meaningful primary gain, forbid important
regressions, and enforce resource limits.

```python
def accept_candidate(*, baseline, incumbent, candidate, config):
    if candidate.render_method.startswith("proxy-fallback"):
        return False, "no deployed-artifact render"
    if config.max_file_size_bytes is not None:
        if candidate.file_size_bytes > config.max_file_size_bytes:
            return False, "file-size budget exceeded"
    if candidate.ssim_srgb < baseline.ssim_srgb - config.max_ssim_regression:
        return False, "SSIM hard gate"
    if candidate.edge_chamfer > baseline.edge_chamfer + config.max_edge_regression:
        return False, "edge hard gate"
    if candidate.worst_roi_error > baseline.worst_roi_error * 1.01:
        return False, "worst-ROI hard gate"

    lpips_gain = incumbent.lpips - candidate.lpips
    salient_gain = incumbent.salient_lpips - candidate.salient_lpips
    delta_e_gain = incumbent.delta_e_ok_p95 - candidate.delta_e_ok_p95

    meaningful_gain = (
        lpips_gain >= config.min_lpips_gain
        or salient_gain >= config.min_lpips_gain
        or delta_e_gain >= 0.25
    )
    return (
        (True, "measured fidelity gain")
        if meaningful_gain
        else (False, "gain below threshold")
    )
```

Thresholds are benchmark parameters, not universal constants. They must be
derived from repeatable rasterization noise and the reference corpus.

## Fidelity Operations

### 1. Multi-scale, local, export-aware optimization

Retain OKLab L1, luminance gradients, and export-target compositing. Add local
structure and a scale curriculum.

The optimization objective is:

```text
L = w_color     * OKLab robust L1
  + w_structure * local MS-SSIM on OKLab L
  + w_edge      * multi-scale luminance-gradient loss
  + w_laplacian * Laplacian-pyramid loss
  + w_roi       * salient/worst-ROI loss
  + w_regular   * parameter regularization
```

Global SSIM remains available as a cheap term but is not sufficient for the
maximum-fidelity profile.

```python
import torch
import torch.nn.functional as F


def downsample_hwc(image: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return image
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(
        nchw,
        scale_factor=1.0 / scale,
        mode="area",
    )[0].permute(1, 2, 0)


def pyramid_charbonnier(
    rendered: torch.Tensor,
    target: torch.Tensor,
    scales=(8, 4, 2, 1),
    weights=(0.10, 0.20, 0.30, 0.40),
    epsilon=1e-3,
) -> torch.Tensor:
    total = rendered.new_zeros(())
    for scale, weight in zip(scales, weights):
        diff = downsample_hwc(rendered, scale) - downsample_hwc(target, scale)
        total = total + weight * torch.sqrt(diff.square() + epsilon**2).mean()
    return total
```

Optimization uses a curriculum:

1. low-frequency/color correction;
2. local structure and edges;
3. full-resolution residual;
4. short export-proxy post-fit;
5. actual-artifact candidate evaluation.

Do not enable all terms at full weight from iteration zero. That encourages
small splats to chase texture before large forms and colors are correct.

### 2. Supersampled training and downsampling

For `max`, render at 2× resolution during the final polish and downsample with a
known filter before loss evaluation. This reduces aliasing and makes subpixel
position, scale, and rotation gradients more informative.

```python
def supersampled_forward(renderer_2x, splats, target_shape):
    high = renderer_2x(splats)
    nchw = high.permute(2, 0, 1).unsqueeze(0)
    low = F.interpolate(
        nchw,
        size=target_shape,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return low[0].permute(1, 2, 0)
```

The downsampling filter used for training must be recorded in the manifest and
matched in Torch and MLX. Supersampling is a late-stage option because it
increases renderer cost and tile pressure.

### 3. Residual topology analysis

Analyze why a region is wrong before adding a splat. Produce a shared bundle:

```python
@dataclass(frozen=True)
class ResidualAnalysis:
    residual_oklab: np.ndarray
    absolute_color_error: np.ndarray
    low_frequency_error: np.ndarray
    high_frequency_error: np.ndarray
    target_edges: np.ndarray
    rendered_edges: np.ndarray
    edge_displacement: np.ndarray
    coverage_error: np.ndarray
    opacity_order_error: np.ndarray
    priority: np.ndarray
    fixed_rois: tuple[tuple[int, int, int, int], ...]
```

Classify high-priority components:

| Residual type | Evidence | Preferred correction |
| --- | --- | --- |
| broad color field | low-frequency error, weak edge | recolor/grow/merge broad splats |
| uncovered region | target mass with high transmittance | add coverage splat |
| edge displacement | close parallel target/render edges | move/rotate/reshape structural splat |
| missing edge | target edge without rendered match | add anisotropic tangent splat or ridge |
| excess blur | edge present but gradient too weak | shrink/split/reduce alpha |
| texture deficit | high-frequency residual inside object | micro-splats or texture primitive |
| chroma drift | high OKLab a/b error, low L error | recolor local contributing splats |
| halo | sign-changing residual around edge | adjust scale/alpha/export blur |
| overlap/order error | correct colors, wrong occlusion pattern | local reorder or alpha rebalance |
| exporter mismatch | proxy good, artifact bad | recipe/post-fit correction |

This extends the current edge/general residual distinction. It must reuse the
existing region-guidance maps instead of introducing a parallel segmentation
system.

### 4. Typed splat populations

Use stage-local roles:

```python
SplatRole = Literal[
    "base",
    "mass",
    "structural",
    "texture",
    "corrective",
]
```

Roles control proposal and optimization behavior:

| Role | Geometry | Main loss | Typical operations |
| --- | --- | --- | --- |
| base | broad, low anisotropy | low-frequency color | recolor, grow, merge |
| mass | medium | silhouette and regional color | move, reshape, recolor |
| structural | anisotropic, edge-tangent | edge distance and gradient | move, rotate, split |
| texture | small | high-frequency/LPIPS | add, recolor, prune |
| corrective | tightly bounded | actual export residual | local post-fit, remove if no gain |

The first implementation stores roles only in fidelity-stage metadata. The raw
`GaussianSplat` schema and current exporters remain unchanged.

### 5. Operator portfolio

The fidelity stage may propose:

- **recolor** splats contributing to a chroma residual;
- **move** a splat toward a matched target edge;
- **rotate/reshape** an anisotropic splat along the edge tangent;
- **split** one broad splat into two lower-error children;
- **merge** redundant smooth-region splats and reuse the freed budget;
- **add** a typed residual splat;
- **prune** a low-contribution or harmful splat;
- **alpha rebalance** overlapping splats while preserving center opacity;
- **local reorder** splats in an overlap component;
- **recipe tune** bounded exporter constants;
- **compile primitive** into one or more exporter-compatible shapes;
- **residual tile** only in explicit hybrid mode.

Each operator is bounded. For example, split proposals preserve approximate
mass and do not immediately double the total budget:

```python
def split_splat(parent: "RawSplat", offset_fraction=0.45):
    tangent = np.array([np.cos(parent.theta), np.sin(parent.theta)])
    offset = tangent * parent.sx * offset_fraction
    child_sx = max(parent.sx * 0.65, 1e-4)
    child_alpha = 0.5 * parent.a

    return (
        replace(
            parent,
            x=parent.x - float(offset[0]),
            y=parent.y - float(offset[1]),
            sx=child_sx,
            a=float(child_alpha),
        ),
        replace(
            parent,
            x=parent.x + float(offset[0]),
            y=parent.y + float(offset[1]),
            sx=child_sx,
            a=float(child_alpha),
        ),
    )
```

The example preserves approximate optical density: if the children were
co-centered, their combined transmittance would match the parent because
rendered layer opacity is `1 - exp(-alpha * G)`. The children still require
post-fit after they are offset.

### 6. Local overlap-graph ordering

Do not attempt an all-pairs differentiable sort. Build an overlap graph and
optimize only connected components that intersect important residual regions.

```python
def local_order_candidates(splats, component, max_swaps=16):
    base_order = list(component)
    yield base_order
    ranked_pairs = rank_adjacent_swaps_by_occlusion_error(splats, component)
    for left, right in ranked_pairs[:max_swaps]:
        candidate = base_order.copy()
        i, j = candidate.index(left), candidate.index(right)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        yield candidate
```

The winning order must be encoded through existing stable importance ordering.
Tied importance values remain stable and deterministic.

### 7. Export-proxy post-fit

Continuous post-fit is performed against the matching deployment proxy:

- canvas: configured native renderer;
- SVG: sRGB/source-over renderer matching the selected recipe;
- PPTX: calibrated soft-edge or blur proxy.

Only parameters supported by the target artifact may move. For example, do not
optimize weighted compositing for an exporter that can only emit source-over.

Recommended late-stage learning-rate hierarchy:

```python
FIDELITY_LEARNING_RATES = {
    "position": 2e-4,
    "scale": 1e-4,
    "theta": 1e-4,
    "color": 8e-4,
    "alpha": 4e-4,
}
```

These are starting values, not accepted defaults. Each backend must demonstrate
matching direction and comparable gain on the benchmark corpus.

### 8. Actual-artifact-in-the-loop search

SVG is non-differentiably emitted and rasterized for final candidate selection.
Use proxy metrics to reject most candidates before paying that cost.

```python
def evaluate_candidate(candidate, context):
    proxy = evaluate_proxy(candidate, context)
    if violates_fast_gates(proxy):
        return proxy.with_status("proxy-rejected")

    artifact_path = emit_candidate(candidate, context.work_dir)
    deployed = rasterize_and_measure(
        artifact_path,
        target=context.target_linear_rgb,
        fixed_rois=context.fixed_rois,
    )
    return deployed
```

Recipe parameters may be searched with deterministic bounded coordinate search:

```python
def recipe_candidates(base):
    for alpha_scale in (base.alpha_scale * 0.95, base.alpha_scale, base.alpha_scale * 1.05):
        for sigma_scale in (base.sigma_scale * 0.95, base.sigma_scale, base.sigma_scale * 1.05):
            yield replace(
                base,
                alpha_scale=alpha_scale,
                sigma_scale=sigma_scale,
            )
```

Search per recipe and target. Do not learn one global constant from one image
and silently apply it to all exporters.

### 9. Cross-rasterizer robustness

The fidelity lab, outside the normal conversion path, renders SVG fixtures with:

- Chromium;
- Safari/WebKit where available;
- Firefox;
- librsvg;
- CairoSVG.

Optimize for robust quality, not a single engine:

```text
robust_score = median(engine_scores)
             + 0.5 * percentile_90(engine_scores)
```

Engine differences are recorded. A recipe may be target-specific when the user
explicitly selects a target, but the general browser recipe must not overfit one
rasterizer.

### 10. Color-management pass

Maximum fidelity requires one explicit color contract:

1. honor or deliberately normalize the source ICC profile;
2. decode input to linear-light working RGB;
3. optimize perceptual terms in OKLab;
4. composite in the deployment space;
5. encode output colors as sRGB unless another supported profile is explicit;
6. avoid accidental double transfer-function conversion;
7. measure the final raster back in the same normalized linear space.

Add small color patches to the calibration corpus: neutrals, skin-like tones,
saturated primaries, dark gradients, and translucent overlaps.

Color correction should first adjust contributing splats. A global 3×3 matrix
is allowed only as a diagnostic because it can improve averages while damaging
already-correct regions.

### 11. Mixed primitive compiler

Gaussians remain the canonical representation in the first slices. The
maximum-fidelity research path may introduce a stage-local primitive dictionary:

```python
PrimitiveKind = Literal[
    "gaussian",
    "ridge",
    "flat-ellipse",
    "short-stroke",
    "residual-tile",
]
```

Target-specific lowering:

| Primitive | SVG | DrawingML | Gaussian-only fallback |
| --- | --- | --- | --- |
| gaussian | gradient/blur recipe | soft edge/blur | native |
| ridge | path/stroke with soft edge | line/freeform or ellipse chain | Gaussian chain |
| flat ellipse | ellipse | ellipse | very broad Gaussian |
| short stroke | path | line/freeform | anisotropic Gaussian chain |
| residual tile | clipped image | picture fill | unsupported in pure-vector mode |

Mixed primitives are accepted only if the actual artifact improves. A primitive
that looks better in the internal preview but degrades PowerPoint is rejected.

### 12. Optional sparse residual atlas

Pure-vector output remains the default. Hybrid mode is explicit:

```text
--fidelity-stage max --allow-raster-residual
```

The residual is not one full-canvas image. Extract sparse, non-overlapping
patches only where splats have poor rate-distortion:

```python
@dataclass(frozen=True)
class ResidualPatch:
    x: int
    y: int
    rgba: np.ndarray
    encoded_bytes: int
    lpips_gain: float

    @property
    def gain_per_kib(self) -> float:
        return self.lpips_gain / max(self.encoded_bytes / 1024.0, 1e-6)
```

Choose patches by measured gain per encoded KiB, apply a soft or exact mask that
does not double-correct surrounding pixels, and stop at the configured byte and
area budgets.

Required safeguards:

- disabled for pure-vector mode;
- obvious manifest declaration;
- total covered area and encoded bytes reported;
- removable without corrupting the vector layer;
- no quality claim that hides whether hybrid mode was used.

### 13. Adaptive compute allocation

Spend remaining time where expected fidelity gain is highest:

```python
def operator_priority(history):
    return {
        name: stats.accepted_gain / max(stats.runtime_sec, 1e-6)
        for name, stats in history.items()
    }
```

After an initial exploration round, prioritize operators by accepted deployed
gain per second while retaining a small deterministic exploration quota.

Stop when:

- no candidate passes in a full round;
- the time budget is exhausted;
- quality gain falls below the noise threshold;
- all hard budgets are consumed.

## Configuration

Proposed profile shape:

```python
MAX_FIDELITY_STAGE = {
    "enabled": True,
    "mode": "max",
    "max_passes": 6,
    "max_candidates_per_pass": 16,
    "supersample": 2,
    "loss": {
        "oklab_charbonnier": 1.0,
        "local_ms_ssim": 0.25,
        "gradient_pyramid": 0.15,
        "laplacian_pyramid": 0.10,
        "salient_roi": 0.30,
    },
    "operators": {
        "recolor": True,
        "reshape": True,
        "split_merge": True,
        "typed_add": True,
        "prune": True,
        "local_reorder": True,
        "recipe_search": True,
    },
    "acceptance": {
        "min_lpips_gain": 0.001,
        "max_ssim_regression": 0.002,
        "max_edge_regression": 0.002,
        "max_worst_roi_regression_fraction": 0.01,
    },
    "allow_mixed_primitives": False,
    "allow_raster_residual": False,
}
```

This belongs in a dedicated typed configuration object before becoming another
large group of flat `refinement_config` keys.

## Observability and Reproducibility

Write fidelity artifacts under the run's selected artifact directory. For local
experiments, use paths such as `./tmp/fidelity/<run-id>/`.

Required artifacts:

```text
fidelity/
  baseline/
    artifact.svg|pptx
    raster.png
    metrics.json
  pass-00/
    analysis/
      residual-oklab.png
      edge-displacement.png
      priority.png
      rois.json
    candidates.jsonl
    accepted/
      artifact.svg|pptx
      raster.png
      metrics.json
  final/
    artifact.svg|pptx
    raster.png
    metrics.json
  decisions.jsonl
```

Manifest additions:

```json
{
  "fidelity_stage": {
    "enabled": true,
    "mode": "max",
    "seed": 42,
    "baseline_metrics": {},
    "final_metrics": {},
    "passes_run": 3,
    "candidates_evaluated": 27,
    "accepted_operations": ["reshape", "typed-add", "recipe-search"],
    "rejected_operations": 24,
    "actual_artifact_rasterizer": "cairosvg",
    "hybrid_residual": false,
    "stop_reason": "no-accepted-candidate"
  }
}
```

Given identical input, seed, configuration, dependencies, and rasterizer
version, candidate ordering and the winning result must be deterministic.

## Implementation Architecture

Add a focused module family:

```text
src/png2svg_gs/fidelity/
  __init__.py
  config.py          # typed config and profile resolution
  stage.py           # candidate loop and state isolation
  analysis.py        # residual topology and fixed ROIs
  metrics.py         # local/perceptual/deployed metric vector
  evaluator.py       # proxy and actual-artifact evaluation
  operators.py       # split/merge/move/recolor/reorder proposals
  primitives.py      # optional stage-local primitive dictionary
  report.py          # manifest and artifact trace
```

Integration points:

- `converter.py`: invoke the stage after current residual-detail passes;
- `renderer.py`: Torch loss terms and supersampled forward;
- `mlx_losses.py`: numerically matching MLX terms;
- `io.py`: candidate emission and actual SVG evaluation;
- `cli.py`: fidelity mode and explicit hybrid-output flags.

The main converter orchestrates; it must not absorb all fidelity logic.

## Delivery Plan

### Phase 0: Benchmark truth

- Freeze a small reference corpus spanning portrait, animal/fur, landscape,
  graphic art, transparency, smooth gradients, tiny hard edges, and text-like
  detail.
- Store source hashes, fixed crops, commands, rasterizer versions, and baseline
  artifacts.
- Add repeatability measurements to determine metric noise floors.
- Add actual SVG and real-PowerPoint calibration fixtures.

Exit gate: baseline results reproduce before any algorithmic change.

### Phase 1: Stage shell and monotonic evaluator

- Add typed config, candidate/result contracts, decision trace, and
  accept-or-revert gate.
- Wrap the current splat result as the baseline candidate.
- Emit/rasterize candidates without changing optimization behavior.
- Add tests proving rejected candidates leave output byte-equivalent to the
  baseline splat state.

Exit gate: enabling a no-op fidelity stage cannot change output.

### Phase 2: Better loss and supersampled polish

- Implement local/windowed MS-SSIM or an equivalent local structural term.
- Add Laplacian/gradient pyramids.
- Add fixed ROI weighting.
- Port exact semantics to MLX.
- Add optional 2× final polish.

Exit gate: actual SVG LPIPS improves on the corpus median and no hard gate fails.

### Phase 3: Residual topology and operators

- Add residual classification.
- Implement recolor, move, reshape, split, merge, typed-add, and prune.
- Add local overlap-graph reorder.
- Record gain/runtime per operator.

Exit gate: each enabled operator demonstrates at least one accepted gain and no
unreverted regression.

### Phase 4: Artifact-level recipe search

- Search bounded SVG recipe parameters after proxy fitting.
- Maintain cross-rasterizer fixture results.
- Expand calibrated PPTX proxy fixtures using real PowerPoint captures.

Exit gate: proxy direction correlates with deployed direction and actual SVG
quality improves.

### Phase 5: Mixed primitives

- Introduce ridge and short-stroke proposals behind a research flag.
- Lower unsupported primitives to Gaussian chains.
- Add exporter-specific validation and editability checks.

Exit gate: mixed primitives beat the Gaussian-only candidate under the same
artifact-size budget on the feature classes they target.

### Phase 6: Sparse residual atlas

- Implement only for explicit hybrid mode.
- Optimize patch selection by deployed gain per encoded KiB.
- Add area, byte, and disclosure gates.

Exit gate: hybrid results are clearly labeled and pure-vector output remains
unchanged by default.

### Phase 7: Learned proposal policy, only if justified

A lightweight model may rank operators or predict proposal parameters from
residual patches after enough accepted/rejected candidate data exists. It may
not become the renderer or the authority on acceptance.

Exit gate: it reduces search time without changing the deployed-artifact gate or
hurting determinism under a fixed model version.

## Validation Matrix

Run every accepted implementation slice across:

| Dimension | Values |
| --- | --- |
| backend | Torch, MLX |
| export target | canvas, standard SVG, browser SVG, scripted SVG, PPTX |
| rasterizer | CairoSVG, librsvg, browser lab; PowerPoint fixture for PPTX |
| content | portrait, fur, landscape, graphic, gradients, transparency, edges |
| budget | fixed splat count, fixed time, fixed artifact size |
| mode | fidelity off, balanced, max |

Required checks:

1. actual-artifact metrics;
2. fixed-ROI metrics;
3. visual side-by-side and residual maps;
4. splat count and file size;
5. runtime and peak memory;
6. determinism;
7. Torch/MLX parity;
8. native SVG/PPTX structure and editability;
9. no raster media in pure-vector PPTX;
10. no proxy-only claim presented as deployed fidelity.

## Acceptance Criteria

An implementation slice is accepted only when:

1. median deployed LPIPS improves on the reference corpus;
2. at least 60% of reference images improve by more than the measured noise
   floor;
3. no image violates worst-ROI, edge, SSIM, or color hard gates;
4. pure-vector output remains pure vector unless hybrid mode is explicit;
5. Torch and MLX remain semantically aligned;
6. deterministic reruns select the same winner;
7. added runtime and size are reported;
8. the accepted gain survives a side-by-side visual review.

Do not set a promised percentage or dB gain before Phase 0 establishes the
baseline and noise floor.

## Consequences

### Positive

- Optimizes the artifact users see rather than trusting only a proxy.
- Turns risky fidelity experiments into reversible candidate operations.
- Makes local defects visible instead of hiding them in global averages.
- Provides a path beyond the Gaussian-only ceiling without forcing an immediate
  schema rewrite.
- Separates maximum fidelity from normal fast conversion.
- Produces evidence for which operation, recipe, or primitive actually helped.

### Negative

- Maximum mode can be substantially slower.
- Actual-artifact evaluation adds dependencies and rasterizer variance.
- The metric and operator portfolio increases implementation and test surface.
- Mixed primitives complicate exporter parity and editability.
- Real PowerPoint remains an offline calibration workflow.
- A hybrid residual, even when bounded, weakens pure-vector semantics and must
  remain explicit.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| metric gaming | multi-metric hard gates plus visual review |
| overfitting the reference corpus | holdout images and content-class reporting |
| overfitting one SVG engine | cross-rasterizer robustness score |
| proxy improves, artifact worsens | actual-artifact acceptance for SVG |
| face/edge regression hidden by average | fixed ROI and worst-region gates |
| splat explosion | explicit count and file-size budgets |
| nondeterministic candidate search | stable ordering, fixed seed, recorded versions |
| converter complexity | dedicated `fidelity/` module family |
| MLX drifts from Torch | parity fixtures for every differentiable term |
| hybrid mode becomes silent default | explicit flag, manifest disclosure, structural tests |

## Alternatives Considered

### Only tune existing profile thresholds

Rejected as the complete answer. Existing profiles already contain extensive
content-specific thresholds. More tuning can help, but it does not add
artifact-level monotonic acceptance or new corrective operations.

### Replace the pipeline with a neural decoder

Rejected for the core product. It can raise raster fidelity but sacrifices the
editable, inspectable vector representation and complicates deterministic
export.

### Use LPIPS alone

Rejected. LPIPS can overlook edge displacement, color outliers, and a severe
small-region defect. It remains the primary perceptual metric inside a guarded
vector.

### Optimize only the internal renderer

Rejected for maximum fidelity. It cannot expose recipe- or application-specific
rendering differences.

### Always embed a raster residual

Rejected. It would inflate quality numbers while concealing the vector model's
actual fidelity. Sparse residuals are allowed only in explicit hybrid mode.

### Immediately replace Gaussians with arbitrary paths

Deferred. Typed Gaussian operators, local ordering, and artifact-level fitting
should be exhausted first. Mixed primitives enter through a compiler and must
earn their complexity under matched budgets.

## Non-Goals

- Guaranteeing identical output in every browser and PowerPoint version.
- Making maximum mode fast enough for interactive conversion.
- Hiding raster content inside a nominally pure-vector result.
- Treating a lower proxy loss as proof of deployed quality.
- Replacing the existing staged optimizer, region guidance, or residual-detail
  pass in one rewrite.

## Follow-up Actions

- [ ] Complete Phase 0 and record metric noise floors.
- [ ] Implement the no-op fidelity-stage shell and decision trace.
- [ ] Add fixed-ROI and local structural metrics.
- [ ] Add actual SVG artifact evaluation to candidate acceptance.
- [ ] Implement multi-scale Torch loss and matching MLX loss.
- [ ] Add residual topology fixtures.
- [ ] Implement operators one at a time with ablation results.
- [ ] Add bounded recipe search.
- [ ] Decide on mixed primitives only after Gaussian operator results.
- [ ] Decide on hybrid residuals only after pure-vector rate-distortion results.

## References

- `docs/adr-002-png2splat-python-pipeline.md`
- `docs/LAYERED_SALIENCY_PASSES_SPEC.md`
- `docs/MLX_RENDERER_OPTIMIZER_SPEC.md`
- `docs/SVG_PPTX_GAUSSIAN_TRICKS.md`
- `src/png2svg_gs/converter.py`
- `src/png2svg_gs/renderer.py`
- `src/png2svg_gs/mlx_losses.py`
- `src/png2svg_gs/io.py`
