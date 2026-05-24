# ADR-002: PNG to Gaussian Splat Pipeline in Python

**Status**: Accepted  
**Date**: 2026-02-15  
**Accepted On**: 2026-02-15  
**Authors**: SplatThis Development Team

## Context

We need a reliable `png2splat` pipeline in Python that:

1. Produces high-fidelity Gaussian reconstructions from PNG input
2. Supports deterministic and repeatable output
3. Exports to practical targets (`SVG`, `DrawingML`, and raw splat data)
4. Improves on the current pipeline, which often runs but does not meet output quality expectations

Recent repo review and research suggest that quality comes from a hybrid approach: strong initialization plus differentiable optimization and adaptive refinement.

## Problem Statement

The current pipeline is not consistently delivering the intended result ("correct pipeline"), especially for detail retention and stable splat allocation. A "perfect png2splat" requires explicit architectural decisions about:

1. Gaussian parameterization
2. Initialization strategy
3. Optimization objectives
4. Densification/pruning policy
5. Export contract

## Decision Drivers

1. Output fidelity is the primary goal
2. Python-first implementation is required
3. Pipeline must remain explainable and debuggable
4. Must support downstream vector/document workflows (including DrawingML)
5. Should leverage proven ideas from `apple/ml-sharp` and image-GS style methods without hard-coupling to one repo

## Options Considered

### Option 1: Heuristic-only conversion (no optimization)
Pros:
- Fast
- Simple implementation

Cons:
- Quality ceiling is low
- Fails on hard images (fine texture, sharp edges, smooth gradients together)

### Option 2: Full end-to-end optimization from random init
Pros:
- Theoretically highest flexibility

Cons:
- Slow convergence
- Fragile and hard to debug
- Poor reproducibility without careful controls

### Option 3: Hybrid deterministic init + optimization + adaptive refinement (Chosen)
Pros:
- Strong starting point and stable convergence
- Better quality/compute tradeoff
- Easier to profile and improve stage-by-stage

Cons:
- More moving parts than heuristic-only
- Requires disciplined interfaces between stages

## Decision

Adopt **Option 3** and define the canonical `png2splat` pipeline as:

1. **Preprocess**
   - Linearize RGB and normalize coordinates to `[0, 1]`
   - Build gradient/error priors for initialization

2. **Initialize Gaussian Set**
   - Content-adaptive placement (mix of gradient/saliency + coverage prior)
   - Anisotropic 2D Gaussian parameterization:
     - center `(x, y)`
     - scale `(sx, sy)` with positivity constraints
     - rotation `theta`
     - color `(r, g, b)`
     - alpha `a`

3. **Differentiable Optimization**
   - Optimize reconstruction with combined losses:
     - pixel loss (`L1`/`L2`)
     - structural/perceptual term (`SSIM` or equivalent)
   - Use parameter-group learning rates for stable training

4. **Adaptive Refinement**
   - Densify where residual error remains high
   - Prune low-contribution splats
   - Repeat optimize/refine cycles until budget or quality target

5. **Export**
   - `raw splat` (authoritative interchange format)
   - `SVG` renderer export
   - `DrawingML` export for document workflows

## What We Learn from `apple/ml-sharp`

`ml-sharp` is not a direct drop-in PNG-to-splat converter for this project, but it provides useful architectural guidance:

1. Separate representation, renderer, and optimization loop cleanly
2. Use stable Gaussian parameterization and constraints
3. Treat refinement (split/prune/reweight) as a first-class stage, not a patch
4. Measure quality and convergence continuously, not only at the end

## Consequences

Positive:
1. Clear, testable pipeline contract for `png2splat`
2. Better path to high fidelity than heuristic-only conversion
3. Direct support for `DrawingML` alongside SVG

Negative:
1. Higher implementation complexity
2. Longer runtime than simple conversion
3. Requires robust test images and quality benchmarks

## Execution Checklist

Phase 1 (Execution Now):
- [ ] Define canonical raw splat schema (`center`, `scale`, `theta`, `rgb`, `alpha`, optional metadata)
- [ ] Enforce parameter constraints and validation rules (positive scale, bounded alpha/color)
- [ ] Implement deterministic initializer with explicit seed threading through API and CLI
- [ ] Add baseline differentiable optimizer with `L1 + SSIM`
- [ ] Add stage-level artifacts for debugging (`init`, `iter`, `final`) and structured logs
- [ ] Add tests for determinism, schema validation, and optimizer regression guardrails

Phase 2:
- [ ] Implement densify/prune cycle driven by residual error and contribution score
- [ ] Add acceptance metrics (PSNR, SSIM, runtime, splat count)
- [ ] Build fixed-image regression suite with thresholded pass criteria

Phase 3:
- [ ] Tune per-parameter learning-rate groups and refinement thresholds
- [ ] Add round-trip validation across exporters (`raw -> SVG`, `raw -> DrawingML`)
- [ ] Add profile presets: `fast`, `balanced`, `max-fidelity`

Definition of Done for this ADR:
- [ ] `png2splat` runs end-to-end with reproducible output under fixed seed
- [ ] Quality metrics beat current baseline on reference set
- [ ] Raw splat output is the authoritative intermediate for all exporters
- [ ] Failures are diagnosable from stage logs/artifacts without stepping through debugger

## Acceptance Criteria

1. Reproducible output given fixed seed and config
2. Quality metrics improve over current baseline on reference images
3. End-to-end pipeline supports `PNG -> raw splat -> SVG` and `PNG -> raw splat -> DrawingML`
4. Failures are diagnosable by stage-level logs and artifacts

## Non-Goals

1. One-click universal perfection for all images in a single pass
2. Replacing every existing pipeline with a CUDA-only architecture
3. Locking the project to one external repository implementation

## References

1. Apple `ml-sharp`: https://github.com/apple/ml-sharp
2. Prior internal decision: `docs/adr-001-image-gs-insights.md`
3. Related method families reviewed: GaussianImage, Instant GaussianImage, DiffVG, LIVE, Bézier-style splat/vector hybrids
