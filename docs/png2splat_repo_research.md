# PNG2Splat Repo Research (2026-02-15)

## Objective
Build a high-fidelity Python `png2splat` pipeline that can export both `SVG` and `DrawingML`, with quality as the primary target.

## Repos Pulled

| Repo | Local Path | Commit | Focus |
|---|---|---:|---|
| GaussianImage | `external/GaussianImage` | `d53393b` | 2D Gaussian fitting + quantization |
| Instant-GI | `external/Instant-GI` | `91bba42` | Learned gaussian initialization + short fine-tuning |
| image-gs | `external/image-gs` | `0308836` | Error-guided progressive optimization for 2D Gaussians |
| gsplat | `external/gsplat` | `6f37836` | CUDA rasterization + split/prune strategies |
| 2d-gaussian-splatting | `external/2d-gaussian-splatting` | `335ad61` | Densify/clone/split/prune training loop |
| diffvg | `external/diffvg` | `85802a7` | Differentiable vector graphics baseline |
| ml-sharp | `ml-sharp` | `1eaa046` | Feed-forward gaussian prediction architecture patterns |
| svg2ooxml | `../svg2ooxml` | `4a16e34` | DrawingML/PPTX builder architecture reference |

## What Each Repo Actually Does

### 1. GaussianImage
- Core loop: initialize splats, optimize with differentiable rasterization, optionally quantize.
- Gaussian construction: random trainable `xy/scale/rot/color` params (`_xyz`, `_scaling`, `_rotation`, `_features_dc`) in `external/GaussianImage/gaussianimage_rs.py:26`.
- Renderer: `project_gaussians_2d_scale_rot` + `rasterize_gaussians_sum` in `external/GaussianImage/gaussianimage_rs.py:75`.
- Losses: `L1/L2/SSIM/fusion` in `external/GaussianImage/utils.py:20`.
- Useful for us: clean 2D gaussian parameterization and straightforward optimization baseline.

### 2. image-gs
- Strongest direct reference for image-fitting quality pipeline.
- Initialization: gradient/saliency/random sampling in `external/image-gs/model.py:220`.
- Progressive schedule: start from `initial_ratio`, then add gaussians over time in `external/image-gs/model.py:145` and `external/image-gs/model.py:547`.
- Loss: weighted `L1 + L2 + SSIM` in `external/image-gs/model.py:444`.
- Useful for us: error-guided progressive addition logic and staged optimization profile.

### 3. Instant-GI
- Adds a learned initializer network for much faster convergence.
- InitNet predicts position probability + gaussian fields from image features in `external/Instant-GI/generalizable_model/init_net.py:89`.
- Uses multiple init modes (`net`, `random`, `quard`) in `external/Instant-GI/train.py:65`.
- Fine-tunes with same gsplat render core in `external/Instant-GI/gaussianimage_rs.py:100`.
- Useful for us: optional learned init stage (not required for first correct pipeline).

### 4. 2d-gaussian-splatting
- Densification-heavy training with prune/clone/split cycle.
- Parameter-group optimizer setup in `external/2d-gaussian-splatting/scene/gaussian_model.py:148`.
- Densify and prune implementation in `external/2d-gaussian-splatting/scene/gaussian_model.py:389`.
- Training loop triggers refinement periodically in `external/2d-gaussian-splatting/train.py:125`.
- Useful for us: battle-tested split/clone/prune mechanics and scheduling pattern.

### 5. gsplat
- Not a pipeline itself; it is the rasterization backend.
- Provides efficient rasterization APIs and strategies in `external/gsplat/gsplat/rendering.py:108` and `external/gsplat/examples/simple_trainer_2dgs.py:317`.
- Includes concrete refine thresholds config (`prune_opa`, `grow_grad2d`, etc.) in `external/gsplat/examples/simple_trainer_2dgs.py:97`.
- Useful for us: this should be the production renderer backend for quality and speed.

### 6. diffvg
- Differentiable vector graphics renderer (paths/shapes/gradients), not gaussian-native.
- Strong for path-level SVG optimization, not ideal as core gaussian splat engine.
- Useful for us: optional post-pass for vector cleanup, not core `png2splat`.

### 7. ml-sharp
- Predicts 3D gaussians from one image in a feed-forward pass.
- Pipeline structure (init model + delta head + composer) is clear in `ml-sharp/src/sharp/models/predictor.py:136`.
- Base gaussian creation is explicit in `ml-sharp/src/sharp/models/initializer.py:127`.
- Useful for us: architectural inspiration, not a direct 2D png-fitting drop-in.

### 8. svg2ooxml (local builder reference)
- Not a splatting repo; this is a mature OOXML/DrawingML conversion toolkit.
- Confirms robust DrawingML concerns we should preserve: EMU conversion and typed XML builders (see `../svg2ooxml/src/svg2ooxml/services/clip_service.py:29` and `../svg2ooxml/src/svg2ooxml/drawingml/xml_builder.py`).
- Useful for us: keep raw splat core independent, but align DrawingML writer patterns with `svg2ooxml` style safety and schema discipline.

## What We Learn About “How It Builds Gaussians”

Across successful repos, gaussian building has 4 explicit stages:
1. **Seed positions**: gradient/saliency/random or learned probability field.
2. **Initialize attributes**: `(x, y), scale(s), rotation, color, alpha`.
3. **Optimize by rendering loss**: render -> compare to target -> backprop.
4. **Refine topology**: add/split/clone/prune based on residual and contribution.

The key gap in low-quality outputs is usually stage 4 and backend quality, not only stage 2.

## Mapping to Current SplatThis

Current `png2svg_gs` already has parts of this:
- Content-adaptive init in `src/png2svg_gs/converter.py:264`.
- Stage optimization loop in `src/png2svg_gs/converter.py:397`.
- Error-driven densify + residual-aware prune in `src/png2svg_gs/converter.py:489` and `src/png2svg_gs/converter.py:575`.
- Parameter-group LR helper in `src/png2svg_gs/optimizer.py:74`.
- DrawingML export path in `src/png2svg_gs/io.py:142`.

Main weakness today:
- Renderer is a simple CPU PyTorch weighted-average compositor (`src/png2svg_gs/renderer.py:4`), not gsplat-class rasterization.

## Recommended Technical Direction

1. **Adopt `gsplat` as the primary renderer backend** for quality mode.
2. Keep current torch renderer as CPU fallback for tests/dev.
3. Preserve current staged init/optimize/densify/prune logic, but run it on gsplat projections/rasterization.
4. Keep `raw splat` JSON as canonical intermediate and continue dual exporters (`SVG`, `DrawingML`).

## Dependency Decision

Decision: requiring `gsplat` for the high-quality `png2splat` path is justified.

Suggested packaging shape:
- default install: current lightweight path
- quality install extra: `pip install .[gsplat]`
- runtime option: `--backend torch|gsplat` (default `gsplat` when available)

## Immediate Next Implementation Steps

1. Add renderer backend abstraction (`torch` vs `gsplat`).
2. Implement `GsplatRenderer2D` adapter (position/scale/rotation/color/alpha tensor contract).
3. Route existing optimization loop through backend interface.
4. Re-tune densify/prune thresholds using image-gs + gsplat strategy defaults.
5. Run regression set with PSNR/SSIM gates and side-by-side outputs.
