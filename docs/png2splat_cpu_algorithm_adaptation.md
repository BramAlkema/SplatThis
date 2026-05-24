# PNG2Splat CPU Algorithm Adaptation

## Scope

This note documents which high-value algorithms from `image-gs`, `GaussianImage`, and `Instant-GI` are adapted for a CPU-only `png2svg_gs` pipeline (no CUDA/gsplat runtime requirement).

## Source Algorithms We Borrowed

1. Proper alpha compositing instead of weighted-average blending.
- Source pattern: `image-gs` forward path uses rasterized compositing output and transmittance-aware blending (`external/image-gs/model.py:356` to `external/image-gs/model.py:364`).

2. Mixed initialization strategy (content-adaptive + random coverage).
- Source pattern: gradient/saliency init with explicit random ratio (`external/image-gs/model.py:220` to `external/image-gs/model.py:246`).

3. Error-guided progressive densification with residual-informed feature/color initialization.
- Source pattern: sample by error map and initialize added Gaussian features from diff map (`external/image-gs/model.py:547` to `external/image-gs/model.py:570`).

4. Plateau-based learning-rate decay and early stopping.
- Source pattern: check PSNR/SSIM improvement and decay LR when stalled (`external/image-gs/model.py:528` to `external/image-gs/model.py:545`).

5. Rotation+scale covariance parameterization.
- Source pattern: build covariance from scale and rotation (`external/GaussianImage/utils.py:96` to `external/GaussianImage/utils.py:121`).

6. Scale regularization idea during optimization.
- Source pattern: explicit scaling loss term (`external/Instant-GI/train_init_net.py:161`).

## Implemented In This Repo (CPU-safe)

1. Alpha compositing renderer.
- Replaced normalized weighted-average blending with front-to-back alpha compositing in tile renderer.
- File: `src/png2svg_gs/renderer.py`.
- Key change region: `GaussianRenderer._render_tile(...)`.
- Implementation note: both modes are available. Default remains `weighted` for backward-compatible regression behavior; the new algorithmic mode is `alpha-over`.

2. Mixed initialization controls.
- Added `init_random_ratio` and `init_gradient_weight` to converter configuration and manifest.
- File: `src/png2svg_gs/converter.py`.
- CLI flags added:
  - `--init-random-ratio`
  - `--init-gradient-weight`
- File: `png2svg`.

3. Residual-guided densification color initialization.
- During densification, color is now initialized from local color plus residual correction gain.
- File: `src/png2svg_gs/converter.py` in `_add_error_driven_splats(...)`.
- Added profile key: `refinement.residual_color_gain`.

4. Plateau LR decay + early-stop in stage optimization.
- Added per-profile `schedule` defaults and runtime schedule handling in `_optimize_stage(...)`.
- File: `src/png2svg_gs/converter.py`.
- Manifest now captures `schedule_config`.

5. Direct `(sx, sy, theta)` optimization parameterization.
- Tensor layout now stores shape as scales+rotation instead of raw covariance entries:
  - `[x, y, sx, sy, theta, reserved, r, g, b, alpha, importance]`
- Renderer computes Gaussian weights in the splat's local rotated frame.
- Conversion to/from `GaussianSplat.sigma` is handled at tensor boundaries.
- Files:
  - `src/png2svg_gs/renderer.py`
  - `src/png2svg_gs/optimizer.py`
  - `src/png2svg_gs/converter.py`

## Not Yet Implemented (Next Phase)

1. Explicit scale-regularization loss term.
- Not yet added to total loss composition.

## Why This Is CPU-Compatible

All implemented changes are pure NumPy/PyTorch logic in existing CPU execution paths. No CUDA kernels or gsplat-only operators are required.
