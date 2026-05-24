# MLX Implementation Status

Date: May 24, 2026

This document captures the current state of the MLX renderer and optimizer work
so the next session can resume without reconstructing the thread.

## Current Verdict

MLX is the strongest Apple Silicon path tested so far.

The useful boundary is not "make NumPy faster". It is:

```text
renderer + loss + gradients + optimizer step inside MLX
```

Preprocessing, image loading, saliency maps, export, and final metrics still use
the existing Python/NumPy/scikit-image/PyTorch code where that is simpler and
not the main bottleneck.

## Implemented Modules

### `src/png2svg_gs/mlx_renderer.py`

Optional MLX renderer.

Implemented:

- `MlxBatchedGaussianRenderer`
- `MlxTilePlan`
- `splats_to_numpy_table()`
- alpha-over and weighted blend modes
- static CPU-built tile plan consumed by MLX render batches
- optional `max_active_splats_per_tile`

Measured parity:

- Small-fixture MLX render matched torch-batched reference with max absolute
  diff around `1.19e-7`.

### `src/png2svg_gs/mlx_losses.py`

Optional MLX loss helpers.

Implemented:

- `linear-l1`
- `oklab-l1`
- `weighted-oklab-l1`
- MLX port of the existing linear-RGB to OKLab transform

Weighted losses accept an `HxW` spatial weight map. The converter now passes the
region guidance weight map when the MLX loss profile starts with `weighted`.

### `src/png2svg_gs/mlx_optimizer.py`

Optional MLX optimizer primitives.

Implemented:

- `MlxSplatParams`
- functional parameter constraints matching the torch path
- `MlxAdam`
- global gradient clipping
- per-group learning rates for position, scale, theta, color, alpha
- canonical table-to-splats conversion preserving template layer metadata

### `src/png2svg_gs/mlx_stage.py`

Experimental MLX stage runner.

Implemented:

- `optimize_stage_mlx()`
- static tile-plan mode
- periodic tile-plan rebuild mode
- time-budget stop hook
- best-loss snapshot restore
- stage metrics for loss, iteration timing, tile-plan rebuilds, and tile-plan
  active density
- spatial-weighted loss plumbing

Guardrails:

- `static` tile plans reject moving geometry groups:
  `position`, `scale`, `theta`
- `periodic` tile plans allow geometry groups and rebuild tile membership every
  `N` iterations

## Converter And CLI Integration

### `src/png2svg_gs/cli.py`

Added:

```bash
--optimizer-backend torch|mlx
--mlx-loss linear-l1|oklab-l1|weighted-oklab-l1
--mlx-tile-plan static|periodic
--mlx-tile-plan-rebuild-interval N
--mlx-trainable-groups color,alpha
```

Example geometry-training MLX smoke:

```bash
./tmp/venv-mlx/bin/python -m png2svg_gs.cli input.png \
  -o ./tmp/chameleon_mlx_periodic_smoke.canvas.html \
  --format canvas \
  --max-edge 96 \
  --splats 32 \
  --stages 2 \
  --profile fast \
  --optimizer-backend mlx \
  --mlx-loss linear-l1 \
  --mlx-tile-plan periodic \
  --mlx-tile-plan-rebuild-interval 1 \
  --mlx-trainable-groups position,scale,theta,color,alpha \
  --artifacts-dir ./tmp/chameleon_mlx_periodic_smoke_artifacts \
  -v
```

Example weighted OKLab MLX smoke:

```bash
./tmp/venv-mlx/bin/python -m png2svg_gs.cli input.png \
  -o ./tmp/chameleon_mlx_weighted_oklab_smoke.canvas.html \
  --format canvas \
  --max-edge 96 \
  --splats 32 \
  --stages 2 \
  --profile fast \
  --optimizer-backend mlx \
  --mlx-loss weighted-oklab-l1 \
  --mlx-tile-plan periodic \
  --mlx-tile-plan-rebuild-interval 1 \
  --mlx-trainable-groups position,scale,theta,color,alpha \
  --artifacts-dir ./tmp/chameleon_mlx_weighted_oklab_smoke_artifacts \
  -v
```

### `src/png2svg_gs/converter.py`

Added:

- `optimizer_backend`
- MLX stage dispatch from `_optimize_stage()`
- manifest fields:
  - `optimizer_backend`
  - `mlx_loss`
  - `mlx_spatial_weighting_enabled`
  - `mlx_tile_plan`
  - `mlx_tile_plan_rebuild_interval`
  - `mlx_trainable_groups`
- region guidance trigger for weighted MLX losses
- spatial weight map passed into `optimize_stage_mlx()`

## Benchmark Scripts

### `scripts/benchmark_renderer_backend.py`

Renderer-only benchmark for fixed raw splat JSON files.

Useful command:

```bash
./tmp/venv-mlx/bin/python scripts/benchmark_renderer_backend.py \
  --input input.png \
  --max-edge 400 \
  --splats-json ./tmp/chameleon_400px_2000_mps_batched_artifacts/final.raw.json \
  --backends mlx-batched,torch-batched \
  --devices mps \
  --tile-sizes 16 \
  --batch-tile-counts 16,32 \
  --warmup 1 \
  --repeat 3 \
  --out ./tmp/renderer_backend_benchmark_mlx.json
```

### `scripts/benchmark_optimizer_backend.py`

Optimizer-stage benchmark for fixed raw splat JSON files.

Useful linear L1 geometry command:

```bash
./tmp/venv-mlx/bin/python scripts/benchmark_optimizer_backend.py \
  --input input.png \
  --max-edge 400 \
  --splats-json ./tmp/chameleon_400px_2000_mps_batched_artifacts/final.raw.json \
  --iters 2 \
  --loss linear-l1 \
  --tile-size 16 \
  --batch-tile-count 16 \
  --trainable-groups position,scale,theta,color,alpha \
  --tile-plan periodic \
  --tile-plan-rebuild-interval 1 \
  --out ./tmp/optimizer_backend_benchmark_mlx_periodic.json
```

Useful weighted OKLab command:

```bash
./tmp/venv-mlx/bin/python scripts/benchmark_optimizer_backend.py \
  --input input.png \
  --max-edge 400 \
  --splats-json ./tmp/chameleon_400px_2000_mps_batched_artifacts/final.raw.json \
  --iters 2 \
  --loss weighted-oklab-l1 \
  --weight-mode center \
  --tile-size 16 \
  --batch-tile-count 16 \
  --trainable-groups position,scale,theta,color,alpha \
  --tile-plan periodic \
  --tile-plan-rebuild-interval 1 \
  --out ./tmp/optimizer_backend_benchmark_mlx_weighted_oklab.json
```

## Measured Results

All numbers below are from local runs on May 24, 2026.

### Renderer-Only

Artifact: `./tmp/renderer_backend_benchmark_mlx.json`

| Backend | Device | Mode | Tile | Batch | Median |
| --- | --- | --- | ---: | ---: | ---: |
| `mlx-batched` | `mlx-default` | forward | 16 | 16 | `93.8 ms` |
| `mlx-batched` | `mlx-default` | forward | 16 | 32 | `94.6 ms` |
| `torch-batched` | `mps` | forward | 16 | 16 | `178.2 ms` |
| `torch-batched` | `mps` | forward | 16 | 32 | `198.7 ms` |

Artifact: `./tmp/renderer_backend_benchmark_mlx_backward.json`

| Backend | Device | Mode | Tile | Batch | Median |
| --- | --- | --- | ---: | ---: | ---: |
| `mlx-batched` | `mlx-default` | forward+backward | 16 | 16 | `300.8 ms` |

### Optimizer: Static Color/Alpha

Artifact: `./tmp/optimizer_backend_benchmark_mlx.json`

Configuration:

- 1923 splats
- 400px chameleon target
- `linear-l1`
- trainable groups: `color,alpha`
- static tile plan
- 5 iterations

Result:

```text
loss 0.092173 -> 0.066250
median_iter 398.9 ms
ssim_srgb 0.6411 -> 0.7034
```

### Optimizer: Periodic Geometry

Artifact: `./tmp/optimizer_backend_benchmark_mlx_periodic.json`

Configuration:

- 1923 splats
- 400px chameleon target
- `linear-l1`
- trainable groups: `position,scale,theta,color,alpha`
- periodic tile plan
- rebuild interval: `1`
- 2 iterations

Result:

```text
loss 0.092173 -> 0.083182
median_iter 398.4 ms
ssim_srgb 0.6411 -> 0.6932
tile_plan_rebuilds 4
tile_plan_rebuild_sec 0.0497 s
```

### Optimizer: Weighted OKLab

Artifact: `./tmp/optimizer_backend_benchmark_mlx_weighted_oklab.json`

Configuration:

- 1923 splats
- 400px chameleon target
- `weighted-oklab-l1`
- benchmark weight mode: `center`
- trainable groups: `position,scale,theta,color,alpha`
- periodic tile plan
- rebuild interval: `1`
- 2 iterations

Result:

```text
loss 0.043069 -> 0.037529
median_iter 446.8 ms
ssim_srgb 0.6411 -> 0.6827
tile_plan_rebuilds 4
tile_plan_rebuild_sec 0.0916 s
```

Notes:

- Weighted OKLab optimizes a different objective, so its loss values are not
  directly comparable to linear L1.
- In the short 2-iteration benchmark, linear L1 improved global `ssim_srgb`
  more than the center-weighted OKLab benchmark.
- The converter smoke using `weighted-oklab-l1` correctly triggered region
  guidance and recorded `spatial_weighted=true`.

## Smoke Artifacts

Static color/alpha:

- `./tmp/chameleon_mlx_cli_smoke.canvas.html`
- `./tmp/chameleon_mlx_cli_smoke_artifacts/run_manifest.json`

Periodic geometry:

- `./tmp/chameleon_mlx_periodic_smoke.canvas.html`
- `./tmp/chameleon_mlx_periodic_smoke_artifacts/run_manifest.json`

Weighted OKLab:

- `./tmp/chameleon_mlx_weighted_oklab_smoke.canvas.html`
- `./tmp/chameleon_mlx_weighted_oklab_smoke_artifacts/run_manifest.json`

Manifest checks observed:

```text
optimizer_backend=mlx
mlx_loss=weighted-oklab-l1
mlx_spatial_weighting_enabled=True
tile_plan_mode=periodic
spatial_weighted=True
```

## Tests Run

Latest full unit run before stopping:

```text
venv/bin/python -m pytest tests/unit
63 passed, 3 skipped
```

The skipped tests are MLX runtime tests skipped in the normal project venv
because that venv does not include MLX.

MLX runtime checks were run through `./tmp/venv-mlx`.

## Current Guardrails

Keep these in place until there is more evidence:

- Static MLX tile plans may only optimize `color,alpha`.
- Moving geometry requires `--mlx-tile-plan periodic`.
- Periodic rebuild interval should start at `1` for correctness experiments,
  then be tuned upward after quality is stable.
- Use `weighted-oklab-l1` only when region guidance is acceptable, because it
  adds preprocessing time.
- Treat short global SSIM changes as diagnostics, not final quality verdicts.

## Open Issues

1. The benchmark scripts are experimental and not part of CI.
2. `./tmp/venv-mlx` does not have pytest installed, so MLX runtime tests are
   checked with direct CLI/script smoke runs.
3. Region guidance emits scikit-image deprecation warnings under the newest
   scikit-image.
4. Some internal logger messages still print bare `tmp/...`; user-facing
   documentation should use `./tmp/...`.
5. Full 400px/2000-splat MLX training has not been run for the full staged
   schedule after periodic geometry and weighted OKLab were added.
6. No 10k MLX run has been attempted yet.
7. Weighted OKLab needs tuning against actual visual output, not just
   2-iteration smoke metrics.

## Suggested Next Work

Do not start with a long run. The next useful sequence is:

1. Run a 400px/2000-splat short schedule with periodic geometry and linear L1:

   ```bash
   ./tmp/venv-mlx/bin/python -m png2svg_gs.cli input.png \
     -o ./tmp/chameleon_400px_mlx_linear_short.canvas.html \
     --format canvas \
     --max-edge 400 \
     --splats 2000 \
     --stages 5 \
     --profile fast \
     --optimizer-backend mlx \
     --mlx-loss linear-l1 \
     --mlx-tile-plan periodic \
     --mlx-tile-plan-rebuild-interval 1 \
     --mlx-trainable-groups position,scale,theta,color,alpha \
     --artifacts-dir ./tmp/chameleon_400px_mlx_linear_short_artifacts \
     -v
   ```

2. Repeat with `weighted-oklab-l1` and compare visual output:

   ```bash
   ./tmp/venv-mlx/bin/python -m png2svg_gs.cli input.png \
     -o ./tmp/chameleon_400px_mlx_weighted_oklab_short.canvas.html \
     --format canvas \
     --max-edge 400 \
     --splats 2000 \
     --stages 5 \
     --profile fast \
     --optimizer-backend mlx \
     --mlx-loss weighted-oklab-l1 \
     --mlx-tile-plan periodic \
     --mlx-tile-plan-rebuild-interval 1 \
     --mlx-trainable-groups position,scale,theta,color,alpha \
     --artifacts-dir ./tmp/chameleon_400px_mlx_weighted_oklab_short_artifacts \
     -v
   ```

3. If both are stable, tune rebuild interval:

   ```text
   1, 2, 5, 10
   ```

4. Only after that, attempt a longer 400px/2000 staged run.

## Files Touched In This MLX Line

Primary MLX files:

- `src/png2svg_gs/mlx_renderer.py`
- `src/png2svg_gs/mlx_losses.py`
- `src/png2svg_gs/mlx_optimizer.py`
- `src/png2svg_gs/mlx_stage.py`

CLI/converter:

- `src/png2svg_gs/cli.py`
- `src/png2svg_gs/converter.py`

Benchmarks:

- `scripts/benchmark_renderer_backend.py`
- `scripts/benchmark_optimizer_backend.py`

Tests:

- `tests/unit/test_mlx_renderer.py`
- `tests/unit/test_mlx_optimizer.py`
- `tests/unit/test_torch_batched_renderer.py`

Specs/status:

- `docs/APPLE_GPU_RENDERER_SPEC.md`
- `docs/MLX_RENDERER_OPTIMIZER_SPEC.md`
- `docs/MLX_IMPLEMENTATION_STATUS.md`
