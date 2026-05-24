# Apple GPU Renderer Spec

## Status

Draft, based on local MacBook Air M4 measurements from May 24, 2026.

## Problem

SplatThis can now run on Apple GPU through PyTorch MPS, but the current renderer
does not use the GPU efficiently.

The current `GaussianRenderer` is tile based for memory control:

- `_render_tiled()` loops over image tiles in Python.
- `_get_tile_splat_bins()` builds per-tile active splat lists and currently
  detaches tile bounds to CPU/NumPy while rebuilding bins.
- `_render_tile()` launches tensor work for one tile at a time.
- Alpha-over compositing uses per-tile `cumprod` over active splats.

This works well on CPU because each tile is a small, local chunk of work. It is a
poor fit for MPS because each tile becomes a small GPU dispatch and Python stays
in the scheduling loop.

## Baseline Evidence

Reference run:

```bash
./tmp/venv-pytorch-mps/bin/python -m png2svg_gs.cli input.png \
  -o ./tmp/chameleon_400px_2000_mps.canvas.html \
  --format canvas \
  --max-edge 400 \
  --splats 2000 \
  --stages 30,20,10 \
  --device mps \
  --no-apple-silicon-splat-cap \
  --artifacts-dir ./tmp/chameleon_400px_2000_mps_artifacts \
  -v
```

Same command with `--device cpu`:

| Device | Total | Optimize | Final Splats | Internal SSIM | Proxy SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 59.47s | 57.56s | 1928 | 0.7540 | 0.7534 |
| MPS | 269.20s | 266.40s | 1924 | 0.7597 | 0.7585 |

Conclusion: MPS is available and correct, but the current renderer is around
4.5x slower than CPU for a 400px, 2000-splat chameleon run.

MLX also has working GPU access on the same machine:

| Backend | Smoke Result |
| --- | --- |
| MLX GPU | 5 matmuls in 0.0376s |
| MLX CPU | 5 matmuls in 0.3279s |

## Goal

Add an Apple-GPU-suitable renderer path that keeps the optimization interface
unchanged but feeds MPS/MLX with fewer, larger tensor operations.

Primary target:

- Make `--device mps` faster than CPU for 400px/2000-splat training.

Secondary target:

- Establish a backend shape that can be ported to MLX without rewriting the
  optimizer or export pipeline.

## Non-Goals

- Do not replace the CPU renderer. CPU remains the reference backend.
- Do not start with custom Metal kernels.
- Do not change splat semantics, ordering, loss functions, or export formats.
- Do not optimize SVG/PPTX output in this spec.

## Backend Strategy

Introduce a new renderer backend family:

- `torch`: existing CPU-friendly tiled renderer.
- `torch-batched`: PyTorch tensor backend designed for MPS and CUDA-style
  dispatch. This is the first implementation target.
- `mlx`: native Apple Silicon backend after `torch-batched` proves the data
  layout and batching strategy.

CLI/API direction:

```bash
--backend torch --device cpu
--backend torch-batched --device mps
--backend mlx --device gpu
```

`auto` should select:

1. `gsplat` only on CUDA where supported.
2. `torch-batched` on MPS once acceptance targets pass.
3. `torch` otherwise.

## Core Design

### 1. Tensor Table Stays Canonical

Keep the current `[N, 11]` splat tensor contract:

```text
x, y, sx, sy, theta, reserved, r, g, b, alpha, importance
```

The batched renderer may internally unpack to structure-of-arrays views, but no
upstream optimizer API should change.

### 2. Replace Per-Tile Python Rendering With Tile Batches

Current shape per tile:

```text
[tile_h, tile_w, active_splats]
```

Target shape per batch:

```text
[batch_tiles, tile_h, tile_w, max_active_splats]
```

Each batch renders many tiles in one call. For a 400px image and 32px tiles,
that is roughly 13 x 13 = 169 tiles. A batch size of 16 to 64 tiles should
reduce dispatch count by that same factor.

### 3. Fixed-Width Active Splat Table

Build a padded active-splat index tensor:

```text
tile_splat_indices: [num_tiles, max_active]
tile_splat_mask:    [num_tiles, max_active]
```

For each tile batch:

```python
indices = tile_splat_indices[batch_tile_ids]
mask = tile_splat_mask[batch_tile_ids]
active_mu = mu[indices]
```

The mask zeroes out padded entries before compositing.

This gives the renderer dense tensor shapes that MPS can compile and reuse.

### 4. Keep Tile Binning Mostly on Device

Phase 1 can still build tile bins on CPU once per forward pass, because the main
win is fewer render dispatches. Phase 2 should move bin construction to tensor
operations:

```text
splats -> x_min/x_max/y_min/y_max -> tile coverage matrix
```

Avoid `.cpu().numpy()` in the MPS path except when writing artifacts or final
metrics.

### 5. Use Larger GPU Tiles

CPU found small tiles useful. MPS needs larger work chunks.

Benchmark tile sizes:

```text
16, 24, 32, 48, 64, 96
```

Expected default candidates:

- CPU: keep current tuned default.
- MPS: start at 32 or 64.
- MLX: start at 32 or 64.

### 6. Preserve Alpha-Over Semantics

The output must match the existing alpha-over renderer within tolerance.

Per batch:

```text
density = weights * alpha
alpha_layers = 1 - exp(-density)
transmittance_prefix = cumprod(1 - alpha_layers)
contributions = transmittance_prefix * alpha_layers
color = sum(contributions * splat_color) + remaining * background
```

The ordering remains `importance` ascending before compositing.

## Implementation Plan

### Phase A: Benchmark Harness

Add a repeatable benchmark command/script for renderer-only timing.

Inputs:

- `input.png`
- `--max-edge 400`
- fixed `2000` splat JSON from an existing run
- CPU, MPS, and later MLX devices

Outputs:

- render forward time
- forward+backward time
- peak memory if available
- SSIM/PSNR parity against CPU reference
- tile/bin statistics

Acceptance:

- One command emits JSON under `./tmp/...`.
- Benchmark does not include SVG/PPTX generation time.

### Phase B: `TorchBatchedGaussianRenderer`

Add `TorchBatchedGaussianRenderer` beside `GaussianRenderer`.

Constructor options:

```python
batch_tile_count: int = 32
gpu_tile_size: int = 32
max_active_splats_per_tile: Optional[int] = None
```

Implementation steps:

1. Unpack and importance-sort splats once.
2. Build padded tile-splat index tensors.
3. Precompute per-tile pixel coordinate blocks.
4. Render `batch_tile_count` tiles per tensor call.
5. Scatter batch outputs back into `[H, W, 3]`.
6. Support both `weighted` and `alpha-over`; optimize `alpha-over` first.

Acceptance:

- CPU parity against `GaussianRenderer` on small deterministic fixtures.
- MPS forward+backward works on the chameleon run.
- No forced CPU sync inside the training iteration except logging/metrics.

### Phase C: MPS Tuning

Benchmark these knobs:

- `gpu_tile_size`: 16, 24, 32, 48, 64, 96
- `batch_tile_count`: 8, 16, 32, 64, all tiles
- `max_active_splats_per_tile`: uncapped, p95, p99, 256, 512, 1024
- dtype: float32 first, then optional mixed precision experiments

Acceptance target:

- 400px, 2000-splat chameleon training is at least 1.5x faster on MPS than CPU.
- Quality delta versus CPU is within `0.002` SSIM for same seed/config.

Stretch target:

- MPS is at least 2x faster than CPU on 10k+ native-photo runs.

### Phase D: MLX Prototype

Port only the batched forward path first.

Requirements:

- Same input tensor table semantics.
- Same output shape and alpha-over semantics.
- No optimizer integration until forward parity is proven.

Then decide between:

- MLX only for rendering, with gradients still in PyTorch not viable.
- Full MLX training loop for renderer+loss+optimizer.
- Keep MLX as an inference/export renderer only.

Acceptance:

- MLX forward image matches CPU reference within a documented tolerance.
- MLX forward is faster than PyTorch MPS for the same fixed splat table.

## Test Plan

### Unit Tests

- Empty splat table renders background.
- One centered circular splat matches current renderer.
- One rotated anisotropic splat matches current renderer.
- Weighted blend matches current renderer.
- Alpha-over blend matches current renderer.
- Partial edge tiles match full interior tiles.
- Padded inactive splat slots have no visual effect.

### Integration Tests

- Chameleon 400px, 2000 splats:
  - CPU reference output saved.
  - `torch-batched` CPU output parity.
  - `torch-batched` MPS smoke if MPS is available.

### Performance Tests

Do not put long performance tests in default CI.

Add an explicit command:

```bash
./tmp/venv-pytorch-mps/bin/python scripts/benchmark_renderer_backend.py \
  --input input.png \
  --max-edge 400 \
  --splats-json ./tmp/chameleon_400px_2000_cpu_artifacts/final.raw.json \
  --devices cpu,mps \
  --backends torch,torch-batched \
  --out ./tmp/renderer_backend_benchmark.json
```

## Risks

### Memory Blow-Up

Dense shape can get large:

```text
batch_tiles * tile_h * tile_w * max_active_splats
```

Mitigation:

- Start with capped `batch_tile_count`.
- Cap or bucket `max_active_splats_per_tile`.
- Fall back to existing renderer if active counts are too high.

### Dynamic Shapes Recompile on MPS

MPS may recompile when tile batches have different active widths.

Mitigation:

- Use fixed padded widths per render.
- Optionally bucket tiles by active-count range.

### Scatter Writes

Writing batched tiles back to the full image can itself become expensive.

Mitigation:

- First use simple Python batch scatter; measure.
- Then switch to reshaped tiled image layout if scatter dominates.

### Autograd Memory

Forward can be fast but backward may keep large intermediates.

Mitigation:

- Measure forward and backward separately.
- Keep existing CPU renderer for memory-constrained runs.
- Consider checkpointing only after baseline speed is proven.

## Acceptance Summary

The spec is satisfied when:

- `torch-batched` exists as a selectable renderer backend.
- It matches the current renderer visually and numerically within tolerance.
- MPS can complete the 400px/2000-splat chameleon benchmark.
- MPS is faster than CPU for that benchmark.
- The benchmark JSON and generated artifacts live under `./tmp/...`.

