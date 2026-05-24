# MLX Renderer + Optimizer Spec

## Status

Draft, based on local MacBook Air M4 measurements from May 24, 2026.

This spec starts after the initial `mlx-batched` renderer prototype:

- `MlxBatchedGaussianRenderer` matches the torch-batched renderer numerically
  on small fixtures.
- A 1923-splat 400px chameleon forward render benchmarks at roughly `94 ms`
  with MLX versus `178-199 ms` with PyTorch MPS.
- MLX forward+backward on the same fixed table runs at roughly `301 ms`.

The evidence says MLX is worth moving into the optimization loop, but only if
we keep the whole hot loop in MLX and avoid per-iteration Python/NumPy bounces.

## Problem

The current training stage is PyTorch-owned:

- `SplatParams` stores trainable tensors as separate `nn.Parameter` groups.
- `torch.optim.Adam` applies group-specific learning rates.
- `renderer(params.as_tensor())` renders the stage image.
- `loss_fn(rendered, target)` computes the training loss.
- `loss.backward()`, gradient clipping, Adam, and clamps happen every iteration.

That structure is correct, but on Apple Silicon the PyTorch MPS path still
spends too much time in small dispatches and framework overhead. The renderer
has now been batched, but the next large speedup requires the renderer, loss,
gradients, optimizer, constraints, and best-snapshot tracking to live in MLX
for the whole stage.

MLX will not speed up existing NumPy, PIL, or scikit-image code automatically.
This spec targets only the hot optimization loop.

## Goal

Add an experimental MLX training path that runs one optimization stage fully in
MLX:

```text
canonical splat table -> MLX params -> MLX render -> MLX loss
-> MLX gradients -> MLX Adam -> MLX constraints -> final table
```

The first target is fixed-splat-count stage optimization. Existing Python,
NumPy, and PyTorch paths remain the reference implementation.

## Non-Goals

- Do not rewrite preprocessing, saliency, Canny/Otsu/morphology, image loading,
  export, SVG/PPTX, or metric reporting in MLX.
- Do not remove the torch optimizer path.
- Do not introduce custom Metal kernels.
- Do not make dynamic densification happen inside an MLX graph in the first
  slice.
- Do not claim parity for full training until the same stage schedule and loss
  semantics are implemented.

## Backend Shape

Add a separate optimizer/training backend concept instead of overloading the
current torch renderer factory.

Suggested public knobs:

```bash
--backend torch-batched
--device mps
--optimizer-backend torch

--backend mlx-batched
--optimizer-backend mlx
```

Resolution rules:

- `optimizer-backend=torch`: existing path.
- `optimizer-backend=mlx`: use the MLX stage runner and require MLX import.
- `backend=mlx-batched` should imply `optimizer-backend=mlx` unless the user
  explicitly picks something else and the combination is supported.
- `auto` can choose MLX only after the acceptance gates below pass.

## Data Contract

The shared interchange format remains the canonical float32 table:

```text
[N, 11]
x, y, sx, sy, theta, reserved, r, g, b, alpha, importance
```

Stage boundaries convert:

```text
List[GaussianSplat] <-> np.ndarray [N, 11] <-> mx.array [N, 11]
```

Within a stage, avoid conversions:

- Target image becomes one `mx.array` before iteration 1.
- Parameters stay as MLX arrays.
- Renderer output stays as MLX arrays.
- Loss and gradients stay as MLX arrays.
- Only progress logging, stage metrics, and final splat conversion may force
  `mx.eval()`/NumPy conversion.

## Module Layout

### `src/png2svg_gs/mlx_renderer.py`

Already exists as the forward renderer. Extend it carefully rather than folding
training code into it.

Responsibilities:

- Convert splats to canonical NumPy table.
- Build/reuse `MlxTilePlan`.
- Render `[N, 11]` tables to `[H, W, 3]`.
- Support `alpha-over` and `weighted` blend modes.

Near-term changes:

- Expose tile-plan diagnostics: `tiles`, `max_active`, `tile_size`,
  `batch_tile_count`.
- Make plan rebuild explicit and cheap to measure.
- Keep fixed-plan rendering supported for color/alpha postfit.

### `src/png2svg_gs/mlx_optimizer.py`

New module for MLX-owned parameter state and Adam.

Proposed objects:

```python
class MlxSplatParams:
    position: mx.array  # [N, 2]
    scale: mx.array     # [N, 2]
    theta: mx.array     # [N]
    color: mx.array     # [N, 3]
    alpha: mx.array     # [N]
    importance: mx.array  # [N], frozen

    def as_table(self) -> mx.array: ...
    def apply_constraints(self, width: int, height: int) -> "MlxSplatParams": ...
    def snapshot(self) -> dict[str, mx.array]: ...
    def restore(self, snapshot: dict[str, mx.array]) -> "MlxSplatParams": ...
```

Use functional updates rather than in-place mutation when that is cleaner for
MLX autograd.

```python
class MlxAdam:
    def step(self, params, grads) -> MlxSplatParams: ...
```

Adam requirements:

- One learning rate per parameter group: position, scale, theta, color, alpha.
- Bias correction.
- Epsilon matching torch Adam closely enough for convergence parity.
- Optional global gradient norm clipping before Adam.

### `src/png2svg_gs/mlx_losses.py`

New module for MLX loss functions used during stage optimization.

Implement in phases:

1. Linear RGB L1.
2. Spatial-weighted linear RGB L1.
3. OKLab L1, matching the existing transform constants.
4. Global SSIM term if it materially affects convergence.
5. Gradient/luminance edge loss only after baseline parity is understood.

Do not block the first training path on every existing loss term. Instead,
record which loss profile was used in the manifest.

### `src/png2svg_gs/mlx_stage.py`

New module for one MLX optimization stage.

Proposed function:

```python
def optimize_stage_mlx(
    splats: list[GaussianSplat],
    target_linear_rgb: np.ndarray,
    width: int,
    height: int,
    num_iters: int,
    renderer_config: MlxRendererConfig,
    learning_rates: dict[str, float],
    loss_config: MlxLossConfig,
    schedule_config: dict[str, Any],
    time_budget: Optional[TimeBudget],
) -> MlxStageResult:
    ...
```

Return:

```python
@dataclass
class MlxStageResult:
    splats: list[GaussianSplat]
    rendered_linear_rgb: np.ndarray
    metrics: dict[str, Any]
```

The converter should call this instead of `_optimize_stage()` when
`optimizer-backend=mlx`.

## Stage Semantics To Preserve

The MLX stage runner must preserve these existing torch-stage behaviors unless
explicitly disabled:

- Same starting splat table.
- Same per-group learning-rate names.
- Same parameter constraints:
  - `x` in `[0, width - 1]`
  - `y` in `[0, height - 1]`
  - `scale >= 1e-4`
  - `theta mod 2pi`
  - `color` in `[0, 1]`
  - `alpha` in `[0, 1]`
- Best-loss snapshot restore at stage end.
- Time-budget interruption.
- Progress logging with average iteration time and ETA.
- LR decay schedule or an explicitly documented MLX-compatible subset.
- Renderer tile-plan stats in stage metrics.

The initial MLX implementation may omit exact torch Adam bit parity. It must
preserve parameter grouping and convergence behavior closely enough to compare
quality on fixed seeds.

## Tile Plan Strategy

There are three increasingly capable modes.

### Mode 1: Static Plan

Build the tile plan once from initial geometry and reuse it for all iterations.

Use for:

- color/alpha-only postfit,
- fixed-geometry experiments,
- first MLX optimizer smoke.

Risk:

- position/scale training can move splats outside their original tile coverage.

### Mode 2: Periodic CPU Rebuild

Rebuild the plan every `N` iterations from the current MLX table converted to
NumPy.

Use for:

- first full parameter training,
- correctness over pure speed.

Rules:

- Default rebuild interval starts at `10`.
- Force rebuild after large scale/position movement if diagnostics show
  coverage misses.
- Count rebuilds and rebuild time in metrics.

### Mode 3: MLX-Side Conservative Plan

Avoid CPU rebuilds by using a more conservative plan:

- Inflate tile footprint radius.
- Use higher `culling_sigma`.
- Optionally cap active splats per tile only after measuring quality impact.

This may be faster than frequent rebuilds even if each tile has more active
splats.

Full device-side binning is out of scope for the first training implementation.

## Loss Plan

### First Acceptance Loss

Use simple linear RGB L1:

```text
loss = mean(abs(rendered - target))
```

Reason:

- Easy to implement.
- Differentiable.
- Enough to prove the loop and Adam.

### Fidelity Loss

Next add OKLab weighted L1:

```text
loss = mean(weight_map * abs(oklab(rendered) - oklab(target)))
```

Requirements:

- Port `torch_linear_rgb_to_oklab` constants exactly.
- Accept optional spatial weights from existing saliency/importance maps.
- Keep target OKLab precomputed once per stage.

### SSIM

Global SSIM can be added after OKLab. Windowed SSIM is not required in the hot
loop initially; final reporting already computes standard windowed SSIM outside
training.

## Optimizer Plan

### Functional Adam

MLX gradients should be produced by a pure loss function:

```python
def loss_for_params(params: MlxSplatParams) -> mx.array:
    table = params.as_table()
    rendered = renderer.render(table, plan=plan)
    return loss_fn(rendered, target)
```

Then:

```python
loss, grads = mx.value_and_grad(loss_for_params)(params)
params = adam.step(params, grads)
params = params.apply_constraints(width, height)
mx.eval(loss, params.position, params.scale, params.theta, params.color, params.alpha)
```

If `value_and_grad` cannot directly handle the class cleanly, represent params
as a dict or tuple of MLX arrays.

### Gradient Clipping

Match the torch path's global gradient norm cap:

```text
max_norm = 1.0
```

Calculate across all trainable groups before Adam.

## Converter Integration

Add a narrow dispatch point around `_optimize_stage()`:

```python
if self.optimizer_backend == "mlx":
    return optimize_stage_mlx(...)
return self._optimize_stage_torch(...)
```

Do not duplicate densification, export, or final metric code. MLX should replace
only the stage optimizer.

Manifest additions:

```json
{
  "optimizer_backend": "mlx",
  "mlx": {
    "version": "...",
    "device": "Device(gpu, 0)",
    "tile_size": 16,
    "batch_tile_count": 16,
    "tile_plan_mode": "periodic-cpu-rebuild",
    "tile_plan_rebuild_interval": 10,
    "tile_plan_rebuilds": 12,
    "loss_profile": "oklab-weighted-l1"
  }
}
```

## CLI/API

Add:

```bash
--optimizer-backend torch|mlx
--mlx-tile-plan static|periodic|conservative
--mlx-tile-plan-rebuild-interval 10
--mlx-loss linear-l1|oklab-l1|weighted-oklab-l1
```

Use existing renderer knobs:

```bash
--renderer-tile-size 16
--renderer-batch-tile-count 16
--renderer-max-active-splats-per-tile 512
```

Example smoke:

```bash
./tmp/venv-mlx/bin/python -m png2svg_gs.cli input.png \
  -o ./tmp/chameleon_400px_2000_mlx.canvas.html \
  --format canvas \
  --max-edge 400 \
  --splats 2000 \
  --stages 5 \
  --backend mlx-batched \
  --optimizer-backend mlx \
  --mlx-loss linear-l1 \
  --mlx-tile-plan static \
  --artifacts-dir ./tmp/chameleon_400px_2000_mlx_artifacts \
  -v
```

Example real comparison:

```bash
./tmp/venv-mlx/bin/python -m png2svg_gs.cli input.png \
  -o ./tmp/chameleon_400px_2000_mlx.canvas.html \
  --format canvas \
  --max-edge 400 \
  --splats 2000 \
  --stages 30,20,10 \
  --backend mlx-batched \
  --optimizer-backend mlx \
  --mlx-loss weighted-oklab-l1 \
  --mlx-tile-plan periodic \
  --mlx-tile-plan-rebuild-interval 10 \
  --artifacts-dir ./tmp/chameleon_400px_2000_mlx_artifacts \
  -v
```

## Benchmark Script Extensions

Extend `scripts/benchmark_renderer_backend.py` or add a second script:

```bash
./tmp/venv-mlx/bin/python scripts/benchmark_optimizer_backend.py \
  --input input.png \
  --max-edge 400 \
  --splats-json ./tmp/chameleon_400px_2000_mps_batched_artifacts/final.raw.json \
  --backends torch-mps,mlx \
  --iters 20 \
  --loss linear-l1 \
  --out ./tmp/optimizer_backend_benchmark.json
```

Report:

- forward median,
- forward+backward median,
- full optimizer step median,
- plan rebuild time,
- iterations/sec,
- start/end loss,
- final `ssim_srgb`,
- memory if cheaply available.

## Acceptance Gates

### Gate 1: MLX Color/Alpha Postfit

Fixed geometry, optimize only color and alpha.

Acceptance:

- Runs without CPU conversion inside iterations.
- Loss decreases on a 400px chameleon fixture.
- Final image is not worse than starting image by `ssim_srgb`.
- Per-iteration median is faster than torch-batched MPS.

### Gate 2: Full Parameter MLX Stage Smoke

Optimize position, scale, theta, color, and alpha for a short stage.

Acceptance:

- 400px chameleon, 2000 splats, `--stages 5` completes.
- No NaN/Inf params.
- Constraints hold after every iteration.
- Loss decreases.
- Manifest records MLX backend and tile-plan stats.

### Gate 3: Comparable Chameleon Run

Run the same schedule used for prior torch-batched evidence:

```text
400px, 2000 splats, stages 30,20,10
```

Acceptance:

- MLX total optimize time is faster than torch-batched MPS.
- MLX quality is within `0.01 ssim_srgb` of torch-batched MPS for same seed,
  or the loss profile difference is explicitly documented.
- No rendered starfield/background-only failure.

### Gate 4: 10k Feasibility

Run a 10k-splat photo or chameleon budget.

Acceptance:

- Completes within the chosen time budget.
- Iteration logging stays responsive.
- Plan rebuild time is not the dominant cost.
- Quality improves with splat count rather than collapsing under tile density.

## Risks

### Static Plans Can Lie

If geometry moves but the tile plan does not, splats can fail to contribute to
new tiles.

Mitigation:

- Static plans are only accepted for fixed-geometry and smoke tests.
- Full geometry training starts with periodic rebuilds.

### Loss Mismatch Can Mislead Benchmarks

A faster MLX loop with a weaker loss is not an apples-to-apples replacement.

Mitigation:

- Always record `loss_profile`.
- Compare both quality and time.
- Bring OKLab/spatial weighting online before judging final quality.

### MLX Autograd Shape Friction

Autograd over a custom parameter class may be awkward.

Mitigation:

- Use tuples or dicts of arrays if classes complicate `value_and_grad`.
- Keep conversion helpers thin and tested.

### Hidden Syncs

Calling `.item()`, `np.asarray()`, or frequent logging can force evaluation and
erase MLX gains.

Mitigation:

- Evaluate only once per iteration after the optimizer step.
- Log scalar loss at configurable intervals.
- Convert rendered output only at stage end or explicit metrics checkpoints.

## Implementation Slices

1. Add `mlx_losses.py` with linear L1 and OKLab conversion tests.
2. Add `MlxSplatParams` and functional constraints.
3. Add `MlxAdam` with per-group learning rates and gradient clipping.
4. Add `optimize_stage_mlx()` static-plan color/alpha-only mode.
5. Add benchmark script for optimizer steps.
6. Add converter CLI flags and dispatch.
7. Add periodic tile-plan rebuild mode.
8. Add weighted OKLab loss and saliency weights.
9. Run chameleon 400px/2000 parity and timing.
10. Run 10k feasibility.

## Decision

Use MLX for the Apple Silicon hot loop, but introduce it as an experimental
optimizer backend with explicit manifests and benchmark gates. Keep NumPy and
scikit-image for preprocessing. Keep torch as the correctness reference and
portable fallback.
