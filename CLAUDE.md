# Claude Development Notes

## Virtual Environment
**IMPORTANT: Always activate the virtual environment before running any Python commands!**

```bash
source venv/bin/activate
```

## Project Structure

The live pipeline is `src/png2svg_gs/` (the legacy `src/splat_this` package was
retired). CLI entry point: `splatlify` (= `png2svg_gs.cli:main`).

| Module | Role |
|---|---|
| `cli.py` | `splatlify` argument parsing, resource-limit resolution |
| `converter.py` | `PNG2SVGConverter` orchestrator: init → staged optimize → densify/prune → postfit → emit |
| `features.py` | Seeding: gradient PDF, structure tensor (`edge_tangent_angle`), Poisson disk |
| `optimizer.py` | Torch `SplatParams` + per-group Adam LRs |
| `renderer.py` | Torch reference renderers (tiled + batched), `L1SSIMLoss`, numpy validator |
| `mlx_*.py` | MLX backend (default on Apple Silicon): renderer, losses, fused-Adam stage |
| `io.py` | Emission: SVG recipes, PPTX/DrawingML, native Canvas/CSS HTML, explicit ImageData pixel runtime, quality metrics |
| `splat.py` | `GaussianSplat`, layer bands, render-order keys |

Splats flow between stages as `List[GaussianSplat]` ⇄ `[N, 11]` tensors
(x, y, sx, sy, theta, reserved, r, g, b, alpha, importance).

- `tests/unit/` — the test suite (`tests/integration` is orphaned)
- `tools/fidelity_lab.py`, `scripts/benchmark_*.py` — experiment harnesses
- `docs/` — GitHub Pages landing + research notes (`SVG_PPTX_GAUSSIAN_TRICKS.md`)
- `external/`, `ml-sharp/` — local clones of reference projects, not committed

## Testing Commands
```bash
# Full unit suite with coverage (what CI runs)
PYTHONPATH=. pytest tests/unit/ --cov=src/png2svg_gs --cov-report=term-missing --tb=short

# Single file, fast
PYTHONPATH=. pytest tests/unit/test_mlx_renderer.py -v --tb=short --no-cov

# Formatters are pinned; CI enforces black --check
black src/png2svg_gs/ tests/unit/
```

## Running the Pipeline
```bash
splatlify docs/demo/source.png -o out.svg --seed 42 --artifacts-dir artifacts/
# Key flags: --profile (default max-fidelity), --splats, --stages, --time-budget,
# --optimizer-backend {mlx,torch}, --format {svg,pptx,canvas,css,pixel-runtime},
# --svg-recipe, --training-export-target {auto,pixel-runtime,browser-gradient,svg,pptx-softedge}
```
MLX is the default optimizer backend (falls back to torch with a warning when
mlx is missing). Torch is the cross-platform reference; keep the two in parity —
`tests/unit/test_mlx_losses.py` and `test_mlx_renderer.py` pin it.

## Fidelity Protocol (non-negotiable)
- Judge native Canvas, browser SVG, and CSS quality on the native-size
  **Playwright Chromium capture** — never on librsvg, CairoSVG, or the internal
  renderer alone. `pixel-runtime` is the separate ImageData software renderer.
- **LPIPS is the trusted metric**; SSIM over-rewards blur. Always eyeball a
  side-by-side before claiming a win; keep wins, revert washes.
- Models trained for SVG/PPTX targets composite in **sRGB** (browsers blend in
  display space); validation renders must pass `compositing_space="srgb"`.
- **Never validate PPTX with soffice/LibreOffice** (known rendering bugs).
  Use `openxml-audit` (local: `~/projects/openxml-audit`) for structure and the
  real-PowerPoint capture tooling in `~/projects/svg2ooxml/tools/ppt_research/`
  for visuals.
- Headless-Chrome canvas screenshots: never set `--window-size` equal to the
  canvas dimensions (bottom rows come out black) — oversize and crop.

## Conventions
- Splat orientation comes from the structure tensor's gradient direction; any
  anisotropic splat creation must go through `features.edge_tangent_angle()`
  (major axis along the edge = gradient direction + π/2).
- Compositing order is ascending importance everywhere (torch stable argsort,
  MLX/numpy stable sorts, SVG document order); don't introduce unstable sorts.
- SVG uses per-splat baked colors — a shared `currentColor` gradient breaks rsvg.
- `convert()` must not leave run-mutated config on the instance; per-run state
  is snapshot/restored in the `convert()` wrapper.
