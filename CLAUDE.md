# Development notes

SplatThis has one product path: image to a fitted Gaussian population to one of
five direct emitters: SVG, PPTX, Canvas, CSS, or EML. Keep additions on that
path. Publishing systems and experiment harnesses belong in separate projects.

The package lives in `src/splatthis/` and the command is `splatthis`.

```bash
source venv/bin/activate
ruff check .
ruff format --check .
pytest
```

Core invariants:

- Torch is the portable default; MLX is an optional optimizer selected
  explicitly and must fall back before fitting when Metal is unavailable.
- Splat orientation follows the local edge tangent.
- Emitters translate the fitter's front-to-back importance order into each
  target's painter order.
- SVG and CSS outputs are static and self-contained.
- PPTX uses editable DrawingML shapes and embeds no raster fallback.
- Canvas uses native 2D gradients, not an ImageData or WebGL payload.
- EML stays script-free, keeps scene declarations inline, and remains below
  its HTML size guard.
- Browser capture is optional and governs only SVG, CSS, and Canvas artifacts.
- PPTX capture uses Microsoft PowerPoint through macOS OSA and adds no Python
  dependency; keep that operating-system adapter isolated from emitters.
- Population carriers remain self-describing; pixel LSB embedding stays opt-in
  because it changes pixels and has finite capacity.
- A fixed seed on Torch CPU must be deterministic within metric tolerance.
- Internal metrics are regression evidence, not browser-fidelity claims.
