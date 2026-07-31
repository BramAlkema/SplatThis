# Changelog

All notable changes to SplatThis are documented here.

## 0.2.0 - 2026-07-30

### Added

- Target-aware Canvas, SVG, and native DrawingML/PPTX export pipelines.
- MLX optimization with periodic tile-plan rebuilds and full geometry training.
- Actual-artifact SVG evaluation and real-PowerPoint corpus capture tooling.
- Monotonic Canvas checkpoint and post-processing gates.
- Browser-pixel-buffer corpus metrics, per-picture comparisons, and budget history.
- Optional layered Canvas parallax.
- Experimental Top-K distillation and mixed-primitive MVP tooling.

### Changed

- MLX now defaults to eight tiles per batch. Three real corpus checkpoints
  measured 12-34% lower median forward-and-backward latency than a 16-tile
  batch without changing render math; on the Chameleon converter-default
  comparison it was 23% lower than 128 tiles.
- The corpus overview renders canvases lazily near the viewport.
- Canvas splats are sorted once during export instead of at every page load.
- Internal PNGs are named and described as splat proxies, not screenshots of
  the emitted SVG or PPTX.
- Package metadata and documentation now report full-corpus, deployed-artifact
  results instead of extrapolating from a favorable single image.

### Fixed

- Correct rotated anisotropic footprint bounds in Canvas and NumPy rendering.
- Fall back to Torch when MLX imports but no Metal device is available.
- Publish SVG, PPTX, Canvas, JSON, manifest, and proxy-PNG outputs atomically.
- Escape Canvas document titles and reject invalid canvas dimensions.
- Report the real number of parallax layers instead of JSON-string length.
- Return concise CLI errors for invalid or unreadable input.

### Compatibility

- Python 3.13 or newer is required.
- MLX remains optional and requires an Apple-Silicon Metal session.
- PPTX quality claims refer to Microsoft PowerPoint slideshow captures;
  internal proxies and LibreOffice renders are labeled separately.
