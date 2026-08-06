# Contributing

Keep changes on the shared fitter and its five direct emitters: SVG, PPTX,
Canvas, CSS, and EML. Please propose unrelated output targets or research
systems separately rather than growing the core package again.

MLX, browser capture, and low-bit population embedding must remain import-safe
without their extras installed. Do not add those packages to the default
dependency set.

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
python -m build
```

Python 3.11 is the supported floor and must pass the same suite and package
build as the current CI interpreter.

Bug fixes should include a focused regression test. Fitter changes must
preserve the end-to-end numerical floor; emitter changes must preserve the
format contracts documented in the README.
