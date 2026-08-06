# Changelog

## 0.3.0

Breaking simplification:

- Keep one product path: bitmap to one fitted Gaussian population to direct
  SVG, native PPTX, Canvas HTML, CSS HTML, or CSS-in-EML emission.
- Keep one numerical backend: Torch.
- Reduce runtime dependencies to NumPy, Pillow, and Torch.
- Retain Canvas, CSS, email-safe CSS, and editable PowerPoint as small emitters
  with no additional runtime dependencies.
- Retain MLX optimization, Chromium capture, and population embedding as
  isolated opt-in capabilities with separate extras.
- Retain the macOS OSA screenshotter from `../svg2pptx` as a dependency-free
  native Microsoft PowerPoint capture path for generated PPTX files.
- Remove WebGL/pixel-runtime, fidelity-lab, corpus-publishing, and compatibility
  frameworks.
- Remove generated corpus artifacts, experiment ledgers, research documents,
  and one-off tools from the package repository.
- Replace SciPy and scikit-image feature/metric calls with local NumPy code.
- Replace Black, isort, flake8, and mypy maintenance with Ruff plus pytest.

The 0.2 history remains available in Git.
