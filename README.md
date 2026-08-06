# SplatThis

SplatThis fits a bitmap with anisotropic 2D Gaussian splats and exports the
fitted scene in five portable formats:

| Format | Output | Runtime behavior |
| --- | --- | --- |
| `svg` | Static SVG | Script-free gradients and ellipses |
| `pptx` | PowerPoint | Native editable DrawingML shapes; no bitmap |
| `canvas` | HTML | Self-contained Canvas 2D gradient renderer |
| `css` | HTML | Script-free CSS gradient elements |
| `eml` | Email message | Inline email-safe CSS plus plain-text and Outlook fallbacks |

The fitting path is deliberately singular: Pillow loads the image, NumPy
computes image structure, Torch fits one Gaussian population, and a direct
emitter writes the selected artifact. MLX, Chromium capture, native PowerPoint
OSA capture, and in-pixel population recovery are isolated optional
capabilities; there is no WebGL runtime or external asset service.

## Install

SplatThis requires Python 3.11 or newer.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

The complete runtime dependency set is NumPy, Pillow, and Torch. PPTX, HTML,
CSS, and EML generation add no packages.

Install optional capabilities only when needed:

```bash
pip install -e ".[mlx]"      # Apple-Silicon MLX optimizer
pip install -e ".[capture]"  # Playwright client; uses installed Chrome/Chromium
pip install -e ".[steg]"     # low-bit PNG population carrier
```

## Use

SVG remains the default:

```bash
splatthis input.png -o output.svg
```

Select another artifact with `--format`:

```bash
splatthis input.png --format pptx   -o output.pptx
splatthis input.png --format canvas -o output-canvas.html
splatthis input.png --format css    -o output-css.html
splatthis input.png --format eml    -o output.eml
```

Without `-o`, SplatThis chooses `.svg`, `.pptx`, `.html`, or `.eml` from the
format. Canvas and CSS both use `.html`, so `--format` remains authoritative.

The fitting controls are intentionally few:

```bash
splatthis input.jpg --format pptx \
  --splats 3000 \
  --stages 300,200,100 \
  --max-edge 512 \
  --device cpu \
  --seed 42
```

- `--splats` caps the final Gaussian population.
- `--stages` controls optimization work at each densification stage.
- `--max-edge` bounds the fitted resolution while preserving aspect ratio.
- `--profile` selects `fast`, `balanced`, or `max-fidelity` defaults.
- `--artifacts-dir` writes raw populations, metrics, and the run manifest.
- `--save-json` writes the final raw population beside the artifact.
- `--preview` writes an internal NumPy render for diagnostics.

## Optional capabilities

Use MLX explicitly on an Apple-Silicon Metal session:

```bash
splatthis input.png --optimizer-backend mlx -o output.svg
```

If MLX is missing or cannot allocate a Metal device, SplatThis warns and falls
back to Torch before fitting. Torch remains the portable default.

Capture the deployed SVG, CSS, or Canvas artifact through Chromium:

```bash
splatthis input.png --format css -o output.html --capture capture.png
```

The capture extra installs the Playwright client, not a bundled browser.
SplatThis uses an installed Chrome/Chromium executable, optionally selected
with `--capture-browser` or `SPLATTHIS_BROWSER_EXECUTABLE`.

On macOS, the same switch captures PPTX through the native Microsoft
PowerPoint renderer using OSA and the system screenshot utility:

```bash
splatthis input.png --format pptx -o output.pptx --capture powerpoint.png
```

This PPTX path adds no Python dependency. It requires PowerPoint and may prompt
for macOS Automation, Accessibility, and Screen Recording permissions. It
closes only the presentation opened for capture and does not quit PowerPoint.

Make a fitted population recoverable from an artifact:

```bash
splatthis input.png -o output.svg --embed-population
splatthis input.png -o output.svg \
  --preview preview.png \
  --embed-population \
  --embed-population-in-pixels
```

SVG and PPTX carry a compressed, self-describing population envelope. Preview
PNG files carry the same envelope in a text chunk. The `steg` extra can also
hide it in one or two low bits per color channel, so it survives metadata
stripping; this costs pixel fidelity and has finite capacity. These payloads
are derivatives of the source image and should be treated accordingly.

For EML, `--email-to`, `--email-from`, and `--email-subject` set message
headers. `--email-splats` defaults to 285 to leave room below Gmail's 102 KB
HTML clipping threshold. The message uses inline legacy CSS and margin-based
placement. Word-based Outlook receives a deliberate fallback because it cannot
render the gradients; mail-client inspection is still required before sending.

## Python API

```python
from splatthis import SplatConverter

converter = SplatConverter(
    max_splats=2000,
    stages=[200, 150, 100, 50],
    quality_profile="max-fidelity",
    device="cpu",
    seed=42,
)
converter.convert("input.png", "output.pptx", output_format="pptx")
```

`SplatConverter` is the neutral public name; the historical
`PNG2SVGConverter` name remains supported. Direct emitter functions for SVG,
PPTX, Canvas, CSS, and EML are also exported from `splatthis` for code that
already has a `GaussianSplat` population. `load_population()` recovers embedded
SVG, PPTX, PNG-text, or PNG-pixel populations.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
python -m build
```

Tests enforce the numerical quality floor, each artifact contract, optional
dependency isolation, browser-capture geometry, PowerPoint OSA routing, MLX
fallback, and population round-trips through SVG, PPTX, PNG metadata, and PNG
low bits.

## Scope

Version 0.3 is intentionally breaking. It retains the five deliverables plus
isolated MLX, Chromium capture, native PowerPoint OSA capture, and population
carriers. It removes the experimental pipeline framework, WebGL and pixel
runtimes, corpus website, research harnesses, generated ledgers, and one-off
tools. Git history retains that experimental work without keeping it in the
package.

## License

MIT. See [LICENSE](LICENSE).
