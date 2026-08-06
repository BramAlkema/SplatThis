# SplatThis

Turn a PNG or JPEG into a portable scene of editable 2D Gaussian splats.

SplatThis fits one Gaussian population and exports it directly to five formats:

| Format | Artifact |
| --- | --- |
| SVG | Static, script-free vector gradients |
| PPTX | Native editable PowerPoint shapes, without a bitmap fallback |
| Canvas | Self-contained HTML using the Canvas 2D API |
| CSS | Script-free HTML made from CSS gradients |
| EML | An email message with inline CSS and plain-text/Outlook fallbacks |

The default installation is deliberately small: NumPy, Pillow, and Torch.
MLX, browser capture, and low-bit population embedding are opt-in.

## Install

SplatThis requires Python 3.11 or newer.

```bash
pip install splatthis
```

Optional capabilities are installed separately:

```bash
pip install "splatthis[mlx]"      # MLX on Apple Silicon
pip install "splatthis[capture]"  # Playwright client for Chrome/Chromium
pip install "splatthis[steg]"     # low-bit PNG population carrier
```

## Command line

SVG is the default output:

```bash
splatthis input.png -o output.svg
```

Select another target with `--format`:

```bash
splatthis input.png --format pptx   -o output.pptx
splatthis input.png --format canvas -o output-canvas.html
splatthis input.png --format css    -o output-css.html
splatthis input.png --format eml    -o output.eml
```

Control the fit with a small set of explicit options:

```bash
splatthis input.jpg --format pptx \
  --splats 3000 \
  --stages 300,200,100 \
  --max-edge 512 \
  --seed 42 \
  -o output.pptx
```

Use MLX explicitly on an Apple-Silicon Metal session:

```bash
splatthis input.png --optimizer-backend mlx -o output.svg
```

If MLX cannot allocate a Metal device, SplatThis warns and falls back to
Torch before fitting.

## Capture native output

Capture SVG, CSS, or Canvas output with an installed Chrome/Chromium browser:

```bash
splatthis input.png --format css -o output.html --capture capture.png
```

On macOS, PPTX capture uses installed Microsoft PowerPoint through OSA and the
system screenshot utility, with no additional Python package:

```bash
splatthis input.png --format pptx -o output.pptx --capture powerpoint.png
```

macOS may request Automation, Accessibility, and Screen Recording permission.

## Recoverable populations

SVG and PPTX can carry the compressed population that produced them. Preview
PNGs can carry the same envelope in metadata and, with the `steg` extra, in
one or two low bits per channel:

```bash
splatthis input.png -o output.svg --embed-population
splatthis input.png -o output.svg \
  --preview preview.png \
  --embed-population \
  --embed-population-in-pixels
```

Embedded populations are derivatives of the input image. Pixel embedding is
not a security feature, changes pixels slightly, and has finite capacity.

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

The historical `PNG2SVGConverter` name remains available. Direct emitters are
also public for applications that already have a `GaussianSplat` population.

## Project

- [Source and full documentation](https://github.com/BramAlkema/SplatThis)
- [Issue tracker](https://github.com/BramAlkema/SplatThis/issues)
- [Security policy](https://github.com/BramAlkema/SplatThis/blob/main/SECURITY.md)

SplatThis is released under the MIT License.
