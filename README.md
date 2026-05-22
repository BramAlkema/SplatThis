# SplatThis 🎨✨

Transform any image into stunning parallax-animated SVG graphics with dynamic Gaussian splats.

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Coverage](https://img.shields.io/badge/coverage-79%25-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-242%20passing-brightgreen.svg)

## 🚀 Features

- **Parallax Animation**: Creates depth-based motion effects on mouse interaction
- **Gaussian Splat Technology**: Advanced image decomposition into smooth circular gradients
- **Performance Optimized**: Up to 1.85x faster processing with memory-efficient algorithms
- **Cross-Browser Compatible**: SVG 1.1 compliant output works everywhere
- **Mobile Ready**: Gyroscope support for device motion parallax
- **Accessibility First**: Respects user motion preferences and screen readers
- **Batch Processing**: Process multiple images efficiently
- **High Quality Output**: Maintains visual fidelity with optimized splat extraction

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/BramAlkema/SplatThis.git
cd SplatThis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Quick Install (Coming Soon)
```bash
pip install splat-this
```

## 🎯 Quick Start

### Basic Usage
```bash
# Convert an image to SVG (2D Gaussian splatting)
splatlify photo.jpg -o output.svg

# More splats + a longer training schedule for higher fidelity
splatlify landscape.png --splats 3000 --stages 400,300,200,150 -o detailed.svg

# Downscale large inputs for faster runs
splatlify portrait.jpg --max-edge 512 --splats 2000 -o portrait.svg

# Export PowerPoint (DrawingML) instead of SVG
splatlify logo.png --format pptx -o logo.pptx
```

## 📖 Documentation

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `INPUT` | Source image (PNG/JPG) | Required |
| `--output, -o` | Output path | `<input>.svg` |
| `--splats` | Max number of Gaussian splats | 2000 |
| `--stages` | Per-stage iteration schedule (comma-separated) | 200,150,100,50 |
| `--profile` | Quality profile | max-fidelity |
| `--blend-mode` | Compositing mode (`alpha-over` / `weighted`) | alpha-over |
| `--max-edge` | Downscale so the longest edge is at most N px | none |
| `--format` | Output format (`svg` / `pptx`) | svg |
| `--device` | Torch device (`cpu` / `cuda`) | cpu |
| `--seed` | Deterministic seed | 0 |
| `--artifacts-dir` | Directory for run manifest + iteration dumps | none |
| `--verbose, -v` | Detailed logging | False |

### Python API

```python
from png2svg_gs.converter import PNG2SVGConverter

converter = PNG2SVGConverter(
    max_splats=2000,
    stages=[200, 150, 100, 50],
    quality_profile="max-fidelity",
)
converter.convert(input_path="input.jpg", output_path="output.svg")
```

## 🎨 How It Works

1. **Content-adaptive initialization**: places 2D Gaussians guided by image gradients (more where there's detail).
2. **Differentiable optimization**: a torch renderer + L1/SSIM loss (in perceptual OKLab space) refines each splat's position, anisotropic covariance, color, and opacity via SGD.
3. **Progressive densification/pruning**: stages add splats in high-error regions and prune low-impact ones, up to the splat budget.
4. **SVG export**: each splat becomes an SVG ellipse with a true-Gaussian radial-gradient falloff (also exportable to PowerPoint DrawingML).
5. **Honest quality gating**: the exported SVG is rasterized and scored (windowed SSIM, perceptual sRGB) against the source.

## 🏗️ Architecture

```
src/png2svg_gs/
├── converter.py   # PNG2SVGConverter: orchestration, stages, acceptance
├── renderer.py    # differentiable torch renderer + L1SSIMLoss + color transforms
├── optimizer.py   # Adam wrapper with per-group learning rates
├── losses.py      # optional perceptual/edge losses
├── splat.py       # Gaussian splat model + raw schema
├── features.py    # gradient/feature maps for initialization
├── io.py          # SVG + DrawingML/PPTX export, rasterization, quality metrics
└── cli.py         # `splatlify` command-line entry point
```

## ⚡ Performance

### Optimization Results
| Image Size | Standard Time | Optimized Time | Improvement |
|------------|---------------|----------------|-------------|
| Small (256×256) | 0.24s | 0.15s | **1.53x faster** |
| Medium (512×512) | 0.84s | 0.50s | **1.66x faster** |
| Large (1024×1024) | 2.54s | 1.46s | **1.74x faster** |
| HD (1920×1080) | 9.79s | 5.28s | **1.85x faster** |

### Memory Efficiency
- **Peak Usage**: <500MB for large images
- **Automatic Downsampling**: Memory-constrained optimization
- **Garbage Collection**: Proactive cleanup

## 🧪 Quality Assurance

- **242 Unit Tests** with 79% coverage
- **Integration Tests** for complete pipeline validation
- **Performance Benchmarks** ensuring optimization targets
- **Browser Compatibility** testing across modern browsers
- **Visual Regression** testing for output consistency
- **Accessibility Compliance** with WCAG guidelines

## 🌐 Browser Support

| Feature | Chrome | Firefox | Safari | Edge | Mobile |
|---------|--------|---------|--------|------|--------|
| SVG Animation | ✅ | ✅ | ✅ | ✅ | ✅ |
| CSS Transforms | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mouse Parallax | ✅ | ✅ | ✅ | ✅ | - |
| Gyroscope | ✅ | ✅ | ✅ | ✅ | ✅ |
| Accessibility | ✅ | ✅ | ✅ | ✅ | ✅ |

## 🔧 Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/

# Type checking
mypy src/

# Style checking
flake8 src/ tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Running Tests
```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance benchmarks
pytest tests/integration/test_performance_benchmarks.py

# Specific test
pytest tests/unit/test_extract.py::TestSplatExtractor::test_basic_extraction -v
```

## 🚀 Deployment

### GitHub Actions
Automated CI/CD pipeline includes:
- Code quality checks (black, flake8, mypy)
- Comprehensive test suite
- Performance benchmarks
- Coverage reporting
- Cross-platform testing

### Production Build
```bash
# Build distribution
python -m build

# Install from wheel
pip install dist/splat_this-0.1.0-py3-none-any.whl
```

## 📝 Examples

### Quick conversion
```bash
splatlify sunset.jpg -o sunset.svg
```

### High-quality portrait
```bash
splatlify portrait.png --splats 3000 --stages 400,300,200,150 -o portrait_hq.svg
```

### Fast preview of a large image
```bash
splatlify landscape.jpg --max-edge 512 --splats 1500 --stages 100,80 -o landscape.svg
```

## 🐛 Troubleshooting

### Common Issues

**Memory Error with Large Images**
```bash
# Use automatic downsampling
splatlify large_image.jpg --splats 1000 -o output.svg
```

**SVG Not Displaying**
- Ensure output file has `.svg` extension
- Check browser console for JavaScript errors
- Verify SVG syntax with online validators

**Poor Quality Output**
```bash
# Increase splat count and train longer
splatlify image.jpg --splats 3000 --stages 400,300,200,150 -o high_quality.svg
```

**Performance Issues**
```bash
# Use optimized components automatically with verbose output
splatlify image.jpg -v -o output.svg
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Run quality checks**: `black src/ tests/ && flake8 src/ tests/ && pytest`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation
- Ensure all tests pass
- Maintain >75% test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-image](https://scikit-image.org/) for image processing
- [Pillow](https://pillow.readthedocs.io/) for image I/O
- [Click](https://click.palletsprojects.com/) for CLI framework
- Inspired by modern web animation techniques

## 📊 Project Status

- ✅ **Core Functionality**: Complete splat extraction and SVG generation
- ✅ **Performance Optimization**: 1.85x speedup achieved
- ✅ **Testing & QA**: 79% coverage, 242 passing tests
- ✅ **Documentation**: Comprehensive guides and examples
- 🚧 **Distribution**: PyPI package preparation
- 🔮 **Future**: Web interface, batch processing UI

---

**Ready to transform your images? Start with `splatlify --help`** 🎨✨
