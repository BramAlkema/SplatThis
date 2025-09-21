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
# Convert an image to animated SVG
splatlify photo.jpg -o parallax.svg

# High-quality output with more splats
splatlify landscape.png --splats 3000 --layers 6 -o detailed.svg

# Enable Gaussian gradient mode for smoother results
splatlify portrait.jpg --gaussian --splats 2000 -o smooth.svg
```

### Advanced Examples
```bash
# Interactive parallax with top-layer elements
splatlify cityscape.jpg --interactive-top 500 --parallax-strength 80 -o interactive.svg

# Process specific GIF frame
splatlify animation.gif --frame 10 --splats 1500 -o frame10.svg

# Fine-tune splat size and transparency
splatlify texture.png --k 3.0 --alpha 0.8 --layers 5 -o custom.svg
```

## 📖 Documentation

### Command Line Options

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `INPUT_FILE` | Source image (JPG, PNG, GIF) | Required | - |
| `--output, -o` | Output SVG file path | Required | - |
| `--splats` | Number of Gaussian splats | 2500 | 100-15000 |
| `--layers` | Depth layers for parallax | 4 | 2-8 |
| `--k` | Splat size multiplier | 1.2 | 0.5-5.0 |
| `--alpha` | Base transparency | 0.65 | 0.1-1.0 |
| `--parallax-strength` | Motion sensitivity | 40 | 0-200 |
| `--interactive-top` | Interactive top splats | 0 | 0-5000 |
| `--gaussian` | Enable gradient mode | False | - |
| `--frame` | GIF frame to process | 0 | - |
| `--verbose, -v` | Detailed output | False | - |

### Python API

```python
from splat_this import SplatExtractor, LayerAssigner, SVGGenerator

# Load and process image
extractor = SplatExtractor()
splats = extractor.extract_splats("input.jpg", n_splats=2000)

# Assign depth layers
assigner = LayerAssigner(n_layers=5)
layered_splats = assigner.assign_layers(splats)

# Generate animated SVG
generator = SVGGenerator()
svg_content = generator.generate_svg(
    layered_splats,
    parallax_strength=60,
    interactive_top=300
)

# Save result
with open("output.svg", "w") as f:
    f.write(svg_content)
```

### Optimized Components

For performance-critical applications, use the optimized components:

```python
from splat_this.core.optimized_extract import OptimizedSplatExtractor
from splat_this.core.optimized_svgout import OptimizedSVGGenerator

# Up to 1.85x faster processing
extractor = OptimizedSplatExtractor()
generator = OptimizedSVGGenerator()
```

## 🎨 How It Works

1. **Image Analysis**: SLIC superpixel segmentation identifies key regions
2. **Splat Extraction**: Converts regions into Gaussian splats with position, size, and color
3. **Quality Scoring**: Importance-based filtering keeps the most visually significant splats
4. **Layer Assignment**: Distributes splats across depth layers for parallax effect
5. **SVG Generation**: Creates animated SVG with CSS transforms and JavaScript interaction

## 🏗️ Architecture

```
src/splat_this/
├── core/
│   ├── extract.py           # Splat extraction engine
│   ├── layering.py          # Depth layer assignment
│   ├── svgout.py            # SVG generation
│   ├── optimized_extract.py # Performance-optimized extractor
│   ├── optimized_layering.py# Optimized layer assignment
│   └── optimized_svgout.py  # Optimized SVG generation
├── utils/
│   ├── image.py             # Image loading and validation
│   ├── math.py              # Mathematical utilities
│   ├── profiler.py          # Performance profiling
│   └── optimized_io.py      # Optimized I/O operations
├── cli.py                   # Command-line interface
└── optimized_cli.py         # Performance CLI variant
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

### Basic Parallax Effect
```bash
splatlify sunset.jpg -o sunset_parallax.svg
```

### High-Quality Portrait
```bash
splatlify portrait.png --gaussian --splats 3000 --layers 6 --alpha 0.8 -o portrait_hq.svg
```

### Interactive Landscape
```bash
splatlify landscape.jpg --interactive-top 800 --parallax-strength 100 --k 2.8 -o landscape_interactive.svg
```

### Animation Frame
```bash
splatlify animation.gif --frame 15 --splats 2500 --layers 5 -o frame15.svg
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
# Increase splat count and layers
splatlify image.jpg --splats 3000 --layers 6 --gaussian -o high_quality.svg
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