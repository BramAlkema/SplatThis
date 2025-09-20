# SplatThis 🎨

Convert images into self-contained parallax-animated SVG splats.

## Installation

```bash
# Clone the repository
git clone https://github.com/BramAlkema/SplatThis.git
cd SplatThis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Basic usage (requires test image)
splatlify input.jpg -o output.svg

# View help
splatlify --help
```

## Current Status

✅ **Completed (T1.1: Project Setup & Structure)**
- ✅ Python package structure with proper `__init__.py` files
- ✅ `pyproject.toml` with dependencies and entry points
- ✅ Development tools configuration (black, mypy, pytest, flake8)
- ✅ Basic CLI entry point with Click framework
- ✅ GitHub Actions CI pipeline setup
- ✅ Comprehensive test suite foundation

**CLI Command:**
```bash
splatlify --help  # ✅ Working
```

**Next Steps:**
- T1.2: Image Loading & Validation (implement actual image processing)
- T1.3: SLIC Superpixel Implementation (integrate scikit-image)
- T1.4: Enhanced Gaussian Splat Data Structure
- And continuing through the 4-week implementation plan...

## Development

```bash
# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/

# Type checking
mypy src/

# Style checking
flake8 src/ tests/
```

## Architecture

```
src/splat_this/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface ✅
├── core/
│   ├── extract.py       # Splat extraction (placeholder)
│   ├── layering.py      # Depth assignment (placeholder)
│   └── svgout.py        # SVG generation (placeholder)
└── utils/
    ├── image.py         # Image utilities (placeholder)
    └── math.py          # Mathematical helpers ✅
```

## Contributing

1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Run quality checks: `black src/ tests/ && flake8 src/ tests/ && pytest`
3. All checks must pass before committing

This project follows the implementation plan detailed in `.agent-os/specs/2025-01-20-splat-cli-core/`.