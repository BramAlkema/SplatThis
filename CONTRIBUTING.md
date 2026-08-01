# Contributing to SplatThis

Thank you for your interest in contributing to SplatThis! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Testing Requirements](#testing-requirements)
7. [Code Style](#code-style)
8. [Documentation](#documentation)
9. [Issue Reporting](#issue-reporting)
10. [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help create a diverse environment
- **Be collaborative**: Work together constructively and give credit where due
- **Be professional**: Keep discussions focused and constructive
- **Be patient**: Remember that everyone has different experience levels

## Getting Started

### Prerequisites

- Python 3.13 or higher
- Git
- Basic familiarity with image processing concepts
- Understanding of SVG and web technologies (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/SplatThis.git
cd SplatThis
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/BramAlkema/SplatThis.git
```

## Development Environment

### Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:

```bash
pip install -e ".[dev,capture]"
```

3. Verify installation:

```bash
splatthis --help
pytest tests/ -v
```

### Development Tools

The project uses several development tools:

- **Black**: Code formatting
- **flake8**: Style checking
- **mypy**: Type checking
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement features or fix bugs
4. **Documentation**: Improve guides, examples, and API docs
5. **Performance Improvements**: Optimize existing code
6. **Tests**: Add or improve test coverage
7. **Examples**: Create tutorials and usage examples

### Before You Start

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: For new features or significant changes
3. **Discuss approach**: Get feedback before implementing large changes
4. **Start small**: Begin with small contributions to familiarize yourself

### Coding Standards

#### Code Quality Requirements

- **Test Coverage**: Maintain >75% coverage for new code
- **Type Hints**: All public functions must have type annotations
- **Documentation**: All public APIs must have docstrings
- **Performance**: Consider performance impact of changes
- **Backwards Compatibility**: Avoid breaking existing APIs

#### Architecture Principles

- **Modularity**: Keep components loosely coupled
- **Extensibility**: Design for future enhancements
- **Testability**: Write code that's easy to test
- **Readability**: Prioritize clear, maintainable code
- **Performance**: Optimize critical paths

## Pull Request Process

### 1. Branch Strategy

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
# or
git checkout -b docs/documentation-improvement
```

### 2. Development Workflow

1. **Make your changes**:
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

2. **Run quality checks**:

```bash
# Format code
black src/ tests/ tools/

# Check style
flake8 src/ tests/ tools/

# Check import order
isort --check-only src/ tests/ tools/

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src/splatthis --cov-report=term-missing
```

3. **Commit your changes**:

```bash
git add .
git commit -m "feat: add new splat extraction algorithm

- Implement optimized SLIC segmentation
- Add tests for new functionality
- Update API documentation

Closes #123"
```

### 3. Commit Message Format

Use conventional commit format:

```
type(scope): brief description

Detailed explanation of the change, including:
- What was changed
- Why it was changed
- Any breaking changes

Closes #issue-number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 4. Submit Pull Request

1. **Push your branch**:

```bash
git push origin feature/your-feature-name
```

2. **Create pull request** on GitHub with:
   - Clear title and description
   - Reference related issues
   - Include screenshots/examples if applicable
   - List any breaking changes

3. **Pull request template**:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #123
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Validate optimization targets
4. **Regression Tests**: Prevent functionality breaks

### Writing Tests

#### Unit Test Example

```python
import numpy as np
import pytest

from splatthis.splat import create_isotropic_splat
from splatthis.io import generate_svg_content


def test_svg_content_emits_one_gradient_per_splat():
    """Each splat should get its own radial gradient in the emitted SVG."""
    splats = [
        create_isotropic_splat(
            center=np.array([10.0, 10.0]),
            sigma=2.0,
            color=np.array([1.0, 0.2, 0.2]),
            alpha=0.8,
        )
    ]
    svg = generate_svg_content(splats, width=32, height=24, k_sigma=2.5)

    assert svg.count("<radialGradient") == len(splats)
    assert 'fill="url(#splat_grad_' in svg
```

#### Integration Test Example

```python
def test_full_pipeline_integration(tmp_path):
    """Image in, parseable SVG out, through the real converter."""
    import xml.etree.ElementTree as ET

    from PIL import Image
    from splatthis.converter import PNG2SVGConverter

    src = tmp_path / "tiny.png"
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    ).save(src)

    out = tmp_path / "tiny.svg"
    converter = PNG2SVGConverter(
        max_splats=16, stages=[2, 1], optimizer_backend="torch"
    )
    converter.convert(str(src), output_path=str(out), verbose=False)

    assert out.exists()
    ET.parse(out)  # emitted SVG must be well-formed
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/test_extract.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Performance tests only
pytest tests/integration/test_performance_benchmarks.py

# Skip slow tests
pytest tests/ -m "not slow"
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# String quotes: Use double quotes for strings
# Imports: Organized with isort

# Example function
def extract_splats(
    self,
    image: np.ndarray,
    n_splats: int = 1500,
    min_size: int = 4
) -> List[Gaussian]:
    """Extract Gaussian splats from image.

    Args:
        image: Input image as numpy array (H, W, 3)
        n_splats: Target number of splats to extract
        min_size: Minimum splat size in pixels

    Returns:
        List of Gaussian splats sorted by importance

    Raises:
        ValueError: If image format is invalid
        RuntimeError: If extraction fails
    """
    # Implementation here
    pass
```

### Type Annotations

All public functions must have type hints:

```python
from typing import List, Optional, Dict, Union
import numpy as np

def process_layers(
    layers: Dict[int, List[Gaussian]],
    options: Optional[Dict[str, Union[str, int]]] = None
) -> str:
    """Process layers with optional configuration."""
    pass
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """Brief description of function.

    Longer description with more details about what this function
    does and how it works.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 with default value

    Returns:
        Dictionary containing results with keys:
            - 'result': Main result value
            - 'metadata': Additional information

    Raises:
        ValueError: When param1 is empty
        RuntimeError: When processing fails

    Example:
        >>> result = complex_function("test", 20)
        >>> print(result['result'])
        'processed_test'
    """
    pass
```

## Documentation

### Types of Documentation

1. **API Documentation**: Comprehensive function/class docs
2. **User Guides**: How-to guides and tutorials
3. **Examples**: Code examples and use cases
4. **README**: Project overview and quick start

### Documentation Workflow

1. **Update docstrings** for any new/modified functions
2. **Add examples** for new features
3. **Update README** if API changes
4. **Create tutorials** for complex features

### Building Documentation

```bash
# Check documentation coverage
python -c "
import pydoc
import splatthis
help(splatthis)
"

# Validate examples in documentation
```

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug

## Environment
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
- Python version: [e.g., 3.9.7]
- SplatThis version: [e.g., 0.2.0]

## Steps to Reproduce
1. Load image with `splatthis image.jpg`
2. See error message

## Expected Behavior
What should have happened

## Actual Behavior
What actually happened

## Error Messages
```
Full error traceback here
```

## Additional Context
- Sample image (if possible)
- Configuration used
- Any workarounds found
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why would this feature be useful?

## Proposed Implementation
Ideas for how this could be implemented

## Alternatives Considered
Other approaches that were considered

## Additional Context
Any other relevant information
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration

### Getting Help

1. **Check documentation** first
2. **Search existing issues** for similar problems
3. **Create new issue** with detailed information
4. **Be patient** - maintainers volunteer their time

### Helping Others

- **Answer questions** in issues and discussions
- **Review pull requests** from other contributors
- **Improve documentation** based on common questions
- **Share examples** of your usage

## Recognition

Contributors are recognized in several ways:

- **Contributors list** in README
- **Changelog mentions** for significant contributions
- **Maintainer privileges** for consistent contributors
- **Conference speaking** opportunities

Thank you for contributing to SplatThis! Your efforts help make image-to-SVG conversion accessible to everyone. 🎨✨
