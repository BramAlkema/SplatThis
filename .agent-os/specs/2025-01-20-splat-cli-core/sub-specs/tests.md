# Test Specification

**Parent Spec:** @../spec.md
**Created:** 2025-01-20
**Version:** 1.0.0

## Testing Strategy Overview

Comprehensive testing approach covering unit tests, integration tests, performance validation, and compatibility verification for the SplatThis CLI tool.

## Test Categories

### 1. Unit Tests

#### Image Loading Tests (`test_image_utils.py`)

```python
def test_load_png_image():
    """Test PNG image loading with various bit depths."""

def test_load_jpg_image():
    """Test JPEG image loading with different quality levels."""

def test_load_gif_frame_extraction():
    """Test GIF frame extraction with various frame indices."""

def test_invalid_image_format():
    """Test error handling for unsupported formats."""

def test_corrupted_image_handling():
    """Test graceful handling of corrupted image files."""

def test_image_size_validation():
    """Test minimum and maximum size constraints."""
```

**Test Data:**
- Sample images: 100x100 to 4096x4096 pixels
- Various formats: PNG (8/16/24/32-bit), JPEG (different qualities), GIF (animated)
- Edge cases: 1x1 pixel, extremely wide/tall images
- Corrupted files: truncated, invalid headers

#### SLIC Extraction Tests (`test_extract.py`)

```python
def test_slic_segmentation_basic():
    """Test SLIC superpixel generation with default parameters."""

def test_covariance_analysis():
    """Test mathematical correctness of covariance calculations."""

def test_splat_parameter_extraction():
    """Test Gaussian splat parameter computation from regions."""

def test_color_extraction_accuracy():
    """Test RGB color extraction from superpixel regions."""

def test_alpha_calculation():
    """Test alpha value calculation from local contrast."""

def test_splat_filtering():
    """Test quality-based splat filtering and culling."""
```

**Mathematical Validation:**
- Known geometric shapes (circles, rectangles) with expected covariance
- Solid color regions should produce minimal variance
- High-contrast edges should produce higher alpha values
- Splat count should match target ±5%

#### Depth Scoring Tests (`test_layering.py`)

```python
def test_importance_scoring():
    """Test splat importance calculation accuracy."""

def test_layer_assignment_distribution():
    """Test even distribution across depth layers."""

def test_depth_value_mapping():
    """Test correct mapping from layer index to depth values."""

def test_edge_case_scoring():
    """Test scoring with extreme values (very large/small splats)."""
```

**Scoring Validation:**
- Larger splats should generally score higher
- High-contrast regions should score higher
- Layer distribution should be reasonably balanced
- Depth values should range from 0.2 to 1.0

#### SVG Generation Tests (`test_svgout.py`)

```python
def test_svg_structure_validation():
    """Test generated SVG has correct structure and syntax."""

def test_layer_grouping():
    """Test proper layer grouping with correct data-depth attributes."""

def test_splat_rendering_solid():
    """Test solid ellipse rendering accuracy."""

def test_splat_rendering_gradient():
    """Test gradient mode rendering."""

def test_animation_script_generation():
    """Test inline JavaScript generation."""

def test_css_styles_generation():
    """Test inline CSS generation for animations."""

def test_accessibility_features():
    """Test prefers-reduced-motion implementation."""
```

**SVG Validation:**
- Well-formed XML structure
- Valid SVG elements and attributes
- Correct numerical precision (3 decimal places)
- No external dependencies
- JavaScript syntax validity

#### CLI Interface Tests (`test_cli.py`)

```python
def test_argument_parsing():
    """Test CLI argument parsing with various combinations."""

def test_parameter_validation():
    """Test validation of parameter ranges and types."""

def test_help_documentation():
    """Test help text generation and accuracy."""

def test_error_messages():
    """Test clear and helpful error messages."""

def test_verbose_output():
    """Test verbose mode logging."""
```

### 2. Integration Tests

#### End-to-End Pipeline Tests (`test_integration.py`)

```python
def test_complete_pipeline_png():
    """Test full pipeline: PNG input → SVG output."""

def test_complete_pipeline_jpg():
    """Test full pipeline: JPEG input → SVG output."""

def test_complete_pipeline_gif():
    """Test full pipeline: GIF input → SVG output."""

def test_parameter_combinations():
    """Test various CLI parameter combinations."""

def test_large_image_processing():
    """Test processing of large images (2K, 4K)."""

def test_batch_consistency():
    """Test consistent output across multiple runs."""
```

**Test Scenarios:**
- Various image sizes: 480p, 720p, 1080p, 4K
- Different splat counts: 500, 1500, 3000, 5000
- Layer variations: 3, 4, 6, 8 layers
- Parameter edge cases: minimum/maximum values

#### Performance Tests (`test_performance.py`)

```python
def test_processing_time_benchmarks():
    """Test processing time meets performance requirements."""

def test_memory_usage_limits():
    """Test memory usage stays within acceptable bounds."""

def test_output_file_size_validation():
    """Test generated file sizes meet requirements."""

def test_animation_performance():
    """Test SVG animation performance in browsers."""
```

**Performance Benchmarks:**
- 1920x1080 image processing: <30 seconds
- Memory usage: <1GB peak
- Output file size: <2MB compressed for 1500 splats
- Animation: >60fps on desktop, >45fps mobile

#### Compatibility Tests (`test_compatibility.py`)

```python
def test_browser_compatibility():
    """Test SVG rendering across different browsers."""

def test_powerpoint_compatibility():
    """Test SVG import and animation in PowerPoint."""

def test_email_client_compatibility():
    """Test SVG display in various email clients."""

def test_cross_platform_consistency():
    """Test consistent behavior across operating systems."""
```

**Compatibility Matrix:**
- Browsers: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- Applications: PowerPoint 2019+, Outlook 2019+
- Platforms: Windows 10+, macOS 10.15+, Ubuntu 20.04+

### 3. Visual Quality Tests

#### Fidelity Validation (`test_quality.py`)

```python
def test_ssim_comparison():
    """Test SSIM score against original image."""

def test_visual_appeal_metrics():
    """Test subjective visual quality metrics."""

def test_color_accuracy():
    """Test color preservation accuracy."""

def test_motion_smoothness():
    """Test animation smoothness and consistency."""
```

**Quality Metrics:**
- SSIM score: ≥0.97 (drop ≤0.03)
- Color accuracy: Delta E <2.0 for major regions
- Animation jitter: <1px deviation at 60fps
- Parallax effect visibility: Clear depth separation

#### Edge Case Validation (`test_edge_cases.py`)

```python
def test_monochrome_images():
    """Test processing of single-color or grayscale images."""

def test_high_contrast_images():
    """Test processing of images with extreme contrast."""

def test_very_small_images():
    """Test minimum viable image sizes."""

def test_panoramic_images():
    """Test extremely wide or tall aspect ratios."""

def test_transparent_backgrounds():
    """Test PNG images with transparency."""
```

### 4. Property-Based Testing

#### Fuzz Testing (`test_property_based.py`)

```python
@given(st.integers(min_value=100, max_value=10000))
def test_splat_count_property(splat_count):
    """Test output quality scales reasonably with splat count."""

@given(st.floats(min_value=1.0, max_value=5.0))
def test_k_parameter_property(k_value):
    """Test splat sizing parameter effects."""

@given(st.integers(min_value=2, max_value=8))
def test_layer_count_property(layer_count):
    """Test layer count parameter validation."""
```

### 5. Security Tests

#### Input Security (`test_security.py`)

```python
def test_malformed_image_handling():
    """Test security against malformed image files."""

def test_path_traversal_prevention():
    """Test output path validation prevents directory traversal."""

def test_resource_exhaustion_protection():
    """Test protection against resource exhaustion attacks."""

def test_svg_output_sanitization():
    """Test SVG output is safe from XSS vectors."""
```

### 6. Test Data Management

#### Test Assets

```
tests/
├── assets/
│   ├── images/
│   │   ├── test_100x100.png
│   │   ├── test_1920x1080.jpg
│   │   ├── test_animated.gif
│   │   ├── test_transparent.png
│   │   └── test_corrupted.jpg
│   ├── expected_outputs/
│   │   ├── reference_output_1500_splats.svg
│   │   └── reference_gradient_mode.svg
│   └── malformed/
│       ├── truncated.jpg
│       └── invalid_header.png
```

#### Test Configuration

```python
# conftest.py
@pytest.fixture
def sample_image():
    """Provide standard test image."""
    return load_test_image("test_1920x1080.jpg")

@pytest.fixture
def temp_output_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def cli_runner():
    """Provide Click CLI test runner."""
    return CliRunner()
```

### 7. Performance Benchmarking

#### Automated Benchmarks

```python
def benchmark_processing_time():
    """Measure processing time across image sizes."""
    sizes = [(480, 270), (1280, 720), (1920, 1080), (3840, 2160)]
    for width, height in sizes:
        image = generate_test_image(width, height)
        start_time = time.time()
        process_image(image)
        duration = time.time() - start_time
        assert duration < expected_time(width, height)

def benchmark_memory_usage():
    """Monitor peak memory usage during processing."""
    with memory_profiler.profile() as prof:
        process_large_image()
    peak_memory = max(prof.memory_usage)
    assert peak_memory < 1024  # 1GB limit
```

### 8. Continuous Integration Tests

#### CI Pipeline Tests

```yaml
# GitHub Actions test matrix
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: [3.8, 3.9, "3.10", "3.11"]
```

**CI Test Stages:**
1. **Lint and Format:** Black, mypy, flake8
2. **Unit Tests:** pytest with coverage reporting
3. **Integration Tests:** End-to-end functionality
4. **Performance Tests:** Benchmark validation
5. **Compatibility Tests:** Browser/application testing

### 9. Test Execution Strategy

#### Local Development

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=splat_this --cov-report=html

# Run performance tests
pytest tests/test_performance.py -v

# Run specific test category
pytest tests/unit/ -v
```

#### Test Categories

```python
# pytest markers
pytest.mark.unit          # Fast unit tests
pytest.mark.integration   # Slower integration tests
pytest.mark.performance   # Performance benchmarks
pytest.mark.compatibility # Cross-platform tests
pytest.mark.slow          # Long-running tests
```

### 10. Test Success Criteria

#### Coverage Requirements
- Unit test coverage: >80%
- Integration test coverage: >90% of major workflows
- Performance test coverage: All specified benchmarks
- Compatibility test coverage: All target platforms/browsers

#### Quality Gates
- All tests must pass before merge
- Performance benchmarks must meet requirements
- No security vulnerabilities in dependencies
- Code quality scores above defined thresholds

#### Regression Testing
- Automated visual regression tests for generated SVGs
- Performance regression detection (>10% degradation fails)
- Compatibility regression testing with reference browsers
- API stability testing for public interfaces