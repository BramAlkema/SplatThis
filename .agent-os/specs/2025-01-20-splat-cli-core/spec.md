# SplatThis CLI Core Specification

**Metadata:**
- **Spec ID:** 2025-01-20-splat-cli-core
- **Created:** 2025-01-20
- **Version:** 1.0.0
- **Status:** Draft
- **Owner:** Development Team
- **Priority:** High
- **Estimated Effort:** 3-4 weeks

**Cross-References:**
- @.agent-os/product/mission.md
- @.agent-os/decisions.md (ADR-001 through ADR-005)
- @.agent-os/product/tech-stack.md

## Overview

SplatThis is a Python CLI tool that converts static images (PNG, JPG, GIF) into self-contained SVG files with parallax animation effects. The tool uses Gaussian splat representation to create depth-layered animations that work universally across browsers, email clients, and presentation software without external dependencies.

## Business Context

### Problem Statement
- Static images lack engagement in presentations and web content
- Existing parallax solutions require complex setups with external dependencies
- Current tools either produce large files or require specific runtime environments
- No single-command solution exists for creating self-contained parallax animations

### Success Metrics
- Generate SVG files under 2MB compressed for 1500-3000 splats
- Achieve >60fps performance on 2019+ hardware
- Maintain visual fidelity with SSIM drop ≤0.03 vs original
- Process typical images (1920x1080) in under 30 seconds

## Functional Requirements

### Core Features

#### F1: Image Input Processing
- **F1.1:** Support PNG, JPG, and GIF formats
- **F1.2:** Optional GIF frame selection with `--frame n` parameter
- **F1.3:** Automatic image format detection and validation
- **F1.4:** Handle common image sizes from 480p to 4K
- **F1.5:** Graceful error handling for corrupted or unsupported files

#### F2: Splat Extraction
- **F2.1:** SLIC superpixel segmentation for region detection
- **F2.2:** Covariance analysis for splat parameter extraction (x, y, rx, ry, θ)
- **F2.3:** Color extraction with RGB and alpha calculations
- **F2.4:** Configurable splat count with `--splats n` parameter (default: 1500)
- **F2.5:** Quality-based splat filtering (remove smallest 2% by default)
- **F2.6:** Splat size scaling with configurable `--k` parameter (default: 2.5)

#### F3: Depth Layer Assignment
- **F3.1:** Score-based depth calculation using area × edge_strength
- **F3.2:** Quantized layer assignment (3-6 layers, configurable via `--layers n`)
- **F3.3:** Depth mapping to data-depth values (0.2 → 1.0, back → front)
- **F3.4:** Optional interactive splat limiting with `--interactive-top n`

#### F4: SVG Generation
- **F4.1:** Single self-contained SVG output with proper viewBox
- **F4.2:** Layer grouping with `<g class="layer" data-depth="...">` structure
- **F4.3:** Solid ellipse rendering (default) or gradient mode (`--gaussian`)
- **F4.4:** Inline CSS for animation styles and responsive behavior
- **F4.5:** Inline JavaScript for parallax interaction
- **F4.6:** Accessibility support with `prefers-reduced-motion` detection

#### F5: Animation System
- **F5.1:** Pointer-based parallax with configurable strength (`--parallax-strength`)
- **F5.2:** Gyroscope support for mobile devices
- **F5.3:** Smooth easing transitions (default: 0.12 easing factor)
- **F5.4:** Layer-based transform optimization for performance
- **F5.5:** Optional per-splat interaction for hero elements

#### F6: CLI Interface
- **F6.1:** Intuitive command structure: `splatlify input.png -o output.svg`
- **F6.2:** Comprehensive parameter configuration
- **F6.3:** Progress indicators for long-running operations
- **F6.4:** Verbose mode for debugging and optimization
- **F6.5:** Help documentation with examples

### CLI Parameters Specification

```bash
splatlify input.(png|jpg|gif) \
  [--frame 0] \
  [--splats 1500] \
  [--layers 4] \
  [--k 2.5] \
  [--alpha 0.65] \
  [--parallax-strength 40] \
  [--interactive-top 0] \
  [--gaussian] \
  [--verbose] \
  -o output.svg
```

**Parameter Details:**
- `input`: Source image file (required)
- `--frame`: GIF frame number (default: 0)
- `--splats`: Target splat count (default: 1500)
- `--layers`: Number of depth layers (default: 4)
- `--k`: Splat size multiplier (default: 2.5)
- `--alpha`: Base alpha transparency (default: 0.65)
- `--parallax-strength`: Parallax effect strength in pixels (default: 40)
- `--interactive-top`: Number of hero splats with individual animation (default: 0)
- `--gaussian`: Enable gradient rendering mode
- `--verbose`: Enable detailed logging
- `-o, --output`: Output SVG file path (required)

## Non-Functional Requirements

### Performance Requirements
- **P1:** Process 1920x1080 images in under 30 seconds on 2019+ hardware
- **P2:** Generate SVG files under 2MB compressed for 1500 splats
- **P3:** SVG animations maintain >60fps on desktop browsers
- **P4:** Mobile performance >45fps on modern smartphones
- **P5:** Memory usage under 1GB during processing

### Compatibility Requirements
- **C1:** Python 3.8+ compatibility
- **C2:** Cross-platform support (Windows, macOS, Linux)
- **C3:** SVG compatibility with modern browsers (Chrome 80+, Firefox 75+, Safari 13+)
- **C4:** PowerPoint and email client compatibility
- **C5:** No external runtime dependencies for SVG playback

### Quality Requirements
- **Q1:** Visual fidelity with SSIM drop ≤0.03 vs original image
- **Q2:** Smooth animation without jitter or performance degradation
- **Q3:** Graceful degradation when motion is disabled
- **Q4:** Consistent output across different hardware configurations

## Technical Architecture

### Module Structure
```
src/
├── cli.py              # Command-line interface and argument parsing
├── extract.py          # SLIC superpixel extraction and splat generation
├── layering.py         # Depth scoring and layer assignment
├── svgout.py          # SVG generation with inline CSS/JS
└── utils.py           # Shared utilities and data structures
```

### Core Data Structures
```python
@dataclass
class Gaussian:
    x: float          # X coordinate
    y: float          # Y coordinate
    rx: float         # X-axis radius
    ry: float         # Y-axis radius
    theta: float      # Rotation angle
    r: int           # Red component (0-255)
    g: int           # Green component (0-255)
    b: int           # Blue component (0-255)
    a: float         # Alpha component (0.0-1.0)
    score: float     # Quality/importance score
    depth: float     # Depth layer assignment (0.0-1.0)
```

### Processing Pipeline
1. **Image Loading:** PIL-based image loading with format detection
2. **SLIC Segmentation:** scikit-image SLIC superpixel generation
3. **Splat Extraction:** Covariance analysis for each superpixel region
4. **Quality Scoring:** Area and edge strength analysis
5. **Depth Assignment:** Score-based quantized layer assignment
6. **SVG Generation:** Template-based SVG output with inline assets

## Dependencies

### Required Dependencies
- **PIL (Pillow):** Image loading and basic processing
- **NumPy:** Numerical computations and array operations
- **scikit-image:** SLIC superpixel segmentation
- **Click:** CLI framework for argument parsing

### Optional Dependencies
- **pytest:** Unit testing framework
- **black:** Code formatting
- **mypy:** Type checking

## User Stories

### US1: Basic Image Conversion
**As a** content creator
**I want to** convert a static background image into a parallax SVG
**So that** I can add engaging motion to my presentations

**Acceptance Criteria:**
- Single command execution: `splatlify bg.jpg -o parallax.svg`
- Output SVG works in PowerPoint without additional setup
- Animation is smooth and visually appealing
- File size is manageable for email/web distribution

### US2: Advanced Customization
**As a** web developer
**I want to** fine-tune parallax parameters for specific visual effects
**So that** I can match my design requirements precisely

**Acceptance Criteria:**
- All CLI parameters function as documented
- Visual differences are clear when adjusting parameters
- Performance remains acceptable with custom settings
- Output quality meets professional standards

### US3: Mobile Compatibility
**As a** mobile user
**I want to** experience parallax effects on my smartphone
**So that** I can enjoy rich content across all devices

**Acceptance Criteria:**
- Touch and gyroscope input work correctly
- Performance is smooth on modern mobile browsers
- Battery drain is minimal during viewing
- Motion can be disabled via system preferences

## Edge Cases and Error Handling

### Input Validation
- Unsupported file formats → Clear error message with supported formats
- Corrupted image files → Graceful failure with diagnostic information
- Extremely large images → Memory management with optional downscaling
- Empty or single-color images → Minimum viable splat generation

### Parameter Validation
- Invalid splat counts → Clamp to reasonable ranges (100-10000)
- Invalid layer counts → Clamp to supported range (2-8)
- Invalid file paths → Clear error messages with suggestions
- Conflicting parameters → Priority rules with warnings

### Output Generation
- Filesystem permissions → Clear error messages
- Disk space limitations → Early detection and user notification
- SVG validation → Optional lint mode for output verification

## Testing Strategy

### Unit Testing
- Image loading and validation functions
- SLIC segmentation parameter handling
- Splat extraction mathematical operations
- SVG generation template rendering
- CLI argument parsing and validation

### Integration Testing
- End-to-end pipeline with sample images
- Performance benchmarking with various image sizes
- Cross-platform compatibility validation
- SVG output compatibility across browsers/applications

### Performance Testing
- Memory usage profiling during processing
- Processing time benchmarks across hardware
- SVG rendering performance in target environments
- File size optimization validation

## Security Considerations

### Input Security
- Image file format validation to prevent malformed file attacks
- Memory limits to prevent resource exhaustion
- Path traversal prevention for output file specification
- Safe temporary file handling during processing

### Output Security
- SVG sanitization to prevent XSS in web contexts
- No external resource references in generated SVG
- Safe JavaScript generation without eval or dynamic execution

## Future Enhancements

### Phase 2: Advanced Features
- **Checkpoint Integration:** Support for pre-trained Gaussian Splatting models
- **Hybrid Rendering:** `--bake-tail` option for raster/vector optimization
- **Animation Presets:** Pre-configured parameter sets for common use cases
- **Batch Processing:** Multi-image processing with consistent settings

### Phase 3: Quality Improvements
- **Advanced Scoring:** Machine learning-based importance detection
- **Color Optimization:** Palette reduction for smaller file sizes
- **Motion Blur:** Realistic motion effects for dynamic content
- **3D Effects:** Pseudo-3D transforms for enhanced depth

## Acceptance Criteria

### Minimum Viable Product (MVP)
- [ ] CLI accepts PNG/JPG input and generates working SVG output
- [ ] Default parameters produce visually appealing results
- [ ] Generated SVG works in Chrome, Firefox, Safari, and PowerPoint
- [ ] Processing completes in under 60 seconds for 1920x1080 images
- [ ] Output files are under 3MB for 1500 splats

### Production Ready
- [ ] All CLI parameters function as documented
- [ ] Comprehensive error handling and user feedback
- [ ] Performance meets specified benchmarks
- [ ] Cross-platform compatibility verified
- [ ] Documentation and examples complete

### Quality Assurance
- [ ] Unit test coverage >80%
- [ ] Integration tests for all major workflows
- [ ] Performance regression testing
- [ ] Accessibility compliance (motion preferences)
- [ ] Security review for input handling

---

**Review Schedule:**
- Technical review: Week 1
- Implementation kickoff: Week 2
- MVP target: Week 3
- Production ready: Week 4

**Stakeholder Sign-off:**
- [ ] Product Owner
- [ ] Technical Lead
- [ ] UX/Design Review
- [ ] Security Review