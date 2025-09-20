# Product Decision Log

**Priority: Override** - Decisions here override all other project documents

Last updated: 2025-01-20

## ADR-001: SVG-Only Output with Inline Animation

**Date:** 2025-01-20
**Status:** Accepted

### Context
SplatThis needs to generate self-contained parallax animations from images. We evaluated several output formats:
- Canvas/WebGL with external JS libraries
- SVG with external CSS/JS dependencies
- Self-contained SVG with inline styles and scripts

### Decision
We will output single, self-contained SVG files with inline CSS and JavaScript for parallax animation.

### Rationale
- **Zero Dependencies:** Works in PowerPoint, email, static hosting without external files
- **Universal Compatibility:** SVG supported across all modern browsers and applications
- **Performance:** CSS transforms more efficient than WebGL for simple parallax effects
- **File Size:** Inline approach keeps everything under 1-3MB compressed
- **Accessibility:** Respects `prefers-reduced-motion` natively

### Consequences
- Positive: Universal compatibility, zero setup, works offline
- Negative: Limited to CSS transform capabilities, larger file size than external dependencies
- Mitigation: Implement `--bake-tail` option for hybrid raster/vector approach

---

## ADR-002: SLIC Superpixel Heuristic Over Gaussian Splatting Training

**Date:** 2025-01-20
**Status:** Accepted

### Context
For splat extraction, we considered:
1. Full Gaussian Splatting training (Image-GS, gsplat)
2. SLIC superpixel heuristic with covariance analysis
3. Edge-based feature detection

### Decision
We will use SLIC superpixel segmentation with covariance-based splat parameter extraction as the primary method.

### Rationale
- **Speed:** Heuristic approach processes images in seconds vs minutes for training
- **Simplicity:** No GPU dependencies, works on any machine
- **Quality:** SLIC preserves perceptual boundaries well for background/parallax use cases
- **Deterministic:** Consistent output without training variance
- **Future-Proof:** Can add `--checkpoint` flag for Image-GS integration later

### Consequences
- Positive: Fast processing, no GPU requirements, predictable output
- Negative: Lower fidelity than trained Gaussian splats
- Mitigation: Implement optional `--checkpoint` support for trained models

---

## ADR-003: Depth-Based Layer Grouping with Transform Parallax

**Date:** 2025-01-20
**Status:** Accepted

### Context
For parallax animation, we evaluated:
1. Per-splat individual animation (high performance cost)
2. Layer-based group transforms (efficient)
3. Hybrid approach with hero splats + layer background

### Decision
We will group splats into 3-6 depth layers and animate entire layers with CSS transforms.

### Rationale
- **Performance:** Layer transforms maintain 60fps on 2019 hardware
- **Visual Quality:** Sufficient parallax depth for backdrop effects
- **Battery Life:** Minimal GPU usage compared to per-element animation
- **Scalability:** Works with thousands of splats without performance degradation

### Consequences
- Positive: Excellent performance, works on mobile/desktop
- Negative: Less granular animation than per-splat approach
- Mitigation: Optional `--interactive-top N` flag for hero splat individual animation

---

## ADR-004: Radial Gradient Strategy for File Size Optimization

**Date:** 2025-01-20
**Status:** Accepted

### Context
For splat rendering, we compared:
1. Unique radial gradient per splat (highest quality)
2. Shared gradient mask with colored rectangles (smallest files)
3. Solid fill ellipses (fastest generation)

### Decision
We will default to solid fill ellipses with optional `--gaussian` flag for shared gradient approach.

### Rationale
- **File Size:** Solid ellipses produce smallest files (~1-2MB vs 2-3MB)
- **Generation Speed:** No gradient computation overhead
- **PowerPoint Compatibility:** Avoids potential gradient rendering issues
- **Performance:** Faster SVG parsing and rendering

### Consequences
- Positive: Optimal file size and compatibility
- Negative: Lower visual fidelity compared to gradient approach
- Mitigation: `--gaussian` flag enables gradient mode when quality is prioritized

---

## ADR-005: Python CLI with Minimal Dependencies

**Date:** 2025-01-20
**Status:** Accepted

### Context
Technology stack options:
1. Python with NumPy/PIL/scikit-image
2. Rust for performance
3. JavaScript/Node.js for web integration

### Decision
We will implement in Python with minimal, well-established dependencies.

### Rationale
- **Ecosystem:** Rich image processing libraries (PIL, scikit-image, NumPy)
- **SLIC Implementation:** scikit-image provides robust SLIC superpixel segmentation
- **Accessibility:** Python more accessible to contributors than Rust
- **Development Speed:** Rapid prototyping and iteration
- **Dependencies:** Stick to packages commonly available in scientific Python distributions

### Consequences
- Positive: Fast development, rich ecosystem, easy contributions
- Negative: Potentially slower than Rust implementation
- Mitigation: Focus on algorithmic efficiency, consider Rust rewrite if performance becomes critical

## Override Decisions

None yet.

---

**Note:** Decisions marked with 🔒 are locked and require team consensus to modify.