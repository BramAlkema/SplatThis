# SplatThis CLI - Quick Reference

**Spec ID:** 2025-01-20-splat-cli-core | **Status:** Draft | **Effort:** 3-4 weeks

## What We're Building

Python CLI that converts images → self-contained parallax SVG animations

```bash
splatlify photo.jpg --splats 1500 --layers 4 -o parallax.svg
```

## Key Features

🎯 **Core Function:** PNG/JPG/GIF → animated SVG with depth layers
🚀 **Performance:** <30s processing, >60fps animation, <2MB files
📱 **Universal:** Works in browsers, PowerPoint, email (no dependencies)
🎨 **Quality:** SSIM drop ≤0.03, smooth parallax with gyro support

## Technical Approach

- **Splat Extraction:** SLIC superpixels → covariance analysis → Gaussian splats
- **Depth Layers:** Score-based grouping (3-6 layers) with CSS transforms
- **Animation:** Inline CSS/JS, pointer + gyroscope parallax
- **Output:** Single SVG with all assets inline

## CLI Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--splats` | 1500 | Target splat count |
| `--layers` | 4 | Depth layers (3-6) |
| `--k` | 2.5 | Splat size multiplier |
| `--parallax-strength` | 40 | Effect intensity (px) |
| `--gaussian` | off | Gradient mode vs solid |
| `--interactive-top` | 0 | Hero splats w/ individual animation |

## Dependencies

**Required:** PIL, NumPy, scikit-image, Click
**Target:** Python 3.8+, cross-platform

## Success Criteria

✅ **MVP:** Working CLI with default parameters
✅ **Performance:** <30s processing, >60fps playback
✅ **Compatibility:** Chrome/Firefox/Safari + PowerPoint
✅ **Quality:** Visually appealing parallax effects

## Architecture

```
src/
├── cli.py      # Argument parsing & main flow
├── extract.py  # SLIC → Gaussian splats
├── layering.py # Depth scoring & layer assignment
└── svgout.py   # SVG generation with inline assets
```

## Delivery Timeline

- **Week 1:** Technical planning & architecture
- **Week 2:** Core pipeline implementation
- **Week 3:** SVG generation & animation
- **Week 4:** Testing, optimization & polish

---
**Full Spec:** @spec.md | **Tasks:** @tasks.md | **Technical Details:** @sub-specs/