# Progressive Allocation - Quick Reference

**Status:** Ready for Implementation
**Priority:** High
**Timeline:** 10 days

## What We're Building
Replace static splat allocation with progressive, error-guided splat placement inspired by Image-GS.

## Key Components
- **ProgressiveAllocator:** Manages splat budget and addition timing
- **ErrorGuidedPlacement:** Samples new positions from reconstruction error
- **Enhanced AdaptiveSplatExtractor:** Integrates progressive allocation

## Algorithm
1. **Start Small:** 30% of target splats in high-gradient regions
2. **Monitor Error:** Continuous reconstruction error computation
3. **Add Strategically:** Place new splats where error is highest
4. **Converge:** Stop when error stabilizes

## Success Metrics
- 20-30% fewer splats for equivalent quality
- 15% lower reconstruction error
- ≤1.5x processing time vs. current

## Implementation Phases
1. Core infrastructure (2 days)
2. Error computation (2 days)
3. Allocation logic (2 days)
4. Integration (2 days)
5. Testing & validation (2 days)

## CLI Usage
```bash
# Enable progressive allocation
python -m splat_this.cli --input image.png --progressive --max-splats 2000
```