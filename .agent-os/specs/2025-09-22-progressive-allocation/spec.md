# Progressive Gaussian Allocation Implementation Spec

**Created:** 2025-09-22
**Version:** 1.0
**Status:** Ready for Implementation
**Priority:** High
**Related:** @docs/adr-001-image-gs-insights.md

## Executive Summary

Implement progressive Gaussian allocation inspired by Image-GS methodology to replace our current static allocation approach. This addresses the fundamental inefficiency of allocating all splats upfront and enables content-adaptive splat density based on reconstruction error.

## Problem Statement

### Current Limitations
1. **Static Allocation** - All 1800+ Gaussians allocated at initialization, regardless of content complexity
2. **Uniform Density** - Same splat density in smooth backgrounds and detailed edges
3. **Wasted Resources** - Many splats contribute minimally to final reconstruction
4. **No Adaptivity** - Cannot respond to reconstruction quality during optimization
5. **Fixed Budget** - Hard limit prevents adding splats where most needed

### Root Cause Analysis
The current `AdaptiveSplatExtractor.extract_adaptive_splats()` method uses a fixed target (`n_splats`) and allocates all splats during initialization. This prevents the system from learning where splats are most valuable and leads to suboptimal resource allocation.

## Solution Overview

Implement **progressive Gaussian allocation** with:

1. **Initial Allocation** - Start with 30% of target splats in high-gradient regions
2. **Error Mapping** - Continuous reconstruction error monitoring
3. **Adaptive Addition** - Add splats incrementally where error is highest
4. **Resource Budget** - Intelligent splat count management
5. **Convergence Detection** - Stop adding when error stabilizes

## Technical Architecture

### Core Components

#### 1. ProgressiveAllocator Class
```python
@dataclass
class ProgressiveConfig:
    initial_ratio: float = 0.3              # Start with 30% of target splats
    max_splats: int = 2000                  # Maximum total splats allowed
    add_interval: int = 50                  # Add splats every N iterations
    max_add_per_step: int = 20              # Limit splats added per step
    error_threshold: float = 0.01           # Minimum error for new placement
    convergence_patience: int = 5           # Steps without improvement to stop

class ProgressiveAllocator:
    def __init__(self, config: ProgressiveConfig):
        self.config = config
        self.iteration_count = 0
        self.error_history = []

    def should_add_splats(self, current_error: float) -> bool:
        """Determine if new splats should be added."""

    def get_addition_count(self, current_splat_count: int) -> int:
        """Calculate how many splats to add this step."""

    def select_placement_positions(self, error_map: np.ndarray, count: int) -> List[Tuple[int, int]]:
        """Select positions for new splats based on error distribution."""
```

#### 2. ErrorGuidedPlacement Class
```python
class ErrorGuidedPlacement:
    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def compute_reconstruction_error(self, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Compute per-pixel reconstruction error."""

    def create_placement_probability(self, error_map: np.ndarray) -> np.ndarray:
        """Convert error map to placement probability distribution."""

    def sample_positions(self, prob_map: np.ndarray, count: int) -> List[Tuple[int, int]]:
        """Sample new splat positions from probability distribution."""
```

#### 3. Enhanced AdaptiveSplatExtractor
```python
class AdaptiveSplatExtractor:
    def __init__(self, config: Optional[AdaptiveSplatConfig] = None,
                 progressive_config: Optional[ProgressiveConfig] = None):
        self.config = config or AdaptiveSplatConfig()
        self.progressive_config = progressive_config or ProgressiveConfig()
        self.allocator = ProgressiveAllocator(self.progressive_config)
        self.placer = ErrorGuidedPlacement()

    def extract_progressive_splats(self, image: np.ndarray, verbose: bool = False) -> List[Gaussian]:
        """Progressive splat extraction with error-guided allocation."""
```

### Algorithm Flow

#### Phase 1: Initial Allocation (30% of budget)
```python
def initial_allocation(self, image: np.ndarray) -> List[Gaussian]:
    # Compute gradient-based saliency
    saliency_map = self.saliency_analyzer.compute_saliency_map(image)

    # Calculate initial splat count
    initial_count = int(self.progressive_config.max_splats * self.progressive_config.initial_ratio)

    # High-priority placement in gradient regions
    initial_positions = self._sample_from_saliency(saliency_map, initial_count)

    # Create initial splats with adaptive sizing
    initial_splats = []
    for pos in initial_positions:
        splat = self._create_adaptive_splat_at_position(image, pos, saliency_map)
        initial_splats.append(splat)

    return initial_splats
```

#### Phase 2: Progressive Addition Loop
```python
def progressive_addition_loop(self, image: np.ndarray, splats: List[Gaussian]) -> List[Gaussian]:
    for iteration in range(self.progressive_config.max_iterations):
        # Render current splats
        rendered = self._render_splats(splats, image.shape[:2])

        # Compute reconstruction error
        error_map = self.placer.compute_reconstruction_error(image, rendered)
        mean_error = np.mean(error_map)

        # Check if we should add more splats
        if not self.allocator.should_add_splats(mean_error):
            break

        if iteration % self.progressive_config.add_interval == 0:
            # Determine how many splats to add
            add_count = self.allocator.get_addition_count(len(splats))

            if add_count > 0 and len(splats) < self.progressive_config.max_splats:
                # Sample new positions from error map
                new_positions = self.allocator.select_placement_positions(error_map, add_count)

                # Create new splats at high-error positions
                new_splats = []
                for pos in new_positions:
                    splat = self._create_adaptive_splat_at_position(image, pos, error_map)
                    new_splats.append(splat)

                splats.extend(new_splats)

                if verbose:
                    print(f"Iteration {iteration}: Added {len(new_splats)} splats, total: {len(splats)}, error: {mean_error:.4f}")

        # Store error for convergence detection
        self.allocator.error_history.append(mean_error)

        # Optional: Perform optimization step on existing splats
        # (This would be part of Phase 2 from the ADR)

    return splats
```

#### Phase 3: Error-Guided Position Selection
```python
def select_placement_positions(self, error_map: np.ndarray, count: int) -> List[Tuple[int, int]]:
    # Convert error to probability distribution
    error_normalized = error_map / (np.sum(error_map) + 1e-8)

    # Apply temperature for sharper/softer sampling
    prob_map = error_normalized ** (1.0 / self.temperature)
    prob_map = prob_map / np.sum(prob_map)

    # Flatten for sampling
    height, width = error_map.shape
    flat_probs = prob_map.flatten()

    # Sample positions without replacement
    sampled_indices = np.random.choice(
        len(flat_probs),
        size=min(count, len(flat_probs)),
        replace=False,
        p=flat_probs
    )

    # Convert back to 2D coordinates
    positions = []
    for idx in sampled_indices:
        y, x = divmod(idx, width)
        positions.append((y, x))

    return positions
```

### Integration with Existing System

#### Modified Configuration
```python
@dataclass
class AdaptiveSplatConfig:
    # Existing parameters...
    min_scale: float = 2.0
    max_scale: float = 20.0

    # New progressive parameters
    enable_progressive: bool = True
    progressive_config: Optional[ProgressiveConfig] = None

    def __post_init__(self):
        if self.enable_progressive and self.progressive_config is None:
            self.progressive_config = ProgressiveConfig()
```

#### Enhanced CLI Interface
```bash
# Enable progressive allocation
python -m splat_this.cli --input image.png --progressive --max-splats 2000

# Configure progressive parameters
python -m splat_this.cli --input image.png --progressive \
    --initial-ratio 0.3 \
    --add-interval 50 \
    --error-threshold 0.01
```

## Implementation Plan

### Phase 1: Core Infrastructure (Days 1-2)
- **T1.1:** Implement `ProgressiveConfig` dataclass with validation
- **T1.2:** Create `ProgressiveAllocator` class with error tracking
- **T1.3:** Implement `ErrorGuidedPlacement` class for position sampling
- **T1.4:** Add configuration integration to existing `AdaptiveSplatConfig`

### Phase 2: Error Computation (Days 3-4)
- **T2.1:** Implement reconstruction error computation (L1/L2 distance)
- **T2.2:** Create error map visualization utilities for debugging
- **T2.3:** Add error-to-probability conversion with temperature control
- **T2.4:** Implement position sampling from probability distributions

### Phase 3: Allocation Logic (Days 5-6)
- **T3.1:** Implement initial allocation with saliency-based sampling
- **T3.2:** Create progressive addition loop with interval control
- **T3.3:** Add convergence detection based on error history
- **T3.4:** Implement resource budget management and splat count limits

### Phase 4: Integration (Days 7-8)
- **T4.1:** Integrate progressive allocation into `AdaptiveSplatExtractor`
- **T4.2:** Add CLI parameters for progressive configuration
- **T4.3:** Create backward compatibility for non-progressive mode
- **T4.4:** Add verbose logging and progress reporting

### Phase 5: Testing & Validation (Days 9-10)
- **T5.1:** Unit tests for all new classes and methods
- **T5.2:** Integration tests with existing pipeline
- **T5.3:** Performance benchmarking vs. static allocation
- **T5.4:** Visual quality validation with test images

## Success Criteria

### Quality Metrics
1. **Adaptive Density:** Splat distribution follows content complexity
2. **Error Reduction:** Progressive addition reduces reconstruction error
3. **Efficiency:** Better quality-to-splat-count ratio than static allocation
4. **Convergence:** Stable error reduction with clear stopping criteria

### Performance Targets
- **Splat Efficiency:** 20-30% fewer splats for equivalent quality
- **Error Reduction:** 15% lower final reconstruction error
- **Processing Time:** ≤ 1.5x current extraction time
- **Memory Usage:** ≤ 1.2x current peak memory

### Technical Validation
- **Error Computation:** Accurate per-pixel reconstruction error
- **Sampling Quality:** Proper probability distribution sampling
- **Resource Management:** Respects splat count budgets
- **Integration:** Seamless backward compatibility

## Risk Mitigation

### Technical Risks
1. **Convergence Issues:** Implement robust error tracking and patience mechanism
2. **Sampling Bias:** Use proper normalization and avoid zero-probability regions
3. **Performance Impact:** Profile error computation and optimize critical paths
4. **Memory Usage:** Monitor memory growth during progressive addition

### Integration Risks
1. **API Compatibility:** Maintain existing `extract_adaptive_splats` interface
2. **Configuration Complexity:** Provide sensible defaults and validation
3. **Testing Coverage:** Comprehensive tests for edge cases and error conditions
4. **SVG Output:** Ensure progressive splats render correctly in SVG format

## Dependencies

### External Libraries
- **NumPy:** Core array operations and random sampling
- **SciPy:** Advanced optimization and signal processing (if needed)
- **Existing Codebase:** Current `AdaptiveSplatExtractor` and `SaliencyAnalyzer`

### Internal Components
- **SaliencyAnalyzer:** For initial gradient-based placement
- **Gaussian class:** For splat representation and properties
- **SVG generation:** For progressive output compatibility
- **CLI framework:** For new parameter integration

## Testing Strategy

### Unit Tests
```python
class TestProgressiveAllocator:
    def test_initial_allocation_ratio(self):
        """Test that initial allocation respects ratio configuration."""

    def test_error_guided_placement(self):
        """Test that high-error regions get more splats."""

    def test_budget_management(self):
        """Test that splat count never exceeds maximum."""

    def test_convergence_detection(self):
        """Test that allocation stops when error stabilizes."""

class TestErrorGuidedPlacement:
    def test_error_computation_accuracy(self):
        """Test reconstruction error computation against known values."""

    def test_probability_distribution(self):
        """Test error-to-probability conversion and normalization."""

    def test_position_sampling(self):
        """Test that sampling follows probability distribution."""
```

### Integration Tests
```python
class TestProgressiveIntegration:
    def test_backward_compatibility(self):
        """Test that non-progressive mode still works."""

    def test_cli_integration(self):
        """Test CLI parameter handling for progressive mode."""

    def test_svg_output_compatibility(self):
        """Test that progressive splats render correctly in SVG."""

    def test_performance_benchmarks(self):
        """Test performance vs. static allocation baseline."""
```

## Future Enhancements

### Short-term (Next Sprint)
- **Optimization Integration:** Combine with existing refinement stages
- **Multi-scale Error:** Compute error at multiple resolutions
- **Smart Initialization:** Better initial placement strategies
- **Real-time Visualization:** Live progress tracking during allocation

### Medium-term (Next Month)
- **Perceptual Error:** Use SSIM or LPIPS instead of L1/L2
- **Semantic Guidance:** Object-aware placement priorities
- **Hierarchical Allocation:** Multi-resolution progressive addition
- **GPU Acceleration:** CUDA-based error computation for large images

## Conclusion

Progressive Gaussian allocation represents a fundamental improvement in resource efficiency and content adaptivity. By starting with a sparse set of splats and progressively adding them where reconstruction error is highest, we achieve better quality with fewer resources while maintaining compatibility with our existing SVG pipeline.

The key benefits are:
1. **Adaptive Density:** More splats where they matter most
2. **Resource Efficiency:** Fewer total splats for equivalent quality
3. **Content Awareness:** Automatic adaptation to image complexity
4. **Quality Control:** Continuous monitoring and improvement

This implementation provides the foundation for subsequent phases in our image-gs integration roadmap.