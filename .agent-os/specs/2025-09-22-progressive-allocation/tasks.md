# Progressive Allocation - Implementation Tasks

**Spec:** @spec.md
**Technical Details:** @sub-specs/technical-spec.md
**Status:** Ready for Implementation

## Task Breakdown

### Phase 1: Core Infrastructure (Days 1-2)

#### T1.1: Implement ProgressiveConfig
- **Description:** Create configuration dataclass with validation
- **Files:** `src/splat_this/core/progressive_allocator.py`
- **Priority:** High
- **Estimated:** 4 hours
- **Dependencies:** None
- **Success Criteria:**
  - [ ] ProgressiveConfig dataclass with all parameters
  - [ ] Parameter validation in `__post_init__`
  - [ ] Unit tests for configuration validation
  - [ ] Documentation with usage examples

#### T1.2: Implement ProgressiveAllocator
- **Description:** Core allocation logic and error tracking
- **Files:** `src/splat_this/core/progressive_allocator.py`
- **Priority:** High
- **Estimated:** 6 hours
- **Dependencies:** T1.1
- **Success Criteria:**
  - [ ] ProgressiveAllocator class with all methods
  - [ ] Error history tracking and convergence detection
  - [ ] Addition timing and count calculation
  - [ ] Unit tests for allocation logic

#### T1.3: Implement ErrorGuidedPlacement
- **Description:** Error computation and probability sampling
- **Files:** `src/splat_this/core/error_guided_placement.py`
- **Priority:** High
- **Estimated:** 6 hours
- **Dependencies:** T1.1
- **Success Criteria:**
  - [ ] ErrorGuidedPlacement class with all methods
  - [ ] Reconstruction error computation (L1/L2)
  - [ ] Error-to-probability conversion with temperature
  - [ ] Position sampling from probability distributions
  - [ ] Unit tests for error computation and sampling

#### T1.4: Configuration Integration
- **Description:** Add progressive config to existing AdaptiveSplatConfig
- **Files:** `src/splat_this/core/extract.py`
- **Priority:** Medium
- **Estimated:** 2 hours
- **Dependencies:** T1.1
- **Success Criteria:**
  - [ ] Add `enable_progressive` flag to AdaptiveSplatConfig
  - [ ] Add `progressive_config` field with default
  - [ ] Backward compatibility for existing configurations
  - [ ] Configuration validation tests

### Phase 2: Error Computation (Days 3-4)

#### T2.1: Reconstruction Error Implementation
- **Description:** Implement accurate per-pixel error computation
- **Files:** `src/splat_this/utils/reconstruction_error.py`
- **Priority:** High
- **Estimated:** 4 hours
- **Dependencies:** T1.3
- **Success Criteria:**
  - [ ] L1 and L2 error computation functions
  - [ ] Support for RGB and grayscale images
  - [ ] Proper normalization and data type handling
  - [ ] Unit tests with known error values

#### T2.2: Error Visualization Utilities
- **Description:** Debug utilities for visualizing error maps
- **Files:** `src/splat_this/utils/visualization.py`
- **Priority:** Low
- **Estimated:** 3 hours
- **Dependencies:** T2.1
- **Success Criteria:**
  - [ ] Error map visualization functions
  - [ ] Probability map visualization
  - [ ] Side-by-side comparison utilities
  - [ ] Export to PNG for debugging

#### T2.3: Probability Distribution Implementation
- **Description:** Temperature-controlled error-to-probability conversion
- **Files:** `src/splat_this/core/error_guided_placement.py` (enhance)
- **Priority:** High
- **Estimated:** 4 hours
- **Dependencies:** T2.1
- **Success Criteria:**
  - [ ] Temperature parameter implementation
  - [ ] Proper probability normalization
  - [ ] Handle edge cases (zero error, uniform error)
  - [ ] Unit tests for distribution properties

#### T2.4: Position Sampling Implementation
- **Description:** Efficient sampling from 2D probability distributions
- **Files:** `src/splat_this/utils/sampling.py`
- **Priority:** High
- **Estimated:** 5 hours
- **Dependencies:** T2.3
- **Success Criteria:**
  - [ ] Efficient flat array sampling with np.random.choice
  - [ ] Coordinate conversion utilities
  - [ ] Support for sampling without replacement
  - [ ] Performance tests for large images

### Phase 3: Allocation Logic (Days 5-6)

#### T3.1: Initial Allocation Implementation
- **Description:** Saliency-based initial splat placement
- **Files:** `src/splat_this/core/adaptive_extract.py` (enhance)
- **Priority:** High
- **Estimated:** 6 hours
- **Dependencies:** T2.4, existing saliency analyzer
- **Success Criteria:**
  - [ ] Initial allocation method with configurable ratio
  - [ ] Integration with existing saliency analyzer
  - [ ] Position sampling from saliency distribution
  - [ ] Splat creation at sampled positions

#### T3.2: Progressive Addition Loop
- **Description:** Main iteration loop with error-guided addition
- **Files:** `src/splat_this/core/adaptive_extract.py` (enhance)
- **Priority:** High
- **Estimated:** 8 hours
- **Dependencies:** T3.1, T1.2, T1.3
- **Success Criteria:**
  - [ ] Main iteration loop with configurable parameters
  - [ ] Error computation and tracking integration
  - [ ] Conditional splat addition based on error
  - [ ] Progress logging and verbose output

#### T3.3: Convergence Detection
- **Description:** Smart stopping criteria based on error history
- **Files:** `src/splat_this/core/progressive_allocator.py` (enhance)
- **Priority:** Medium
- **Estimated:** 4 hours
- **Dependencies:** T3.2
- **Success Criteria:**
  - [ ] Error history analysis for convergence
  - [ ] Configurable patience parameter
  - [ ] Early stopping when error stabilizes
  - [ ] Unit tests for convergence detection

#### T3.4: Resource Budget Management
- **Description:** Splat count limits and addition control
- **Files:** `src/splat_this/core/progressive_allocator.py` (enhance)
- **Priority:** High
- **Estimated:** 3 hours
- **Dependencies:** T3.2
- **Success Criteria:**
  - [ ] Respect maximum splat count limits
  - [ ] Control addition rate and batch size
  - [ ] Handle edge cases (budget exceeded, no positions)
  - [ ] Resource usage tracking and reporting

### Phase 4: Integration (Days 7-8)

#### T4.1: AdaptiveSplatExtractor Integration
- **Description:** Integrate progressive allocation into main extractor
- **Files:** `src/splat_this/core/adaptive_extract.py` (major enhancement)
- **Priority:** High
- **Estimated:** 8 hours
- **Dependencies:** T3.4, all previous tasks
- **Success Criteria:**
  - [ ] Progressive extraction method implementation
  - [ ] Integration with existing extract_adaptive_splats
  - [ ] Backward compatibility with static allocation
  - [ ] Mode selection based on configuration

#### T4.2: CLI Parameter Integration
- **Description:** Add command-line interface for progressive parameters
- **Files:** `src/splat_this/cli/main.py`
- **Priority:** Medium
- **Estimated:** 4 hours
- **Dependencies:** T4.1
- **Success Criteria:**
  - [ ] --progressive flag for enabling progressive mode
  - [ ] Configuration parameters (--initial-ratio, --max-splats, etc.)
  - [ ] Help text and parameter validation
  - [ ] CLI integration tests

#### T4.3: Backward Compatibility
- **Description:** Ensure existing functionality remains unchanged
- **Files:** All modified files
- **Priority:** High
- **Estimated:** 4 hours
- **Dependencies:** T4.2
- **Success Criteria:**
  - [ ] All existing tests pass without modification
  - [ ] Default behavior unchanged when progressive disabled
  - [ ] No breaking changes to public APIs
  - [ ] Migration guide for new features

#### T4.4: Logging and Progress Reporting
- **Description:** Comprehensive logging and user feedback
- **Files:** `src/splat_this/core/adaptive_extract.py` (enhance)
- **Priority:** Low
- **Estimated:** 3 hours
- **Dependencies:** T4.1
- **Success Criteria:**
  - [ ] Verbose logging for each allocation step
  - [ ] Progress indicators for long-running operations
  - [ ] Error and convergence reporting
  - [ ] Debug information for troubleshooting

### Phase 5: Testing & Validation (Days 9-10)

#### T5.1: Unit Tests Implementation
- **Description:** Comprehensive unit tests for all new components
- **Files:** `tests/unit/core/test_progressive_*.py`
- **Priority:** High
- **Estimated:** 8 hours
- **Dependencies:** All implementation tasks
- **Success Criteria:**
  - [ ] Unit tests for ProgressiveAllocator
  - [ ] Unit tests for ErrorGuidedPlacement
  - [ ] Unit tests for reconstruction error computation
  - [ ] Unit tests for probability sampling
  - [ ] 100% code coverage for new components

#### T5.2: Integration Tests
- **Description:** End-to-end tests for progressive allocation pipeline
- **Files:** `tests/integration/test_progressive_integration.py`
- **Priority:** High
- **Estimated:** 6 hours
- **Dependencies:** T5.1
- **Success Criteria:**
  - [ ] Full pipeline tests with real images
  - [ ] Comparison tests vs. static allocation
  - [ ] CLI integration tests with progressive parameters
  - [ ] Backward compatibility tests

#### T5.3: Performance Benchmarking
- **Description:** Performance validation against success criteria
- **Files:** `tests/performance/test_progressive_performance.py`
- **Priority:** Medium
- **Estimated:** 4 hours
- **Dependencies:** T5.2
- **Success Criteria:**
  - [ ] Processing time benchmarks (≤1.5x baseline)
  - [ ] Memory usage validation (≤1.2x baseline)
  - [ ] Quality metrics comparison (error reduction)
  - [ ] Splat efficiency measurement

#### T5.4: Visual Quality Validation
- **Description:** Manual and automated visual quality assessment
- **Files:** `tests/visual/test_progressive_quality.py`
- **Priority:** Medium
- **Estimated:** 6 hours
- **Dependencies:** T5.3
- **Success Criteria:**
  - [ ] Visual comparison with reference images
  - [ ] Quality metrics computation (PSNR, SSIM)
  - [ ] Splat distribution analysis
  - [ ] Edge case handling validation

## Task Dependencies Graph

```
T1.1 (ProgressiveConfig)
├── T1.2 (ProgressiveAllocator)
├── T1.3 (ErrorGuidedPlacement)
└── T1.4 (Configuration Integration)

T1.3 → T2.1 (Error Implementation) → T2.3 (Probability Distribution)
T2.1 → T2.2 (Visualization)
T2.3 → T2.4 (Position Sampling)

T2.4 + T1.2 → T3.1 (Initial Allocation)
T3.1 + T1.3 → T3.2 (Progressive Addition)
T3.2 → T3.3 (Convergence Detection)
T3.2 → T3.4 (Resource Management)

T3.4 → T4.1 (Integration)
T4.1 → T4.2 (CLI Parameters)
T4.2 → T4.3 (Backward Compatibility)
T4.1 → T4.4 (Logging)

T4.3 → T5.1 (Unit Tests)
T5.1 → T5.2 (Integration Tests)
T5.2 → T5.3 (Performance)
T5.3 → T5.4 (Quality Validation)
```

## Risk Mitigation

### High-Risk Tasks
- **T3.2 (Progressive Addition Loop):** Complex integration point
  - **Mitigation:** Implement in small increments with testing
- **T4.1 (AdaptiveSplatExtractor Integration):** Major API changes
  - **Mitigation:** Maintain backward compatibility, extensive testing
- **T5.3 (Performance Benchmarking):** May reveal performance issues
  - **Mitigation:** Profile early, optimize critical paths

### Medium-Risk Tasks
- **T2.4 (Position Sampling):** Potential performance bottleneck
  - **Mitigation:** Use efficient NumPy operations, profile sampling
- **T3.3 (Convergence Detection):** Algorithm may not converge
  - **Mitigation:** Implement safety limits, fallback strategies

## Success Validation

### Phase Completion Criteria
- **Phase 1:** All core classes implemented with unit tests
- **Phase 2:** Error computation working with visual validation
- **Phase 3:** Progressive allocation functional in isolation
- **Phase 4:** Full integration with CLI and backward compatibility
- **Phase 5:** All tests passing, performance criteria met

### Overall Success Metrics
- [ ] 20-30% reduction in splat count for equivalent quality
- [ ] 15% reduction in reconstruction error
- [ ] ≤1.5x processing time vs. static allocation
- [ ] ≤1.2x memory usage vs. static allocation
- [ ] All existing tests continue to pass
- [ ] New functionality accessible via CLI
- [ ] Comprehensive test coverage (>90%)
- [ ] Documentation complete and accurate