# Claude Development Notes

## Virtual Environment
**IMPORTANT: Always activate the virtual environment before running any Python commands!**

```bash
source venv/bin/activate
```

## Testing Commands
```bash
# Run all unit tests with coverage
PYTHONPATH=. pytest tests/unit/ --cov=src --cov-report=term-missing --tb=short

# Run specific test file
PYTHONPATH=. pytest tests/unit/test_progressive_refinement.py -v --tb=short --no-cov
```

## Project Structure
- `src/splat_this/` - Main source code
- `tests/unit/` - Unit tests
- `tests/e2e/` - End-to-end tests
- `demo_*.py` - Demonstration scripts

## Task Progress

### Phase 3: Progressive Refinement - COMPLETED ✅
- ✅ Manual gradient computation (T3.1)
- ✅ SGD optimization loop (T3.2)
- ✅ Progressive refinement system (T3.3)

### Phase 4: Advanced Features - ONGOING
- ✅ Anisotropic refinement (T4.1)
- ✅ Advanced error metrics (T4.2)
- ✅ Performance optimization (T4.3)

## Recent Completion: T4.2 Advanced Error Metrics - COMPLETED ✅
- ✅ LPIPS (Learned Perceptual Image Patch Similarity) integration
- ✅ Edge-aware error weighting system
- ✅ Frequency-domain error analysis with FFT
- ✅ Region-based error aggregation with content awareness
- ✅ Comparative quality assessment framework
- ✅ Advanced error map generation (content-weighted, frequency-weighted)
- ✅ Multi-scale SSIM, gradient similarity, texture similarity, edge coherence
- ✅ Comprehensive unit tests (43 tests passing)
- ✅ Full demonstration script with performance benchmarking

## Demo Scripts Available
```bash
# T4.2 Advanced Error Metrics Demo
python demo_advanced_error_metrics.py                    # Full demo
python demo_advanced_error_metrics.py --demo lpips       # LPIPS only
python demo_advanced_error_metrics.py --demo frequency   # Frequency analysis
python demo_advanced_error_metrics.py --demo content     # Content-aware analysis
python demo_advanced_error_metrics.py --demo comparative # Method comparison
python demo_advanced_error_metrics.py --demo maps        # Error maps
python demo_advanced_error_metrics.py --demo performance # Performance benchmark

# T4.3 Performance Optimization Demo
python demo_performance_optimization.py

# T4.1 Anisotropic Refinement Demo
python demo_anisotropic_refinement.py
```

## Bug Fixes Applied
- **Gradient Calculation Fix**: Added size check for small regions in `_analyze_content_complexity()` to prevent numpy gradient calculation errors on regions smaller than 2x2 pixels. Uses variance as proxy for complexity in such cases.
- **LPIPS Non-negative Fix**: Ensured LPIPS scores are always non-negative by clipping results.
- **Gabor Kernel Size Fix**: Fixed kernel size calculation to ensure odd dimensions.
- **Content Classification Priority**: Adjusted content type classification order for better accuracy.