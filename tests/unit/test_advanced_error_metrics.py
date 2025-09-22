"""Unit tests for advanced error metrics module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.splat_this.core.advanced_error_metrics import (
    AdvancedErrorMetrics,
    ContentRegion,
    PerceptualMetric,
    FrequencyBand,
    LPIPSCalculator,
    FrequencyAnalyzer,
    ContentAwareAnalyzer,
    ComparativeQualityAssessment,
    AdvancedErrorAnalyzer,
    compute_advanced_reconstruction_error,
    compare_reconstruction_methods
)


class TestAdvancedErrorMetrics:
    """Test AdvancedErrorMetrics data class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        metrics = AdvancedErrorMetrics()

        assert metrics.lpips_score == 0.0
        assert metrics.ms_ssim_score == 0.0
        assert metrics.gradient_similarity == 0.0
        assert metrics.texture_similarity == 0.0
        assert metrics.edge_coherence == 0.0
        assert metrics.spectral_distortion == 0.0
        assert metrics.high_freq_preservation == 0.0
        assert metrics.content_weighted_error == 0.0
        assert metrics.saliency_weighted_error == 0.0
        assert metrics.semantic_consistency == 0.0
        assert metrics.quality_rank is None
        assert metrics.preference_score == 0.0
        assert isinstance(metrics.frequency_response, dict)
        assert len(metrics.frequency_response) == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AdvancedErrorMetrics(
            lpips_score=0.25,
            ms_ssim_score=0.85,
            frequency_response={'freq_low': 0.1, 'freq_high': 0.3}
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result['lpips_score'] == 0.25
        assert result['ms_ssim_score'] == 0.85
        assert result['freq_low'] == 0.1
        assert result['freq_high'] == 0.3


class TestContentRegion:
    """Test ContentRegion data class."""

    def test_init_and_properties(self):
        """Test initialization and properties."""
        region = ContentRegion(
            region_id=1,
            bbox=(10, 20, 50, 80),
            area=1200,
            content_type="textured",
            importance_weight=1.5,
            texture_complexity=0.8,
            edge_density=0.6
        )

        assert region.region_id == 1
        assert region.bbox == (10, 20, 50, 80)
        assert region.area == 1200
        assert region.content_type == "textured"
        assert region.importance_weight == 1.5
        assert region.texture_complexity == 0.8
        assert region.edge_density == 0.6

    def test_size_property(self):
        """Test size property calculation."""
        region = ContentRegion(
            region_id=1,
            bbox=(10, 20, 50, 80),  # height=40, width=60
            area=0,
            content_type="",
            importance_weight=0,
            texture_complexity=0,
            edge_density=0
        )

        assert region.size == (40, 60)


class TestLPIPSCalculator:
    """Test LPIPS calculator."""

    def test_init(self):
        """Test initialization."""
        calc = LPIPSCalculator()
        assert calc.metric_type == PerceptualMetric.LPIPS_VGG
        assert calc.patch_size == 64
        assert calc.stride == 32
        assert len(calc.conv_weights) > 0

    def test_compute_lpips_identical_images(self):
        """Test LPIPS computation for identical images."""
        calc = LPIPSCalculator()
        image = np.random.rand(128, 128, 3)

        lpips_score = calc.compute_lpips(image, image)

        assert isinstance(lpips_score, float)
        assert lpips_score >= 0.0
        assert lpips_score <= 1.0
        # Identical images should have low LPIPS score
        assert lpips_score < 0.1

    def test_compute_lpips_different_images(self):
        """Test LPIPS computation for different images."""
        calc = LPIPSCalculator()
        image1 = np.random.rand(128, 128, 3)
        image2 = np.random.rand(128, 128, 3)

        lpips_score = calc.compute_lpips(image1, image2)

        assert isinstance(lpips_score, float)
        assert lpips_score >= 0.0
        assert lpips_score <= 1.0

    def test_compute_lpips_grayscale(self):
        """Test LPIPS computation for grayscale images."""
        calc = LPIPSCalculator()
        image1 = np.random.rand(128, 128)
        image2 = np.random.rand(128, 128)

        lpips_score = calc.compute_lpips(image1, image2)

        assert isinstance(lpips_score, float)
        assert lpips_score >= 0.0

    def test_gabor_kernel(self):
        """Test Gabor kernel generation."""
        calc = LPIPSCalculator()
        kernel = calc._gabor_kernel(0, 0.1, size=5)

        assert kernel.shape == (5, 5)
        assert isinstance(kernel, np.ndarray)

    def test_extract_patches(self):
        """Test patch extraction."""
        calc = LPIPSCalculator()
        image = np.random.rand(128, 128)

        patches = calc._extract_patches(image)

        assert len(patches) > 0
        for patch in patches:
            assert patch.shape == (calc.patch_size, calc.patch_size)

    def test_extract_features(self):
        """Test feature extraction from patch."""
        calc = LPIPSCalculator()
        patch = np.random.rand(64, 64)

        features = calc._extract_features(patch)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert len(features) == len(calc.conv_weights) + 4  # Filters + statistics


class TestFrequencyAnalyzer:
    """Test frequency domain analyzer."""

    def test_init(self):
        """Test initialization."""
        analyzer = FrequencyAnalyzer()
        assert len(analyzer.frequency_bands) == 4
        assert FrequencyBand.LOW in analyzer.frequency_bands
        assert FrequencyBand.HIGH in analyzer.frequency_bands

    def test_compute_frequency_metrics_identical(self):
        """Test frequency metrics for identical images."""
        analyzer = FrequencyAnalyzer()
        image = np.random.rand(64, 64, 3)

        metrics = analyzer.compute_frequency_metrics(image, image)

        assert isinstance(metrics, dict)
        assert 'spectral_distortion' in metrics
        assert 'high_freq_preservation' in metrics
        assert 'freq_low' in metrics
        assert 'freq_high' in metrics

        # Identical images should have minimal spectral distortion
        assert metrics['spectral_distortion'] < 1e-10
        assert abs(metrics['high_freq_preservation'] - 1.0) < 0.1

    def test_compute_frequency_metrics_different(self):
        """Test frequency metrics for different images."""
        analyzer = FrequencyAnalyzer()
        image1 = np.random.rand(64, 64, 3)
        image2 = np.random.rand(64, 64, 3)

        metrics = analyzer.compute_frequency_metrics(image1, image2)

        assert isinstance(metrics, dict)
        assert metrics['spectral_distortion'] >= 0.0
        assert metrics['high_freq_preservation'] >= 0.0
        assert metrics['high_freq_preservation'] <= 1.5  # Allow some tolerance

    def test_compute_frequency_metrics_grayscale(self):
        """Test frequency metrics for grayscale images."""
        analyzer = FrequencyAnalyzer()
        image1 = np.random.rand(64, 64)
        image2 = np.random.rand(64, 64)

        metrics = analyzer.compute_frequency_metrics(image1, image2)

        assert isinstance(metrics, dict)
        assert 'spectral_distortion' in metrics
        assert len([k for k in metrics.keys() if k.startswith('freq_')]) == 4


class TestContentAwareAnalyzer:
    """Test content-aware analyzer."""

    def test_init(self):
        """Test initialization."""
        analyzer = ContentAwareAnalyzer()
        assert analyzer.num_segments == 50
        assert analyzer.edge_detector_sigma == 1.0

    def test_analyze_content_regions(self):
        """Test content region analysis."""
        analyzer = ContentAwareAnalyzer()
        # Create test image with some structure
        image = np.zeros((100, 100, 3))
        image[25:75, 25:75] = 1.0  # Bright square
        image[40:60, 40:60] = 0.5  # Darker center

        regions = analyzer.analyze_content_regions(image)

        assert isinstance(regions, list)
        assert len(regions) > 0
        for region in regions:
            assert isinstance(region, ContentRegion)
            assert region.area > 0
            assert region.content_type in ["smooth", "textured", "bright", "dark", "general"]
            assert region.importance_weight > 0
            assert region.texture_complexity >= 0
            assert region.edge_density >= 0

    def test_analyze_content_regions_grayscale(self):
        """Test content region analysis for grayscale images."""
        analyzer = ContentAwareAnalyzer()
        image = np.random.rand(64, 64)

        regions = analyzer.analyze_content_regions(image)

        assert isinstance(regions, list)
        # Should handle grayscale input properly

    def test_compute_content_weighted_error(self):
        """Test content-weighted error computation."""
        analyzer = ContentAwareAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = np.random.rand(64, 64, 3)

        # Create mock regions
        regions = [
            ContentRegion(1, (10, 10, 30, 30), 400, "textured", 1.5, 0.8, 0.6),
            ContentRegion(2, (40, 40, 60, 60), 400, "smooth", 0.8, 0.2, 0.1)
        ]

        error = analyzer.compute_content_weighted_error(target, rendered, regions)

        assert isinstance(error, float)
        assert error >= 0.0

    def test_classify_content(self):
        """Test content classification."""
        analyzer = ContentAwareAnalyzer()

        # Test smooth region (low variation)
        smooth_image = np.full((64, 64, 3), 0.5)
        smooth_mask = np.ones((64, 64), dtype=bool)
        smooth_type = analyzer._classify_content(smooth_image, smooth_mask)
        assert smooth_type == "smooth"

        # Test bright region
        bright_image = np.full((64, 64, 3), 0.9)
        bright_type = analyzer._classify_content(bright_image, smooth_mask)
        assert bright_type == "bright"

        # Test dark region
        dark_image = np.full((64, 64, 3), 0.1)
        dark_type = analyzer._classify_content(dark_image, smooth_mask)
        assert dark_type == "dark"

    def test_compute_importance_weight(self):
        """Test importance weight computation."""
        analyzer = ContentAwareAnalyzer()
        image = np.random.rand(64, 64, 3)
        mask = np.ones((64, 64), dtype=bool)

        # Test different content types
        textured_weight = analyzer._compute_importance_weight(image, mask, "textured")
        smooth_weight = analyzer._compute_importance_weight(image, mask, "smooth")

        assert textured_weight > smooth_weight  # Textured should be more important
        assert textured_weight > 0
        assert smooth_weight > 0

    def test_compute_texture_complexity(self):
        """Test texture complexity computation."""
        analyzer = ContentAwareAnalyzer()

        # Smooth image
        smooth_image = np.full((64, 64), 0.5)
        smooth_mask = np.ones((64, 64), dtype=bool)
        smooth_complexity = analyzer._compute_texture_complexity(smooth_image, smooth_mask)

        # Textured image
        textured_image = np.random.rand(64, 64)
        textured_complexity = analyzer._compute_texture_complexity(textured_image, smooth_mask)

        assert isinstance(smooth_complexity, float)
        assert isinstance(textured_complexity, float)
        assert textured_complexity > smooth_complexity

    def test_compute_edge_density(self):
        """Test edge density computation."""
        analyzer = ContentAwareAnalyzer()

        # Image with edges
        image = np.zeros((64, 64))
        image[30:34, :] = 1.0  # Horizontal edge
        mask = np.ones((64, 64), dtype=bool)

        edge_density = analyzer._compute_edge_density(image, mask)

        assert isinstance(edge_density, float)
        assert edge_density >= 0.0
        assert edge_density <= 1.0


class TestComparativeQualityAssessment:
    """Test comparative quality assessment."""

    def test_init(self):
        """Test initialization."""
        assessor = ComparativeQualityAssessment()
        assert assessor.methods_compared == 0
        assert len(assessor.comparison_history) == 0

    def test_compare_methods_insufficient_methods(self):
        """Test comparison with insufficient methods."""
        assessor = ComparativeQualityAssessment()
        target = np.random.rand(64, 64, 3)
        reconstructions = {"method1": np.random.rand(64, 64, 3)}

        with pytest.raises(ValueError, match="Need at least 2 methods"):
            assessor.compare_methods(target, reconstructions)

    def test_compare_methods_valid(self):
        """Test valid method comparison."""
        assessor = ComparativeQualityAssessment()
        target = np.random.rand(64, 64, 3)
        method1 = target + 0.1 * np.random.rand(64, 64, 3)  # Similar to target
        method2 = np.random.rand(64, 64, 3)  # Different from target

        reconstructions = {
            "method1": method1,
            "method2": method2
        }

        results = assessor.compare_methods(target, reconstructions)

        assert isinstance(results, dict)
        assert "method1" in results
        assert "method2" in results

        for method_name, metrics in results.items():
            assert "basic" in metrics
            assert "advanced" in metrics
            assert "combined_score" in metrics
            assert "quality_rank" in metrics["advanced"]
            assert isinstance(metrics["combined_score"], float)
            assert 0.0 <= metrics["combined_score"] <= 1.0

        # Method1 should rank better (closer to target)
        rank1 = results["method1"]["advanced"]["quality_rank"]
        rank2 = results["method2"]["advanced"]["quality_rank"]
        score1 = results["method1"]["combined_score"]
        score2 = results["method2"]["combined_score"]

        assert rank1 <= rank2  # Lower rank number is better
        assert score1 >= score2  # Higher score is better

    def test_compute_combined_score(self):
        """Test combined score computation."""
        assessor = ComparativeQualityAssessment()

        # Mock metrics
        from src.splat_this.core.error_analysis import ErrorMetrics
        basic_metrics = ErrorMetrics(l1_error=0.1, ssim_score=0.8, psnr=30.0)
        advanced_metrics = AdvancedErrorMetrics(
            lpips_score=0.2,
            high_freq_preservation=0.9,
            content_weighted_error=0.15
        )

        score = assessor._compute_combined_score(basic_metrics, advanced_metrics)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_rank_methods(self):
        """Test method ranking."""
        assessor = ComparativeQualityAssessment()

        method_metrics = {
            "good_method": {"combined_score": 0.8},
            "bad_method": {"combined_score": 0.3},
            "medium_method": {"combined_score": 0.6}
        }

        rankings = assessor._rank_methods(method_metrics)

        assert rankings["good_method"] == 1    # Best rank
        assert rankings["medium_method"] == 2  # Middle rank
        assert rankings["bad_method"] == 3     # Worst rank


class TestAdvancedErrorAnalyzer:
    """Test advanced error analyzer."""

    def test_init(self):
        """Test initialization."""
        analyzer = AdvancedErrorAnalyzer()
        assert hasattr(analyzer, 'lpips_calculator')
        assert hasattr(analyzer, 'frequency_analyzer')
        assert hasattr(analyzer, 'content_analyzer')
        assert hasattr(analyzer, 'comparative_assessor')

    def test_compute_advanced_metrics(self):
        """Test advanced metrics computation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = np.random.rand(64, 64, 3)

        metrics = analyzer.compute_advanced_metrics(target, rendered)

        assert isinstance(metrics, AdvancedErrorMetrics)
        assert metrics.lpips_score >= 0.0
        assert metrics.ms_ssim_score >= 0.0
        assert metrics.gradient_similarity >= 0.0
        assert metrics.texture_similarity >= 0.0
        assert metrics.edge_coherence >= 0.0
        assert metrics.spectral_distortion >= 0.0
        assert metrics.high_freq_preservation >= 0.0
        assert metrics.content_weighted_error >= 0.0

    def test_create_advanced_error_map_content_weighted(self):
        """Test content-weighted error map creation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = np.random.rand(64, 64, 3)

        error_map = analyzer.create_advanced_error_map(target, rendered, 'content_weighted')

        assert isinstance(error_map, np.ndarray)
        assert error_map.shape == target.shape[:2]
        assert np.all(error_map >= 0)

    def test_create_advanced_error_map_frequency_weighted(self):
        """Test frequency-weighted error map creation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = np.random.rand(64, 64, 3)

        error_map = analyzer.create_advanced_error_map(target, rendered, 'frequency_weighted')

        assert isinstance(error_map, np.ndarray)
        assert error_map.shape == target.shape[:2]
        assert np.all(error_map >= 0)

    def test_compute_multiscale_ssim(self):
        """Test multi-scale SSIM computation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = target.copy()  # Identical images

        ms_ssim = analyzer._compute_multiscale_ssim(target, rendered)

        assert isinstance(ms_ssim, float)
        assert ms_ssim >= 0.0
        assert ms_ssim <= 1.0
        # Identical images should have high MS-SSIM
        assert ms_ssim > 0.9

    def test_compute_gradient_similarity(self):
        """Test gradient similarity computation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = target.copy()  # Identical images

        grad_sim = analyzer._compute_gradient_similarity(target, rendered)

        assert isinstance(grad_sim, float)
        assert grad_sim >= 0.0
        assert grad_sim <= 1.0
        # Identical images should have high gradient similarity
        assert grad_sim > 0.9

    def test_compute_texture_similarity(self):
        """Test texture similarity computation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = target.copy()  # Identical images

        texture_sim = analyzer._compute_texture_similarity(target, rendered)

        assert isinstance(texture_sim, float)
        assert texture_sim >= 0.0
        assert texture_sim <= 1.0

    def test_compute_edge_coherence(self):
        """Test edge coherence computation."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(64, 64, 3)
        rendered = target.copy()  # Identical images

        edge_coherence = analyzer._compute_edge_coherence(target, rendered)

        assert isinstance(edge_coherence, float)
        assert edge_coherence >= 0.0
        assert edge_coherence <= 1.0
        # Identical images should have perfect edge coherence
        assert edge_coherence == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_advanced_reconstruction_error(self):
        """Test advanced reconstruction error convenience function."""
        target = np.random.rand(32, 32, 3)
        rendered = np.random.rand(32, 32, 3)

        metrics = compute_advanced_reconstruction_error(target, rendered)

        assert isinstance(metrics, AdvancedErrorMetrics)

    def test_compare_reconstruction_methods(self):
        """Test reconstruction method comparison convenience function."""
        target = np.random.rand(32, 32, 3)
        reconstructions = {
            "method1": np.random.rand(32, 32, 3),
            "method2": np.random.rand(32, 32, 3)
        }

        results = compare_reconstruction_methods(target, reconstructions)

        assert isinstance(results, dict)
        assert len(results) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_image(self):
        """Test with empty/zero images."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.zeros((32, 32, 3))
        rendered = np.zeros((32, 32, 3))

        metrics = analyzer.compute_advanced_metrics(target, rendered)
        assert isinstance(metrics, AdvancedErrorMetrics)

    def test_single_pixel_image(self):
        """Test with very small images."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(2, 2, 3)
        rendered = np.random.rand(2, 2, 3)

        # Should not crash
        metrics = analyzer.compute_advanced_metrics(target, rendered)
        assert isinstance(metrics, AdvancedErrorMetrics)

    def test_mismatched_dimensions(self):
        """Test with mismatched image dimensions."""
        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(32, 32, 3)
        rendered = np.random.rand(64, 64, 3)

        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            analyzer.compute_advanced_metrics(target, rendered)

    @patch('src.splat_this.core.advanced_error_metrics.slic')
    def test_segmentation_fallback(self, mock_slic):
        """Test segmentation fallback when slic fails."""
        mock_slic.side_effect = Exception("Segmentation failed")

        analyzer = ContentAwareAnalyzer()
        image = np.random.rand(32, 32, 3)

        # Should fall back to felzenszwalb
        with patch('src.splat_this.core.advanced_error_metrics.felzenszwalb') as mock_felz:
            mock_felz.return_value = np.ones((32, 32), dtype=int)
            regions = analyzer.analyze_content_regions(image)
            assert isinstance(regions, list)

    @patch('skimage.feature.local_binary_pattern')
    def test_texture_similarity_fallback(self, mock_lbp):
        """Test texture similarity fallback when LBP fails."""
        mock_lbp.side_effect = Exception("LBP failed")

        analyzer = AdvancedErrorAnalyzer()
        target = np.random.rand(32, 32, 3)
        rendered = np.random.rand(32, 32, 3)

        # Should fall back to simple texture measure
        texture_sim = analyzer._compute_texture_similarity(target, rendered)
        assert isinstance(texture_sim, float)
        assert 0.0 <= texture_sim <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])