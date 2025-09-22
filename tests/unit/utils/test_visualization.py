#!/usr/bin/env python3
"""Unit tests for visualization utilities."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set matplotlib to use a non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

from src.splat_this.utils.visualization import (
    visualize_error_map,
    visualize_probability_map,
    visualize_side_by_side_comparison,
    visualize_splat_placement,
    visualize_progressive_allocation_steps,
    create_error_histogram,
    export_to_png,
    create_debug_summary
)


class TestVisualizeErrorMap:
    """Test cases for error map visualization."""

    def test_basic_visualization(self):
        """Test basic error map visualization."""
        error_map = np.random.rand(10, 10).astype(np.float32)
        rgb_array = visualize_error_map(error_map)

        # Should return RGB array with proper shape
        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3
        assert rgb_array.dtype == np.uint8

    def test_with_save_path(self):
        """Test error map visualization with file saving."""
        error_map = np.random.rand(5, 5).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_error_map.png"
            rgb_array = visualize_error_map(error_map, save_path=str(save_path))

            # Should create the file
            assert save_path.exists()
            assert rgb_array.shape[2] == 3

    def test_custom_parameters(self):
        """Test error map visualization with custom parameters."""
        error_map = np.random.rand(8, 8).astype(np.float32)
        rgb_array = visualize_error_map(
            error_map,
            title="Custom Title",
            colormap="viridis",
            show_colorbar=False,
            figsize=(6, 4)
        )

        assert rgb_array.dtype == np.uint8
        assert len(rgb_array.shape) == 3


class TestVisualizeProbabilityMap:
    """Test cases for probability map visualization."""

    def test_basic_visualization(self):
        """Test basic probability map visualization."""
        prob_map = np.random.rand(10, 10).astype(np.float32)
        prob_map = prob_map / np.sum(prob_map)  # Normalize
        rgb_array = visualize_probability_map(prob_map)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3
        assert rgb_array.dtype == np.uint8

    def test_with_sampled_positions(self):
        """Test probability map visualization with sample positions."""
        prob_map = np.random.rand(10, 10).astype(np.float32)
        prob_map = prob_map / np.sum(prob_map)
        positions = [(2, 3), (5, 7), (8, 1)]

        rgb_array = visualize_probability_map(prob_map, positions)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3

    def test_with_save_path(self):
        """Test probability map visualization with file saving."""
        prob_map = np.random.rand(5, 5).astype(np.float32)
        prob_map = prob_map / np.sum(prob_map)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_prob_map.png"
            rgb_array = visualize_probability_map(prob_map, save_path=str(save_path))

            assert save_path.exists()
            assert rgb_array.shape[2] == 3

    def test_empty_positions(self):
        """Test probability map visualization with empty positions list."""
        prob_map = np.random.rand(5, 5).astype(np.float32)
        positions = []

        rgb_array = visualize_probability_map(prob_map, positions)
        assert rgb_array.dtype == np.uint8


class TestSideBySideComparison:
    """Test cases for side-by-side comparison visualization."""

    def test_grayscale_images(self):
        """Test side-by-side comparison with grayscale images."""
        target = np.random.rand(10, 10).astype(np.float32)
        rendered = np.random.rand(10, 10).astype(np.float32)
        error_map = np.abs(target - rendered)

        rgb_array = visualize_side_by_side_comparison(target, rendered, error_map)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3
        assert rgb_array.dtype == np.uint8

    def test_rgb_images(self):
        """Test side-by-side comparison with RGB images."""
        target = np.random.rand(10, 10, 3).astype(np.float32)
        rendered = np.random.rand(10, 10, 3).astype(np.float32)
        error_map = np.mean(np.abs(target - rendered), axis=2)

        rgb_array = visualize_side_by_side_comparison(target, rendered, error_map)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3

    def test_with_save_path(self):
        """Test side-by-side comparison with file saving."""
        target = np.random.rand(5, 5).astype(np.float32)
        rendered = np.random.rand(5, 5).astype(np.float32)
        error_map = np.abs(target - rendered)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_comparison.png"
            rgb_array = visualize_side_by_side_comparison(
                target, rendered, error_map, save_path=str(save_path)
            )

            assert save_path.exists()
            assert rgb_array.shape[2] == 3


class TestSplatPlacement:
    """Test cases for splat placement visualization."""

    def test_basic_placement(self):
        """Test basic splat placement visualization."""
        image = np.random.rand(20, 20).astype(np.float32)
        positions = [(5.0, 8.0), (12.0, 15.0), (18.0, 3.0)]

        rgb_array = visualize_splat_placement(image, positions)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3
        assert rgb_array.dtype == np.uint8

    def test_with_scales(self):
        """Test splat placement visualization with scale information."""
        image = np.random.rand(20, 20, 3).astype(np.float32)
        positions = [(5.0, 8.0), (12.0, 15.0)]
        scales = [2.0, 4.0]

        rgb_array = visualize_splat_placement(image, positions, scales)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3

    def test_empty_positions(self):
        """Test splat placement visualization with empty positions."""
        image = np.random.rand(10, 10).astype(np.float32)
        positions = []

        rgb_array = visualize_splat_placement(image, positions)
        assert rgb_array.dtype == np.uint8

    def test_with_save_path(self):
        """Test splat placement visualization with file saving."""
        image = np.random.rand(10, 10).astype(np.float32)
        positions = [(3.0, 4.0), (7.0, 8.0)]

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_splats.png"
            rgb_array = visualize_splat_placement(image, positions, save_path=str(save_path))

            assert save_path.exists()
            assert rgb_array.shape[2] == 3


class TestProgressiveAllocationSteps:
    """Test cases for progressive allocation step visualization."""

    def test_basic_steps_visualization(self):
        """Test basic progressive steps visualization."""
        steps_data = []
        for i in range(3):
            steps_data.append({
                'iteration': i * 10,
                'error_map': np.random.rand(10, 10) * (0.5 ** i),  # Decreasing error
                'splat_count': 50 + i * 20,
                'mean_error': 0.1 * (0.8 ** i),
                'added_positions': [(i*2, i*3), (i*3, i*2)]
            })

        rgb_arrays = visualize_progressive_allocation_steps(steps_data)

        assert len(rgb_arrays) == 3
        for rgb_array in rgb_arrays:
            assert len(rgb_array.shape) == 3
            assert rgb_array.shape[2] == 3
            assert rgb_array.dtype == np.uint8

    def test_with_save_dir(self):
        """Test progressive steps visualization with save directory."""
        steps_data = [{
            'iteration': 0,
            'error_map': np.random.rand(5, 5),
            'splat_count': 30,
            'mean_error': 0.1,
            'added_positions': []
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            rgb_arrays = visualize_progressive_allocation_steps(steps_data, save_dir=temp_dir)

            # Should create files in the directory
            files = list(Path(temp_dir).glob("*.png"))
            assert len(files) == 1
            assert len(rgb_arrays) == 1

    def test_empty_steps(self):
        """Test progressive steps visualization with empty data."""
        rgb_arrays = visualize_progressive_allocation_steps([])
        assert len(rgb_arrays) == 0


class TestErrorHistogram:
    """Test cases for error histogram creation."""

    def test_basic_histogram(self):
        """Test basic error histogram creation."""
        error_map = np.random.exponential(0.1, (10, 10)).astype(np.float32)
        rgb_array = create_error_histogram(error_map)

        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3
        assert rgb_array.dtype == np.uint8

    def test_with_zeros(self):
        """Test error histogram with zero values."""
        error_map = np.random.rand(10, 10).astype(np.float32)
        error_map[error_map < 0.5] = 0  # Set some values to zero

        rgb_array = create_error_histogram(error_map)
        assert rgb_array.dtype == np.uint8

    def test_custom_parameters(self):
        """Test error histogram with custom parameters."""
        error_map = np.random.rand(8, 8).astype(np.float32)
        rgb_array = create_error_histogram(
            error_map,
            title="Custom Histogram",
            bins=30,
            figsize=(6, 4)
        )

        assert rgb_array.dtype == np.uint8

    def test_with_save_path(self):
        """Test error histogram with file saving."""
        error_map = np.random.rand(5, 5).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_histogram.png"
            rgb_array = create_error_histogram(error_map, save_path=str(save_path))

            assert save_path.exists()
            assert rgb_array.shape[2] == 3


class TestExportToPNG:
    """Test cases for PNG export functionality."""

    @patch('builtins.__import__')
    def test_basic_export(self, mock_import):
        """Test basic PNG export."""
        # Mock imageio module
        mock_imageio = MagicMock()
        mock_import.return_value = mock_imageio

        image_array = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        output_path = "test_output.png"

        export_to_png(image_array, output_path)

        mock_imageio.imwrite.assert_called_once()
        args, kwargs = mock_imageio.imwrite.call_args
        assert args[0] == output_path
        assert args[1].dtype == np.uint8

    @patch('builtins.__import__')
    def test_normalized_export(self, mock_import):
        """Test PNG export with normalization."""
        # Mock imageio module
        mock_imageio = MagicMock()
        mock_import.return_value = mock_imageio

        image_array = np.random.rand(10, 10).astype(np.float32)
        output_path = "test_output.png"

        export_to_png(image_array, output_path, normalize=True)

        mock_imageio.imwrite.assert_called_once()
        args, kwargs = mock_imageio.imwrite.call_args
        assert args[1].dtype == np.uint8
        assert np.max(args[1]) <= 255
        assert np.min(args[1]) >= 0

    @patch('builtins.__import__')
    def test_rgb_export(self, mock_import):
        """Test PNG export with RGB image."""
        # Mock imageio module
        mock_imageio = MagicMock()
        mock_import.return_value = mock_imageio

        image_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        output_path = "test_output.png"

        export_to_png(image_array, output_path)

        mock_imageio.imwrite.assert_called_once()
        args, kwargs = mock_imageio.imwrite.call_args
        assert args[1].shape == (10, 10, 3)


class TestDebugSummary:
    """Test cases for comprehensive debug summary creation."""

    def test_basic_debug_summary(self):
        """Test basic debug summary creation."""
        target = np.random.rand(10, 10, 3).astype(np.float32)
        rendered = np.random.rand(10, 10, 3).astype(np.float32)
        error_map = np.mean(np.abs(target - rendered), axis=2)
        prob_map = np.random.rand(10, 10).astype(np.float32)
        prob_map = prob_map / np.sum(prob_map)
        positions = [(2, 3), (5, 7)]

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = create_debug_summary(
                target, rendered, error_map, prob_map, positions, temp_dir
            )

            # Should create multiple files
            assert len(saved_files) == 6
            expected_keys = ['error_map', 'probability_map', 'comparison',
                           'histogram', 'raw_error', 'raw_probability']
            for key in expected_keys:
                assert key in saved_files
                assert Path(saved_files[key]).exists()

    def test_debug_summary_with_prefix(self):
        """Test debug summary creation with custom prefix."""
        target = np.random.rand(5, 5).astype(np.float32)
        rendered = np.random.rand(5, 5).astype(np.float32)
        error_map = np.abs(target - rendered)
        prob_map = np.random.rand(5, 5).astype(np.float32)
        prob_map = prob_map / np.sum(prob_map)
        positions = []

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = create_debug_summary(
                target, rendered, error_map, prob_map, positions, temp_dir, prefix="custom"
            )

            # Check that files have custom prefix
            for file_path in saved_files.values():
                filename = Path(file_path).name
                assert filename.startswith("custom_")

    def test_debug_summary_creates_directory(self):
        """Test that debug summary creates directory if it doesn't exist."""
        target = np.random.rand(5, 5).astype(np.float32)
        rendered = np.random.rand(5, 5).astype(np.float32)
        error_map = np.abs(target - rendered)
        prob_map = np.random.rand(5, 5).astype(np.float32)
        prob_map = prob_map / np.sum(prob_map)
        positions = []

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "new_subdir"
            assert not save_dir.exists()

            saved_files = create_debug_summary(
                target, rendered, error_map, prob_map, positions, str(save_dir)
            )

            # Directory should be created
            assert save_dir.exists()
            assert len(saved_files) > 0


# Integration test to ensure all visualization functions work together
class TestVisualizationIntegration:
    """Integration tests for visualization utilities."""

    def test_full_workflow(self):
        """Test a complete visualization workflow."""
        # Create synthetic data
        target = np.random.rand(20, 20, 3).astype(np.float32)
        rendered = target + np.random.normal(0, 0.1, target.shape).astype(np.float32)
        rendered = np.clip(rendered, 0, 1)

        error_map = np.mean(np.abs(target - rendered), axis=2)
        prob_map = error_map / np.sum(error_map)
        positions = [(5, 7), (12, 15), (18, 3)]
        scales = [2.0, 3.5, 1.8]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test all visualization functions
            error_rgb = visualize_error_map(error_map)
            prob_rgb = visualize_probability_map(prob_map, positions)
            comp_rgb = visualize_side_by_side_comparison(target, rendered, error_map)
            splat_rgb = visualize_splat_placement(target, positions, scales)
            hist_rgb = create_error_histogram(error_map)

            # All should produce valid RGB arrays
            for rgb_array in [error_rgb, prob_rgb, comp_rgb, splat_rgb, hist_rgb]:
                assert len(rgb_array.shape) == 3
                assert rgb_array.shape[2] == 3
                assert rgb_array.dtype == np.uint8

            # Test debug summary
            debug_files = create_debug_summary(
                target, rendered, error_map, prob_map, positions, temp_dir
            )

            assert len(debug_files) == 6
            for file_path in debug_files.values():
                assert Path(file_path).exists()