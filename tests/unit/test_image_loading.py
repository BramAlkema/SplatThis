"""Tests for image loading and validation functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import io

from splat_this.utils.image import (
    ImageLoader,
    load_image,
    validate_image_dimensions,
    get_image_info,
    suggest_processing_options,
)


class TestImageLoader:
    """Test the ImageLoader class."""

    def create_test_image(self, size=(200, 100), mode="RGB", format="PNG"):
        """Create a test image in memory."""
        img = Image.new(mode, size, color=(128, 64, 32))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes.getvalue()

    def create_test_file(self, content, suffix=".png"):
        """Create a temporary test file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)

    def test_supported_formats(self):
        """Test that supported formats are correctly defined."""
        assert ".png" in ImageLoader.SUPPORTED_FORMATS
        assert ".jpg" in ImageLoader.SUPPORTED_FORMATS
        assert ".jpeg" in ImageLoader.SUPPORTED_FORMATS
        assert ".gif" in ImageLoader.SUPPORTED_FORMATS

    def test_valid_png_loading(self):
        """Test loading a valid PNG image."""
        png_data = self.create_test_image(size=(200, 100), format="PNG")
        test_file = self.create_test_file(png_data, ".png")

        try:
            loader = ImageLoader(test_file)
            array = loader.load()

            assert isinstance(array, np.ndarray)
            assert array.shape == (100, 200, 3)  # height, width, channels
            assert array.dtype == np.uint8
        finally:
            test_file.unlink()

    def test_valid_jpg_loading(self):
        """Test loading a valid JPEG image."""
        jpg_data = self.create_test_image(size=(300, 150), format="JPEG")
        test_file = self.create_test_file(jpg_data, ".jpg")

        try:
            loader = ImageLoader(test_file)
            array = loader.load()

            assert isinstance(array, np.ndarray)
            assert array.shape == (150, 300, 3)
            assert array.dtype == np.uint8
        finally:
            test_file.unlink()

    def test_rgba_to_rgb_conversion(self):
        """Test RGBA to RGB conversion with transparency handling."""
        # Create RGBA image with transparency
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))  # Semi-transparent red
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        test_file = self.create_test_file(img_bytes.getvalue(), ".png")

        try:
            loader = ImageLoader(test_file)
            array = loader.load()

            assert array.shape == (100, 100, 3)  # Should be RGB
            assert array.dtype == np.uint8
            # Should have been composited on white background
            assert array[0, 0, 0] > 128  # Red channel mixed with white
        finally:
            test_file.unlink()

    def test_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        # Create a text file with image extension
        test_file = self.create_test_file(b"not an image", ".bmp")

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                ImageLoader(test_file)
        finally:
            test_file.unlink()

    def test_nonexistent_file(self):
        """Test error handling for non-existent files."""
        nonexistent = Path("does_not_exist.png")
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            ImageLoader(nonexistent)

    def test_directory_path(self):
        """Test error handling when path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir) / "test.png"
            dir_path.mkdir()

            with pytest.raises(ValueError, match="Path is not a file"):
                ImageLoader(dir_path)

    def test_corrupted_image(self):
        """Test error handling for corrupted image files."""
        # Create a file that looks like an image but is corrupted
        test_file = self.create_test_file(b"fake image data", ".png")

        try:
            loader = ImageLoader(test_file)
            with pytest.raises(RuntimeError, match="Failed to load"):
                loader.load()
        finally:
            test_file.unlink()


class TestGIFHandling:
    """Test GIF-specific functionality."""

    def create_animated_gif(self, frames=3, size=(100, 100)):
        """Create a simple animated GIF for testing."""
        images = []
        for i in range(frames):
            # Create different colored frames
            color = (i * 80, (i * 60) % 255, (i * 40) % 255)
            img = Image.new("RGB", size, color)
            images.append(img)

        gif_bytes = io.BytesIO()
        images[0].save(
            gif_bytes,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0,
        )
        gif_bytes.seek(0)
        return gif_bytes.getvalue()

    def test_static_gif_loading(self):
        """Test loading a static (non-animated) GIF."""
        # Create single frame GIF
        gif_data = self.create_animated_gif(frames=1)
        test_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        test_file.write(gif_data)
        test_file.close()
        test_path = Path(test_file.name)

        try:
            loader = ImageLoader(test_path, frame=0)
            array = loader.load()

            assert isinstance(array, np.ndarray)
            assert array.shape == (100, 100, 3)
        finally:
            test_path.unlink()

    def test_animated_gif_frame_extraction(self):
        """Test extracting specific frames from animated GIF."""
        gif_data = self.create_animated_gif(frames=3)
        test_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        test_file.write(gif_data)
        test_file.close()
        test_path = Path(test_file.name)

        try:
            # Test frame 0
            loader0 = ImageLoader(test_path, frame=0)
            array0 = loader0.load()

            # Test frame 1
            loader1 = ImageLoader(test_path, frame=1)
            array1 = loader1.load()

            # Arrays should be different (different colored frames)
            assert not np.array_equal(array0, array1)
            assert array0.shape == array1.shape == (100, 100, 3)
        finally:
            test_path.unlink()

    def test_gif_frame_out_of_range(self):
        """Test error handling for out-of-range frame numbers."""
        gif_data = self.create_animated_gif(frames=2)
        test_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        test_file.write(gif_data)
        test_file.close()
        test_path = Path(test_file.name)

        try:
            loader = ImageLoader(test_path, frame=5)  # Frame 5 doesn't exist
            with pytest.raises(ValueError, match="Frame 5 not available"):
                loader.load()
        finally:
            test_path.unlink()

    def test_negative_frame_number(self):
        """Test error handling for negative frame numbers."""
        gif_data = self.create_animated_gif(frames=2)
        test_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        test_file.write(gif_data)
        test_file.close()
        test_path = Path(test_file.name)

        try:
            with pytest.raises(ValueError, match="Frame number must be non-negative"):
                ImageLoader(test_path, frame=-1)
        finally:
            test_path.unlink()


class TestImageValidation:
    """Test image dimension validation."""

    def test_valid_dimensions(self):
        """Test validation with valid image dimensions."""
        # Create valid image array
        image = np.zeros((500, 800, 3), dtype=np.uint8)
        validate_image_dimensions(image)  # Should not raise

    def test_too_small_image(self):
        """Test validation error for too small images."""
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image too small"):
            validate_image_dimensions(small_image)

    def test_too_large_image(self):
        """Test validation error for too large images."""
        # Create dimensions that are too large
        with pytest.raises(ValueError, match="Image too large"):
            # Don't actually create the array, just test the validation logic
            fake_large_image = np.zeros((10000, 10000, 3), dtype=np.uint8)
            validate_image_dimensions(fake_large_image)

    def test_too_many_pixels(self):
        """Test validation error for too many total pixels."""
        # Create an image that's not too large in either dimension but has too many pixels
        image = np.zeros((6000, 6000, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too many pixels"):
            validate_image_dimensions(image)

    def test_wrong_shape(self):
        """Test validation error for wrong array shape."""
        # Grayscale image (missing color channel)
        grayscale = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB image"):
            validate_image_dimensions(grayscale)

        # RGBA image (too many channels)
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB image"):
            validate_image_dimensions(rgba)


class TestLoadImageConvenience:
    """Test the load_image convenience function."""

    def test_load_image_function(self):
        """Test the load_image convenience function."""
        # Create test image
        img = Image.new("RGB", (200, 150), (255, 128, 64))
        test_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(test_file.name)
        test_path = Path(test_file.name)

        try:
            array, (height, width) = load_image(test_path)

            assert isinstance(array, np.ndarray)
            assert array.shape == (150, 200, 3)
            assert height == 150
            assert width == 200
        finally:
            test_path.unlink()


class TestImageInfo:
    """Test image information functions."""

    def test_get_image_info(self):
        """Test getting image information."""
        # Create test image
        img = Image.new("RGB", (300, 200), (255, 0, 0))
        test_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(test_file.name)
        test_path = Path(test_file.name)

        try:
            info = get_image_info(test_path)

            assert info["format"] == "PNG"
            assert info["mode"] == "RGB"
            assert info["width"] == 300
            assert info["height"] == 200
            assert info["size"] == (300, 200)
            assert not info["has_transparency"]
            assert not info["is_animated"]
            assert info["n_frames"] == 1
            assert info["memory_mb"] > 0
        finally:
            test_path.unlink()

    def test_suggest_processing_options(self):
        """Test processing option suggestions."""
        # Create small test image
        img = Image.new("RGB", (400, 300), (255, 0, 0))  # 0.12MP
        test_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(test_file.name)
        test_path = Path(test_file.name)

        try:
            suggestions = suggest_processing_options(test_path)

            assert "splats" in suggestions
            assert "layers" in suggestions
            assert isinstance(suggestions["splats"], int)
            assert isinstance(suggestions["layers"], int)
            assert suggestions["splats"] > 0
            assert suggestions["layers"] > 0
        finally:
            test_path.unlink()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_extremely_wide_image(self):
        """Test handling of extremely wide images."""
        # Create very wide but short image
        wide_image = np.zeros((100, 1000, 3), dtype=np.uint8)
        validate_image_dimensions(wide_image)  # Should pass

    def test_extremely_tall_image(self):
        """Test handling of extremely tall images."""
        # Create very tall but narrow image
        tall_image = np.zeros((1000, 100, 3), dtype=np.uint8)
        validate_image_dimensions(tall_image)  # Should pass

    def test_square_minimum_image(self):
        """Test minimum valid square image."""
        min_image = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_image_dimensions(min_image)  # Should pass

    def test_just_under_minimum(self):
        """Test image just under minimum size."""
        under_min = np.zeros((99, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            validate_image_dimensions(under_min)
