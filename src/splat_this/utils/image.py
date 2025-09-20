"""Image loading and validation utilities."""

from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple


class ImageLoader:
    """Handle loading and validation of input images."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif"}

    def __init__(self, path: Path, frame: int = 0):
        self.path = path
        self.frame = frame
        self._validate_format()

    def _validate_format(self) -> None:
        """Validate image format is supported."""
        if self.path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.path.suffix}")

    def load(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        try:
            with Image.open(self.path) as img:
                if img.format == "GIF" and img.is_animated:
                    return self._extract_gif_frame(img)
                return self._process_static_image(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.path}: {e}")

    def _extract_gif_frame(self, img: Image.Image) -> np.ndarray:
        """Extract specific frame from animated GIF."""
        if self.frame >= img.n_frames:
            raise ValueError(
                f"Frame {self.frame} not available (max: {img.n_frames-1})"
            )

        img.seek(self.frame)
        return self._process_static_image(img)

    def _process_static_image(self, img: Image.Image) -> np.ndarray:
        """Convert PIL image to RGB numpy array."""
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        return np.array(img)


def load_image(path: Path, frame: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Convenience function to load image and return array + dimensions."""
    loader = ImageLoader(path, frame)
    array = loader.load()
    return array, array.shape[:2]


def validate_image_dimensions(image: np.ndarray) -> None:
    """Validate image meets size requirements."""
    height, width = image.shape[:2]

    if width < 100 or height < 100:
        raise ValueError(f"Image too small: {width}x{height} (minimum: 100x100)")

    if width > 8192 or height > 8192:
        raise ValueError(f"Image too large: {width}x{height} (maximum: 8192x8192)")

    total_pixels = width * height
    if total_pixels > 33_554_432:  # 8192^2 / 2 for memory safety
        raise ValueError(f"Image has too many pixels: {total_pixels}")
