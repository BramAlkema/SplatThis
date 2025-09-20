"""Image loading and validation utilities."""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """Handle loading and validation of input images."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif"}

    def __init__(self, path: Path, frame: int = 0):
        self.path = path
        self.frame = frame
        self._validate_format()
        self._validate_frame_number()

    def _validate_format(self) -> None:
        """Validate image format is supported."""
        if not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            supported = ", ".join(sorted(self.SUPPORTED_FORMATS))
            raise ValueError(
                f"Unsupported format '{suffix}'. Supported formats: {supported}"
            )

    def _validate_frame_number(self) -> None:
        """Validate frame number is non-negative."""
        if self.frame < 0:
            raise ValueError(f"Frame number must be non-negative, got {self.frame}")

    def load(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        try:
            with Image.open(self.path) as img:
                logger.debug(f"Loaded image: {img.format} {img.mode} {img.size}")

                # Verify the image is not corrupted
                img.verify()

            # Reopen for actual processing (verify() closes the image)
            with Image.open(self.path) as img:
                if img.format == "GIF" and getattr(img, "is_animated", False):
                    return self._extract_gif_frame(img)
                return self._process_static_image(img)

        except ValueError:
            # Re-raise ValueError (frame validation errors) without wrapping
            raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to load image {self.path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {self.path}: {e}")

    def _extract_gif_frame(self, img: Image.Image) -> np.ndarray:
        """Extract specific frame from animated GIF."""
        total_frames = getattr(img, "n_frames", 1)

        if self.frame >= total_frames:
            raise ValueError(
                f"Frame {self.frame} not available. GIF has {total_frames} frames (0-{total_frames-1})"
            )

        try:
            img.seek(self.frame)
            logger.debug(f"Extracted frame {self.frame} from GIF")
            return self._process_static_image(img)
        except EOFError:
            raise ValueError(f"Cannot seek to frame {self.frame} in GIF")

    def _process_static_image(self, img: Image.Image) -> np.ndarray:
        """Convert PIL image to RGB numpy array."""
        # Apply EXIF orientation correction
        try:
            img = ImageOps.exif_transpose(img)
            logger.debug("Applied EXIF orientation correction")
        except Exception:
            # Not all images have EXIF data, continue without correction
            pass

        # Convert to RGB
        original_mode = img.mode
        if img.mode != "RGB":
            if img.mode == "RGBA":
                # Handle transparency by compositing on white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            else:
                img = img.convert("RGB")

            logger.debug(f"Converted image from {original_mode} to RGB")

        # Convert to numpy array
        array = np.array(img)

        # Validate the result
        if array.ndim != 3 or array.shape[2] != 3:
            raise RuntimeError(f"Expected RGB array, got shape {array.shape}")

        if array.dtype != np.uint8:
            raise RuntimeError(f"Expected uint8 array, got {array.dtype}")

        logger.debug(f"Converted to numpy array: {array.shape}")
        return array


def load_image(path: Path, frame: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Convenience function to load image and return array + dimensions."""
    loader = ImageLoader(path, frame)
    array = loader.load()
    return array, array.shape[:2]


def validate_image_dimensions(image: np.ndarray) -> None:
    """Validate image meets size requirements."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")

    height, width = image.shape[:2]

    # Minimum size check
    if width < 100 or height < 100:
        raise ValueError(
            f"Image too small: {width}×{height}. "
            f"Minimum size: 100×100 pixels. "
            f"Please use a larger image for meaningful splat generation."
        )

    # Maximum dimension check
    if width > 8192 or height > 8192:
        raise ValueError(
            f"Image too large: {width}×{height}. "
            f"Maximum size: 8192×8192 pixels. "
            f"Consider downscaling your image before processing."
        )

    # Total pixel count check for memory safety
    total_pixels = width * height
    max_pixels = 33_554_432  # 8192^2 / 2 for memory safety

    if total_pixels > max_pixels:
        megapixels = total_pixels / 1_000_000
        max_megapixels = max_pixels / 1_000_000
        raise ValueError(
            f"Image has too many pixels: {megapixels:.1f}MP. "
            f"Maximum: {max_megapixels:.1f}MP. "
            f"This helps prevent memory issues during processing."
        )

    logger.debug(
        f"Image dimensions validated: {width}×{height} ({total_pixels:,} pixels)"
    )


def get_image_info(path: Path) -> dict:
    """Get basic information about an image without loading it fully."""
    try:
        with Image.open(path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "has_transparency": img.mode in ("RGBA", "LA")
                or "transparency" in img.info,
                "is_animated": getattr(img, "is_animated", False),
                "n_frames": getattr(img, "n_frames", 1),
            }

            # Estimate file size in memory (RGB)
            info["memory_mb"] = (img.width * img.height * 3) / (1024 * 1024)

            return info
    except Exception as e:
        raise RuntimeError(f"Failed to get image info for {path}: {e}")


def suggest_processing_options(path: Path) -> dict:
    """Suggest optimal processing options based on image characteristics."""
    info = get_image_info(path)
    suggestions = {}

    # Splat count suggestions
    total_pixels = info["width"] * info["height"]
    if total_pixels < 500_000:  # < 0.5MP
        suggestions["splats"] = 800
    elif total_pixels < 2_000_000:  # < 2MP
        suggestions["splats"] = 1500
    elif total_pixels < 8_000_000:  # < 8MP
        suggestions["splats"] = 2500
    else:
        suggestions["splats"] = 3500

    # Layer suggestions based on complexity
    if info["memory_mb"] > 10:
        suggestions["layers"] = 6  # More layers for large images
    else:
        suggestions["layers"] = 4

    # GIF specific suggestions
    if info["is_animated"]:
        suggestions["frame_options"] = f"0-{info['n_frames']-1}"
        suggestions["note"] = (
            f"Animated GIF with {info['n_frames']} frames. Use --frame to select."
        )

    return suggestions
