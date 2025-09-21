"""Splat extraction using SLIC superpixel segmentation."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import logging
from skimage.segmentation import slic
from skimage.color import rgb2lab

from ..utils.math import safe_eigendecomposition, clamp_value

logger = logging.getLogger(__name__)


@dataclass
class Gaussian:
    """Represents a single Gaussian splat with position, size, rotation, and color."""

    x: float
    y: float
    rx: float
    ry: float
    theta: float
    r: int
    g: int
    b: int
    a: float
    score: float = 0.0
    depth: float = 0.5

    def __post_init__(self):
        """Validate splat parameters."""
        self.validate()

    def validate(self) -> None:
        """Ensure splat parameters are valid."""
        if self.rx <= 0 or self.ry <= 0:
            raise ValueError(f"Invalid radii: rx={self.rx}, ry={self.ry}")

        if not (0 <= self.r <= 255 and 0 <= self.g <= 255 and 0 <= self.b <= 255):
            raise ValueError(f"Invalid RGB: ({self.r}, {self.g}, {self.b})")

        if not (0.0 <= self.a <= 1.0):
            raise ValueError(f"Invalid alpha: {self.a}")

    def area(self) -> float:
        """Calculate ellipse area."""
        return np.pi * self.rx * self.ry

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "position": (self.x, self.y),
            "size": (self.rx, self.ry),
            "rotation": self.theta,
            "color": (self.r, self.g, self.b, self.a),
            "score": self.score,
            "depth": self.depth,
        }

    def __lt__(self, other: "Gaussian") -> bool:
        """Less than comparison based on depth (for sorting)."""
        if not isinstance(other, Gaussian):
            return NotImplemented
        return self.depth < other.depth

    def __le__(self, other: "Gaussian") -> bool:
        """Less than or equal comparison based on depth."""
        if not isinstance(other, Gaussian):
            return NotImplemented
        return self.depth <= other.depth

    def __gt__(self, other: "Gaussian") -> bool:
        """Greater than comparison based on depth."""
        if not isinstance(other, Gaussian):
            return NotImplemented
        return self.depth > other.depth

    def __ge__(self, other: "Gaussian") -> bool:
        """Greater than or equal comparison based on depth."""
        if not isinstance(other, Gaussian):
            return NotImplemented
        return self.depth >= other.depth

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on all parameters."""
        if not isinstance(other, Gaussian):
            return NotImplemented
        return (
            abs(self.x - other.x) < 1e-6
            and abs(self.y - other.y) < 1e-6
            and abs(self.rx - other.rx) < 1e-6
            and abs(self.ry - other.ry) < 1e-6
            and abs(self.theta - other.theta) < 1e-6
            and self.r == other.r
            and self.g == other.g
            and self.b == other.b
            and abs(self.a - other.a) < 1e-6
            and abs(self.score - other.score) < 1e-6
            and abs(self.depth - other.depth) < 1e-6
        )

    def translate(self, dx: float, dy: float) -> "Gaussian":
        """Create a new Gaussian translated by dx, dy."""
        return Gaussian(
            x=self.x + dx,
            y=self.y + dy,
            rx=self.rx,
            ry=self.ry,
            theta=self.theta,
            r=self.r,
            g=self.g,
            b=self.b,
            a=self.a,
            score=self.score,
            depth=self.depth,
        )

    def scale(self, sx: float, sy: float = None) -> "Gaussian":
        """Create a new Gaussian scaled by sx, sy (sy=sx if not provided)."""
        if sy is None:
            sy = sx
        return Gaussian(
            x=self.x * sx,
            y=self.y * sy,
            rx=self.rx * sx,
            ry=self.ry * sy,
            theta=self.theta,
            r=self.r,
            g=self.g,
            b=self.b,
            a=self.a,
            score=self.score,
            depth=self.depth,
        )

    def rotate(
        self, angle: float, center_x: float = 0, center_y: float = 0
    ) -> "Gaussian":
        """Create a new Gaussian rotated by angle around center point."""
        import math

        # Translate to origin
        tx = self.x - center_x
        ty = self.y - center_y

        # Rotate position
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_x = tx * cos_a - ty * sin_a + center_x
        new_y = tx * sin_a + ty * cos_a + center_y

        # Update rotation angle
        new_theta = self.theta + angle

        return Gaussian(
            x=new_x,
            y=new_y,
            rx=self.rx,
            ry=self.ry,
            theta=new_theta,
            r=self.r,
            g=self.g,
            b=self.b,
            a=self.a,
            score=self.score,
            depth=self.depth,
        )

    def blend_with(self, other: "Gaussian", weight: float = 0.5) -> "Gaussian":
        """Create a new Gaussian by blending with another (weight: 0=self, 1=other)."""
        if not isinstance(other, Gaussian):
            raise TypeError("Can only blend with another Gaussian")

        weight = max(0.0, min(1.0, weight))  # Clamp weight to [0, 1]
        inv_weight = 1.0 - weight

        return Gaussian(
            x=self.x * inv_weight + other.x * weight,
            y=self.y * inv_weight + other.y * weight,
            rx=self.rx * inv_weight + other.rx * weight,
            ry=self.ry * inv_weight + other.ry * weight,
            theta=self.theta * inv_weight + other.theta * weight,
            r=int(self.r * inv_weight + other.r * weight),
            g=int(self.g * inv_weight + other.g * weight),
            b=int(self.b * inv_weight + other.b * weight),
            a=self.a * inv_weight + other.a * weight,
            score=self.score * inv_weight + other.score * weight,
            depth=self.depth * inv_weight + other.depth * weight,
        )

    def distance_to(self, other: "Gaussian") -> float:
        """Calculate Euclidean distance to another Gaussian's center."""
        if not isinstance(other, Gaussian):
            raise TypeError("Can only calculate distance to another Gaussian")
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def overlaps_with(self, other: "Gaussian") -> bool:
        """Check if this Gaussian overlaps with another (simple bounding box check)."""
        if not isinstance(other, Gaussian):
            raise TypeError("Can only check overlap with another Gaussian")

        # Simple bounding box overlap check
        self_left = self.x - self.rx
        self_right = self.x + self.rx
        self_top = self.y - self.ry
        self_bottom = self.y + self.ry

        other_left = other.x - other.rx
        other_right = other.x + other.rx
        other_top = other.y - other.ry
        other_bottom = other.y + other.ry

        return not (
            self_right < other_left
            or self_left > other_right
            or self_bottom < other_top
            or self_top > other_bottom
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Gaussian":
        """Create from dictionary."""
        return cls(
            x=data["position"][0],
            y=data["position"][1],
            rx=data["size"][0],
            ry=data["size"][1],
            theta=data["rotation"],
            r=data["color"][0],
            g=data["color"][1],
            b=data["color"][2],
            a=data["color"][3],
            score=data.get("score", 0.0),
            depth=data.get("depth", 0.5),
        )


class SplatCollection:
    """Collection of Gaussian splats with utility methods."""

    def __init__(self, splats: List[Gaussian]):
        self.splats = splats

    def __len__(self) -> int:
        """Return number of splats in collection."""
        return len(self.splats)

    def __iter__(self):
        """Make collection iterable."""
        return iter(self.splats)

    def __getitem__(self, index: int) -> Gaussian:
        """Access splats by index."""
        return self.splats[index]

    def filter_by_score(self, threshold: float) -> "SplatCollection":
        """Filter splats by minimum score."""
        filtered = [s for s in self.splats if s.score >= threshold]
        return SplatCollection(filtered)

    def filter_by_area(
        self, min_area: float = None, max_area: float = None
    ) -> "SplatCollection":
        """Filter splats by area range."""
        filtered = self.splats
        if min_area is not None:
            filtered = [s for s in filtered if s.area() >= min_area]
        if max_area is not None:
            filtered = [s for s in filtered if s.area() <= max_area]
        return SplatCollection(filtered)

    def sort_by_depth(self, reverse: bool = False) -> "SplatCollection":
        """Sort splats by depth (front to back by default)."""
        sorted_splats = sorted(self.splats, key=lambda s: s.depth, reverse=reverse)
        return SplatCollection(sorted_splats)

    def sort_by_score(self, reverse: bool = True) -> "SplatCollection":
        """Sort splats by score (highest first by default)."""
        sorted_splats = sorted(self.splats, key=lambda s: s.score, reverse=reverse)
        return SplatCollection(sorted_splats)

    def sort_by_area(self, reverse: bool = True) -> "SplatCollection":
        """Sort splats by area (largest first by default)."""
        sorted_splats = sorted(self.splats, key=lambda s: s.area(), reverse=reverse)
        return SplatCollection(sorted_splats)

    def get_statistics(self) -> dict:
        """Get collection statistics."""
        if not self.splats:
            return {
                "count": 0,
                "score_range": (0, 0),
                "area_range": (0, 0),
                "depth_range": (0, 0),
                "avg_score": 0,
                "avg_area": 0,
                "avg_depth": 0,
            }

        scores = [s.score for s in self.splats]
        areas = [s.area() for s in self.splats]
        depths = [s.depth for s in self.splats]

        return {
            "count": len(self.splats),
            "score_range": (min(scores), max(scores)),
            "area_range": (min(areas), max(areas)),
            "depth_range": (min(depths), max(depths)),
            "avg_score": sum(scores) / len(scores),
            "avg_area": sum(areas) / len(areas),
            "avg_depth": sum(depths) / len(depths),
        }

    def remove_overlapping(self, threshold: float = 0.8) -> "SplatCollection":
        """Remove overlapping splats, keeping higher-scored ones."""
        if not self.splats:
            return SplatCollection([])

        # Sort by score (highest first)
        sorted_splats = sorted(self.splats, key=lambda s: s.score, reverse=True)
        filtered = []

        for splat in sorted_splats:
            # Check if this splat overlaps significantly with any already kept
            should_keep = True
            for kept_splat in filtered:
                if splat.overlaps_with(kept_splat):
                    # Calculate overlap ratio (simplified)
                    distance = splat.distance_to(kept_splat)
                    avg_radius = (
                        splat.rx + splat.ry + kept_splat.rx + kept_splat.ry
                    ) / 4
                    overlap_ratio = max(0, 1 - distance / avg_radius)

                    if overlap_ratio > threshold:
                        should_keep = False
                        break

            if should_keep:
                filtered.append(splat)

        return SplatCollection(filtered)

    def translate_all(self, dx: float, dy: float) -> "SplatCollection":
        """Translate all splats by dx, dy."""
        translated = [splat.translate(dx, dy) for splat in self.splats]
        return SplatCollection(translated)

    def scale_all(self, sx: float, sy: float = None) -> "SplatCollection":
        """Scale all splats by sx, sy."""
        scaled = [splat.scale(sx, sy) for splat in self.splats]
        return SplatCollection(scaled)

    def to_list(self) -> List[Gaussian]:
        """Convert collection back to list."""
        return list(self.splats)

    def to_dicts(self) -> List[dict]:
        """Convert all splats to dictionaries."""
        return [splat.to_dict() for splat in self.splats]

    @classmethod
    def from_dicts(cls, data_list: List[dict]) -> "SplatCollection":
        """Create collection from list of dictionaries."""
        splats = [Gaussian.from_dict(data) for data in data_list]
        return cls(splats)

    def merge_with(self, other: "SplatCollection") -> "SplatCollection":
        """Merge with another collection."""
        if not isinstance(other, SplatCollection):
            raise TypeError("Can only merge with another SplatCollection")
        combined_splats = self.splats + other.splats
        return SplatCollection(combined_splats)


class SplatExtractor:
    """Extract Gaussian splats from images using SLIC segmentation."""

    def __init__(self, k: float = 2.5, base_alpha: float = 0.65):
        self.k = k
        self.base_alpha = base_alpha

    def extract_splats(self, image: np.ndarray, n_splats: int) -> List[Gaussian]:
        """Extract Gaussian splats from image using SLIC segmentation."""
        height, width = image.shape[:2]
        logger.info(
            f"Extracting {n_splats} splats from {width}×{height} image using SLIC"
        )

        # Convert to LAB color space for better perceptual uniformity
        lab_image = rgb2lab(image)

        # Perform SLIC superpixel segmentation
        segments = slic(
            lab_image,
            n_segments=n_splats,
            compactness=10.0,
            sigma=1.0,
            start_label=1,
            convert2lab=False,  # Already converted
        )

        logger.debug(f"SLIC generated {len(np.unique(segments))} unique segments")

        # Extract splats from segments
        splats = []
        for segment_id in np.unique(segments):
            if segment_id == 0:  # Skip background
                continue

            # Create mask for this segment
            mask = segments == segment_id

            # Skip very small segments
            pixel_count = np.sum(mask)
            if pixel_count < 10:
                continue

            try:
                splat = self._segment_to_gaussian(image, mask, segment_id)
                if splat is not None:
                    splats.append(splat)
            except Exception as e:
                logger.warning(
                    f"Failed to extract splat from segment {segment_id}: {e}"
                )
                continue

        logger.info(f"Successfully extracted {len(splats)} splats")
        return splats

    def _segment_to_gaussian(
        self, image: np.ndarray, mask: np.ndarray, segment_id: int
    ) -> Optional[Gaussian]:
        """Convert a segmented region to a Gaussian splat."""
        # Get pixel coordinates and colors for this segment
        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return None

        # Calculate centroid
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))

        # Calculate covariance matrix for ellipse parameters
        coords = np.column_stack([x_coords - centroid_x, y_coords - centroid_y])

        if len(coords) < 2:
            return None

        # Compute covariance matrix
        cov_matrix = np.cov(coords.T)

        # Handle degenerate cases
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            logger.warning(f"Invalid covariance matrix for segment {segment_id}")
            return None

        # Eigendecomposition for ellipse parameters
        eigenvalues, eigenvectors = safe_eigendecomposition(cov_matrix)

        if eigenvalues is None:
            logger.warning(f"Eigendecomposition failed for segment {segment_id}")
            return None

        # Simple, direct eigenvalue-based sizing - preserves details
        rx = max(1.0, np.sqrt(eigenvalues[0]) * self.k)
        ry = max(1.0, np.sqrt(eigenvalues[1]) * self.k)

        # Calculate rotation angle using principal eigenvector
        principal_idx = int(np.argmax(eigenvalues))
        theta = float(
            np.arctan2(
                eigenvectors[1, principal_idx],
                eigenvectors[0, principal_idx],
            )
        )

        # Extract average color from the segment
        segment_pixels = image[mask]
        avg_color = np.mean(segment_pixels, axis=0)

        r = clamp_value(int(avg_color[0]), 0, 255)
        g = clamp_value(int(avg_color[1]), 0, 255)
        b = clamp_value(int(avg_color[2]), 0, 255)

        # Calculate alpha based on segment size and color variance
        pixel_count = len(segment_pixels)
        color_variance = np.var(segment_pixels, axis=0).mean()

        # More pixels and less variance = higher alpha
        size_factor = min(1.0, pixel_count / 1000.0)
        variance_factor = max(0.3, 1.0 - (color_variance / 2500.0))
        alpha = clamp_value(self.base_alpha * size_factor * variance_factor, 0.1, 1.0)

        # Calculate quality score based on segment properties
        compactness = pixel_count / (rx * ry * np.pi) if (rx * ry) > 0 else 0
        score = clamp_value(compactness * (1.0 - color_variance / 10000.0), 0.0, 1.0)

        try:
            return Gaussian(
                x=centroid_x,
                y=centroid_y,
                rx=rx,
                ry=ry,
                theta=theta,
                r=r,
                g=g,
                b=b,
                a=alpha,
                score=score,
            )
        except ValueError as e:
            logger.warning(f"Invalid Gaussian parameters for segment {segment_id}: {e}")
            return None
