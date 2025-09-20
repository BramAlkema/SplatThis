"""Splat extraction using SLIC superpixel segmentation."""

from dataclasses import dataclass
from typing import List
import numpy as np


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


class SplatExtractor:
    """Extract Gaussian splats from images using SLIC segmentation."""

    def __init__(self, k: float = 2.5, base_alpha: float = 0.65):
        self.k = k
        self.base_alpha = base_alpha

    def extract_splats(self, image: np.ndarray, n_splats: int) -> List[Gaussian]:
        """Extract Gaussian splats from image using SLIC."""
        # TODO: Implement SLIC segmentation and splat extraction
        # This is a placeholder implementation
        splats = []

        # Create a simple test splat for now
        height, width = image.shape[:2]
        test_splat = Gaussian(
            x=width / 2,
            y=height / 2,
            rx=10.0,
            ry=8.0,
            theta=0.0,
            r=128,
            g=128,
            b=128,
            a=self.base_alpha,
        )
        splats.append(test_splat)

        return splats
