"""SplatThis - Convert images to parallax-animated SVG splats."""

__version__ = "0.1.0"
__author__ = "SplatThis Team"
__description__ = "Convert images into self-contained parallax-animated SVG splats"

from .core.extract import SplatExtractor, Gaussian
from .core.layering import LayerAssigner, ImportanceScorer, QualityController
from .core.svgout import SVGGenerator
from .utils.image import load_image

__all__ = [
    "SplatExtractor",
    "Gaussian",
    "LayerAssigner",
    "ImportanceScorer",
    "QualityController",
    "SVGGenerator",
    "load_image",
]
