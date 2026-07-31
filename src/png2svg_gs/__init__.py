"""
PNG to SVG Gaussian Splatting Pipeline

A lean, practical tool for converting PNG images to SVG using anisotropic 2D Gaussian splats.
Based on Image-GS methodology but optimized for real-world PNG→SVG conversion.
"""

from ._version import __version__
from .converter import PNG2SVGConverter
from .io import (
    evaluate_svg_export_quality,
    load_png,
    load_splats_json,
    render_splats_preview_png,
    save_pptx_with_splat_png,
    save_pptx_with_splats,
    save_side_by_side_html,
    save_svg,
    validate_export_roundtrip,
)
from .splat import GaussianSplat, RawSplat

__all__ = [
    "__version__",
    "GaussianSplat",
    "RawSplat",
    "load_png",
    "save_svg",
    "save_pptx_with_splats",
    "save_pptx_with_splat_png",
    "load_splats_json",
    "render_splats_preview_png",
    "save_side_by_side_html",
    "evaluate_svg_export_quality",
    "validate_export_roundtrip",
    "PNG2SVGConverter",
]
