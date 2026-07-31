"""Target-aware 2D Gaussian splat conversion and browser runtimes."""

from ._version import __version__
from .converter import PNG2SVGConverter
from .io import (
    evaluate_css_export_quality,
    evaluate_native_canvas_export_quality,
    evaluate_svg_export_quality,
    generate_css_splat_html,
    generate_native_canvas_html,
    generate_parallax_pixel_runtime_html,
    generate_pixel_runtime_html,
    generate_webgl_pixel_runtime_html,
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
    "generate_css_splat_html",
    "generate_native_canvas_html",
    "generate_pixel_runtime_html",
    "generate_webgl_pixel_runtime_html",
    "generate_parallax_pixel_runtime_html",
    "evaluate_css_export_quality",
    "evaluate_native_canvas_export_quality",
    "evaluate_svg_export_quality",
    "validate_export_roundtrip",
    "PNG2SVGConverter",
]
