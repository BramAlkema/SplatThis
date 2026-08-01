"""Backward-compatible facade for artifact support functions.

Production code imports the focused storage, quality, evaluation, reporting,
and round-trip modules directly.  This module preserves the established API.
"""

from .artifact_evaluation import (  # noqa: F401
    _evaluate_browser_export_quality,
    evaluate_css_export_quality,
    evaluate_native_canvas_export_quality,
    evaluate_pixel_runtime_export_quality,
    evaluate_svg_export_quality,
)
from .quality import _global_ssim_np, _image_ssim, compute_quality_metrics  # noqa: F401
from .reporting import save_side_by_side_html  # noqa: F401
from .roundtrip import validate_export_roundtrip  # noqa: F401
from .storage import (  # noqa: F401
    atomic_output_path,
    atomic_write_text,
    load_png,
    load_splats_json,
    render_splats_preview_png,
    save_linear_rgb_png,
    save_splats_json,
)

__all__ = [
    "_evaluate_browser_export_quality",
    "_global_ssim_np",
    "_image_ssim",
    "atomic_output_path",
    "atomic_write_text",
    "compute_quality_metrics",
    "evaluate_css_export_quality",
    "evaluate_native_canvas_export_quality",
    "evaluate_pixel_runtime_export_quality",
    "evaluate_svg_export_quality",
    "load_png",
    "load_splats_json",
    "render_splats_preview_png",
    "save_linear_rgb_png",
    "save_side_by_side_html",
    "save_splats_json",
    "validate_export_roundtrip",
]
