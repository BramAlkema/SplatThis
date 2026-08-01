"""Target-aware 2D Gaussian splat conversion and browser runtimes."""

from ._version import __version__
from .artifact_evaluation import (
    evaluate_css_export_quality,
    evaluate_native_canvas_export_quality,
    evaluate_svg_export_quality,
)
from .browser_export import generate_css_splat_html, generate_native_canvas_html
from .config import ConversionRequest, ConverterConfig
from .converter import PNG2SVGConverter
from .domain import EvidenceLevel, SplatScene
from .expectations import CompositorFidelity, compositor_fidelity
from .pixel_runtime import (
    generate_parallax_pixel_runtime_html,
    generate_pixel_runtime_html,
    generate_webgl_pixel_runtime_html,
)
from .pptx_export import save_pptx_with_splat_png, save_pptx_with_splats
from .reporting import save_side_by_side_html
from .roundtrip import validate_export_roundtrip
from .splat import GaussianSplat, RawSplat
from .storage import load_png, load_splats_json, render_splats_preview_png
from .svg_export import save_svg

__all__ = [
    "__version__",
    "GaussianSplat",
    "RawSplat",
    "SplatScene",
    "EvidenceLevel",
    "CompositorFidelity",
    "compositor_fidelity",
    "ConversionRequest",
    "ConverterConfig",
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
