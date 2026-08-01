"""Shared constants and geometry helpers for vector export backends."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .splat import SPLAT_LAYER_NAMES, GaussianSplat, render_order_key

EMU_PER_PX = 9525
# Favor fidelity over blanket coverage in export geometry.
ELLIPSE_OVERLAP_BOOST = 1.15
MIN_ELLIPSE_RADIUS_PX = 0.35
# Upper bound on radial-gradient stops used to approximate each splat's
# Gaussian opacity falloff in exported SVG. Per-splat the count is chosen
# adaptively (see _adaptive_gradient_stops) so that the piecewise-linear
# stop interpolation stays within SVG_GRADIENT_STOP_MAX_ERROR of the true
# Gaussian curve. Low-alpha splats often need just 2; mid-alpha 3-4; only
# high-alpha sharp curves want the full 8.
SVG_GRADIENT_STOPS = 8
SVG_GRADIENT_STOPS_HIGH = 9
SVG_GRADIENT_STOPS_MIN = 2
# Max absolute opacity error (0..1) tolerated between the true Gaussian
# curve and the linear interpolation between adjacent gradient stops.
# This is the per-export *baseline* used when callers don't compute a
# density-aware value via `_density_aware_stop_error(splat_count)`.
#
# Empirically (May 2026): sparse runs (~1800 splats / 1Mpx, large splats,
# few overlaps per pixel) tolerate 0.05 -- the piecewise-linear stop
# interpolation under sRGB compositing happens to land slightly closer to
# source than the "more accurate" many-stop Gaussian does. Dense runs
# (~3900 splats / 1Mpx, small splats, deep overlap stacks) need ~0.02
# because per-splat 2-stop linear ramps stack visibly as "unsmoothed"
# artifacts. The density-aware helper interpolates between them so users
# don't have to think about it. See tmp/stops_sweep_visual.html and
# tmp/forced_4000_thresholds_visual.html for the data.
SVG_GRADIENT_STOP_MAX_ERROR = 0.05
SVG_GRADIENT_STOP_HIGH_MAX_ERROR = 0.005
SVG_GRADIENT_QUALITY_STANDARD = "standard"
SVG_GRADIENT_QUALITY_HIGH = "high"
SVG_PAINTER_ORDER_BACK_TO_FRONT = "back-to-front"
SVG_PAINTER_ORDER_LEGACY = "legacy"


def _density_aware_stop_error(
    splat_count: int,
    *,
    baseline: float = SVG_GRADIENT_STOP_MAX_ERROR,
    floor: float = 0.01,
    ceiling: float = 0.05,
) -> float:
    """Pick the adaptive-stop error threshold for an export.

    The two empirical fit points are 1862 splats -> 0.05 and 3905 splats ->
    0.02, both at ~1Mpx canvas. `threshold ~ 100 / N` interpolates them
    cleanly: at N=1862 it gives 0.054 (clamped to ceiling 0.05), at N=3905
    it gives 0.026. Floor at 0.01 prevents runaway stop counts on
    pathological dense scenes.
    """

    if splat_count <= 0:
        return float(baseline)
    raw = 100.0 / float(splat_count)
    return float(np.clip(raw, floor, ceiling))


def _normalize_svg_gradient_quality(value: str) -> str:
    """Normalize the SVG gradient-fidelity policy without adding a recipe."""

    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"", "standard", "balanced", "default"}:
        return SVG_GRADIENT_QUALITY_STANDARD
    if normalized in {"high", "max", "max-fidelity", "exact"}:
        return SVG_GRADIENT_QUALITY_HIGH
    raise ValueError(f"Unsupported SVG gradient quality: {value}")


def _svg_gradient_settings(
    gradient_quality: str, splat_count: int
) -> Tuple[float, int, int]:
    """Return max error, max stops, and opacity precision for SVG gradients."""

    normalized = _normalize_svg_gradient_quality(gradient_quality)
    density_error = _density_aware_stop_error(splat_count)
    if normalized == SVG_GRADIENT_QUALITY_HIGH:
        return (
            min(density_error, SVG_GRADIENT_STOP_HIGH_MAX_ERROR),
            SVG_GRADIENT_STOPS_HIGH,
            4,
        )
    return density_error, SVG_GRADIENT_STOPS, 2


def _normalize_svg_painter_order(value: str) -> str:
    """Normalize static SVG element-order semantics."""

    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"", "back-to-front", "correct", "corrected", "painter"}:
        return SVG_PAINTER_ORDER_BACK_TO_FRONT
    if normalized in {"legacy", "forward", "front-to-back"}:
        return SVG_PAINTER_ORDER_LEGACY
    raise ValueError(f"Unsupported SVG painter order: {value}")


def _svg_painter_indices(
    splat_count: int,
    painter_order: str = SVG_PAINTER_ORDER_BACK_TO_FRONT,
) -> range:
    """Map front-to-back alpha-over input to back-to-front SVG paint order.

    The mathematical renderers consume lower render-order keys first and let
    those layers claim transmittance. SVG, CSS, and DrawingML instead paint
    later elements on top. SVG emitters must therefore write splat elements in
    reverse order while keeping each element paired with its original paint
    server. Pixel-runtime code must not use this helper.
    """

    normalized = _normalize_svg_painter_order(painter_order)
    if normalized == SVG_PAINTER_ORDER_LEGACY:
        return range(int(splat_count))
    return range(int(splat_count) - 1, -1, -1)


DEFAULT_EXPORT_ORDER = "importance"
DEFAULT_PPTX_SPLAT_STYLE = "gradient"
PPTX_PAINTER_ORDER_LEGACY = "legacy"
PPTX_PAINTER_ORDER_BACK_TO_FRONT = "back-to-front"
PPTX_SOFT_EDGE_ALPHA_SCALE = 0.25
PPTX_SOFT_EDGE_RADIUS_FACTOR = 0.20
PPTX_SOFT_EDGE_K_SIGMA_SCALE = 0.92
PPTX_GRADIENT_ALPHA_SCALE = 0.40
# DrawingML blur calibration: sigma_blur (px) = radius (px) / 3.25, empirical
# erf-fit measurement on macOS desktop PowerPoint. See
# svg2ooxml/docs/reference/research/blur-fidelity-results.md.
PPTX_BLUR_RAD_PER_SIGMA = 3.25
# Core shape extent as fraction of σ. Convolution of uniform-disk(R=k·σ)
# with Gaussian(σ) gives peak α·(1 − e^(−k²/2)) at center; k=1.0 → 0.39α,
# k=1.5 → 0.68α. Larger k means more peak intensity but more top-hat
# character in the result. 1.0 is a balanced compromise.
PPTX_BLUR_CORE_K_SIGMA = 1.0
SVG_BROWSER_COMPAT_RECIPE = "browser-compatible"
SVG_SCRIPTED_MATRIX_RECIPE = "scripted-matrix"
SVG_PALETTE_QUANTIZED_RECIPE = "palette-quantized"
SVG_BLUR_RECIPE = "blur"
# Quantized sigma buckets for shared feGaussianBlur filters.
SVG_BLUR_SIGMA_BUCKETS = 32
# See PPTX_BLUR_CORE_K_SIGMA — same trade-off.
SVG_BLUR_CORE_K_SIGMA = 1.0
# Default palette size when the palette-quantized recipe is selected without an
# explicit override. 128 colors visually covers photographic input without
# bloating definitions; raise via refinement_config['svg_palette_size'] if needed.
SVG_PALETTE_QUANTIZED_DEFAULT_SIZE = 128
SVG_BACKGROUND_ALPHA_CAP = 0.20
SVG_FEATHER_EXTENT = 2.0
SVG_PRECOMP_ALPHA_THRESHOLD = 0.90
SVG_PRECOMP_MAX_SRGB = 160.0


class RegionMasks:
    """Validated region-guidance masks + the shared safe-background test.

    One implementation for every SVG recipe (standard/browser, scripted,
    palette) instead of three drifting nested-closure copies.
    """

    def __init__(
        self,
        width: int,
        height: int,
        foreground: Optional[npt.NDArray[Any]] = None,
        background_safe: Optional[npt.NDArray[Any]] = None,
        edge_band: Optional[npt.NDArray[Any]] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.foreground = self._validated(foreground)
        self.background_safe = self._validated(background_safe)
        self.edge_band = self._validated(edge_band)

    def _validated(
        self, mask: Optional[npt.NDArray[Any]]
    ) -> Optional[npt.NDArray[Any]]:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.shape != (self.height, self.width):
            raise ValueError("SVG region masks must match output height/width")
        return arr.astype(bool, copy=False)

    def splat_center(self, splat: "GaussianSplat") -> Tuple[int, int]:
        x = int(np.clip(round(float(splat.mu[0])), 0, max(self.width - 1, 0)))
        y = int(np.clip(round(float(splat.mu[1])), 0, max(self.height - 1, 0)))
        return x, y

    def in_safe_background(self, splat: "GaussianSplat") -> bool:
        if self.background_safe is None:
            return False
        x, y = self.splat_center(splat)
        if not bool(self.background_safe[y, x]):
            return False
        if self.foreground is not None and bool(self.foreground[y, x]):
            return False
        if self.edge_band is not None and bool(self.edge_band[y, x]):
            return False
        return True


def _normalize_svg_export_recipe(export_recipe: str) -> str:
    normalized = str(export_recipe).strip().lower().replace("_", "-")
    if normalized in {"browser", "browser-compatible"}:
        return SVG_BROWSER_COMPAT_RECIPE
    if normalized in {"scripted", "scripted-standard", "scripted-matrix", "matrix"}:
        return SVG_SCRIPTED_MATRIX_RECIPE
    if normalized in {
        "palette",
        "palette-quantized",
        "quantized",
        "shared",
        "shared-currentcolor",
    }:
        return SVG_PALETTE_QUANTIZED_RECIPE
    if normalized in {"blur", "gaussian-blur", "feblur"}:
        return SVG_BLUR_RECIPE
    if normalized == "standard":
        return "standard"
    raise ValueError(f"Unsupported SVG export recipe: {export_recipe}")


def _layer_name(layer: Optional[int]) -> str:
    if layer is None:
        return "unassigned"
    return SPLAT_LAYER_NAMES.get(int(layer), f"layer-{int(layer)}")


def _layer_title(layer: Optional[int]) -> str:
    return _layer_name(layer).replace("-", " ").title()


def _splat_layer(splat: GaussianSplat) -> Optional[int]:
    return splat.to_raw_splat().layer


def _gaussian_opacity_curve(
    t: npt.NDArray[Any], alpha: float, gradient_footprint: float
) -> npt.NDArray[Any]:
    """The opacity curve sampled by SVG gradient stops.

    Matches the renderer's per-splat alpha-over opacity at a normalized radius
    t in [0, 1] where 1.0 is `gradient_footprint` sigmas from the center.
    """
    return 1.0 - np.exp(
        -float(alpha) * np.exp(-0.5 * (t * float(gradient_footprint)) ** 2)
    )


def _adaptive_gradient_stops(
    alpha: float,
    gradient_footprint: float,
    inner_end: float,
    *,
    min_stops: int = SVG_GRADIENT_STOPS_MIN,
    max_stops: int = SVG_GRADIENT_STOPS,
    max_error: float = SVG_GRADIENT_STOP_MAX_ERROR,
    opacity_precision: int = 2,
) -> List[Tuple[float, float]]:
    """Return (offset, opacity) tuples approximating the Gaussian opacity curve.

    Picks the smallest stop count N in [min_stops, max_stops] whose linear
    interpolation between adjacent stops has max absolute error <= max_error
    against the true curve. Offsets span [0, inner_end].
    """

    min_stops = max(2, int(min_stops))
    max_stops = max(min_stops, int(max_stops))
    opacity_precision = max(0, int(opacity_precision))

    if float(alpha) <= 1e-6 or float(gradient_footprint) <= 0.0:
        # Effectively transparent or degenerate: a flat zero ramp suffices.
        return [(0.0, 0.0), (float(inner_end), 0.0)]

    sample_t = np.linspace(0.0, 1.0, 65)
    true_op = _gaussian_opacity_curve(sample_t, alpha, gradient_footprint)

    for n_stops in range(min_stops, max_stops + 1):
        stop_t = np.linspace(0.0, 1.0, n_stops)
        # Quantize to the precision the SVG emitter writes, so the error check
        # judges exactly the curve a browser will interpolate.
        stop_op = np.round(
            _gaussian_opacity_curve(stop_t, alpha, gradient_footprint),
            opacity_precision,
        )
        interp_op = np.interp(sample_t, stop_t, stop_op)
        if float(np.max(np.abs(interp_op - true_op))) <= float(max_error):
            return [(float(t * inner_end), float(op)) for t, op in zip(stop_t, stop_op)]

    stop_t = np.linspace(0.0, 1.0, max_stops)
    stop_op = np.round(
        _gaussian_opacity_curve(stop_t, alpha, gradient_footprint),
        opacity_precision,
    )
    return [(float(t * inner_end), float(op)) for t, op in zip(stop_t, stop_op)]


def _normalize_pptx_splat_style(splat_style: str) -> str:
    normalized = str(splat_style).strip().lower().replace("_", "-")
    if normalized in {"softedge", "soft-edge", "soft"}:
        return "soft-edge"
    if normalized in {"gradient", "grad"}:
        return "gradient"
    if normalized in {"blur", "gaussian-blur"}:
        return "blur"
    raise ValueError(f"Unsupported PPTX splat style: {splat_style}")


def _normalize_pptx_painter_order(value: str) -> str:
    """Normalize native DrawingML shape-stack semantics."""

    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"", "legacy", "forward", "front-to-back"}:
        return PPTX_PAINTER_ORDER_LEGACY
    if normalized in {"back-to-front", "correct", "corrected", "painter"}:
        return PPTX_PAINTER_ORDER_BACK_TO_FRONT
    raise ValueError(f"Unsupported PPTX painter order: {value}")


def _pptx_painter_splats(
    splats: List[GaussianSplat],
    painter_order: str = PPTX_PAINTER_ORDER_LEGACY,
) -> List[GaussianSplat]:
    """Convert front-to-back renderer input to a DrawingML shape stack."""

    normalized = _normalize_pptx_painter_order(painter_order)
    if normalized == PPTX_PAINTER_ORDER_LEGACY:
        return list(splats)
    return list(reversed(splats))


def _sort_splats_for_export(
    splats: List[GaussianSplat],
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    sort_by_area: bool = False,
) -> List[GaussianSplat]:
    """
    Return an export-ordered splat list.

    `importance` returns the canonical front-to-back alpha-over order used by
    the mathematical renderers. Browser/vector emitters must convert that to
    their back-to-front painter order at the final element-emission boundary.
    """
    if sort_by_area:
        return sorted(splats, key=lambda s: s.area(), reverse=True)

    normalized = str(sort_mode).strip().lower()
    if normalized == "input":
        return list(splats)
    if normalized == "area":
        return sorted(splats, key=lambda s: s.area(), reverse=True)
    if normalized == "importance":
        return sorted(splats, key=render_order_key)
    raise ValueError(f"Unsupported export sort mode: {sort_mode}")


# OOXML requires slide edges to be at least one inch; PowerPoint flags a
# smaller slide-size element as a schema violation and may repair or reject it.
MIN_SLIDE_EMU = 914400


def pptx_emu_scale(width: int, height: int) -> float:
    """Uniform EMU upscale so a small canvas still yields a legal slide size.

    Images below 96 px would otherwise emit `sldSz` under the OOXML minimum.
    The factor is applied to the slide box *and* every shape coordinate, so
    the composition is unchanged — only the deck's physical size grows.
    """
    w = max(int(round(width * EMU_PER_PX)), 1)
    h = max(int(round(height * EMU_PER_PX)), 1)
    return max(1.0, MIN_SLIDE_EMU / w, MIN_SLIDE_EMU / h)


def px_to_emu(value: float, scale: float = 1.0) -> int:
    """Convert pixels to EMU units used by DrawingML.

    Negative values are legitimate: shape offsets (`a:off`) of splats that
    overlap the slide's left/top edge must stay negative, or every border
    splat gets displaced inward while keeping its full extent.

    `scale` is the uniform upscale from `pptx_emu_scale` (1.0 normally).
    """
    return int(round(value * EMU_PER_PX * scale))
