"""Fidelity metric vector (ADR-003): no single metric is authoritative."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from ..color import linear_to_srgb_float32 as _np_linear_to_srgb

logger = logging.getLogger(__name__)

Roi = Tuple[int, int, int, int]  # (y0, x0, y1, x1)

_LPIPS_NET = None
_LPIPS_FAILED = False


def linear_rgb_to_oklab_np(rgb: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Numpy port of the torch/MLX linear-RGB -> OKLab transform."""
    rgb = np.clip(rgb, 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lms_l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    lms_m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    lms_s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(np.maximum(lms_l, 1e-8))
    m_ = np.cbrt(np.maximum(lms_m, 1e-8))
    s_ = np.cbrt(np.maximum(lms_s, 1e-8))
    lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([lab_l, lab_a, lab_b], axis=-1).astype(np.float32)


def _luma(srgb: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return (
        0.2126 * srgb[..., 0] + 0.7152 * srgb[..., 1] + 0.0722 * srgb[..., 2]
    ).astype(np.float32)


def _lpips_score(a_srgb: npt.NDArray[Any], b_srgb: npt.NDArray[Any]) -> float:
    """LPIPS (AlexNet) or NaN when the optional dependency is unavailable.

    NaN degrades gracefully in the acceptance gate: NaN comparisons are
    False, so gains fall back to the delta-E / salient terms.
    """
    global _LPIPS_NET, _LPIPS_FAILED
    if _LPIPS_FAILED:
        return float("nan")
    try:
        import lpips  # type: ignore
        import torch
    except Exception:
        _LPIPS_FAILED = True
        logger.warning(
            "lpips is not installed; fidelity metrics degrade to the "
            "delta-E/SSIM/edge vector (install `lpips` for the primary "
            "perceptual measure)."
        )
        return float("nan")
    if _LPIPS_NET is None:
        _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False)

    # AlexNet's pooling stack needs a minimum spatial extent; upsample tiny
    # inputs (small test images, small salient crops) to a 64px min side.
    h, w = a_srgb.shape[:2]
    if min(h, w) < 64:
        from skimage.transform import resize

        scale = 64.0 / max(min(h, w), 1)
        shape = (max(int(round(h * scale)), 64), max(int(round(w * scale)), 64))
        a_srgb = resize(a_srgb, shape, anti_aliasing=False, preserve_range=True)
        b_srgb = resize(b_srgb, shape, anti_aliasing=False, preserve_range=True)

    def prep(x: npt.NDArray[Any]):
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
        return t.float() * 2.0 - 1.0

    with torch.no_grad():
        return float(_LPIPS_NET(prep(a_srgb), prep(b_srgb)).item())


def _windowed_ssim(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    from skimage.metrics import structural_similarity

    side = min(x.shape[0], x.shape[1])
    win = min(7, side if side % 2 == 1 else side - 1)
    if win < 3:
        return 1.0 if np.allclose(x, y, atol=1e-4) else 0.0
    return float(structural_similarity(x, y, data_range=1.0, win_size=win))


def _ms_ssim_luma(
    target_srgb: npt.NDArray[Any], rendered_srgb: npt.NDArray[Any]
) -> float:
    """Windowed SSIM on display luma averaged over scales 1/2/4.

    A cheap local structural term: windowed (not global) so local blur and
    displacement register, multi-scale so both fine and mid structure count.
    """
    from skimage.transform import resize

    t = _luma(target_srgb)
    r = _luma(rendered_srgb)
    scores = []
    for scale in (1, 2, 4):
        if scale == 1:
            ts, rs = t, r
        else:
            shape = (max(t.shape[0] // scale, 8), max(t.shape[1] // scale, 8))
            ts = resize(t, shape, anti_aliasing=True, preserve_range=True)
            rs = resize(r, shape, anti_aliasing=True, preserve_range=True)
        scores.append(_windowed_ssim(ts.astype(np.float32), rs.astype(np.float32)))
    return float(np.mean(scores))


def _edge_maps(target_srgb: npt.NDArray[Any], rendered_srgb: npt.NDArray[Any]):
    from skimage.feature import canny

    t_edges = canny(_luma(target_srgb), sigma=1.5)
    r_edges = canny(_luma(rendered_srgb), sigma=1.5)
    return t_edges, r_edges


def _edge_chamfer(t_edges: npt.NDArray[Any], r_edges: npt.NDArray[Any]) -> float:
    """Symmetric mean chamfer distance between edge sets, in pixels."""
    from scipy import ndimage

    if not t_edges.any() and not r_edges.any():
        return 0.0
    diag = float(np.hypot(*t_edges.shape))
    if not t_edges.any() or not r_edges.any():
        return diag
    dist_to_t = ndimage.distance_transform_edt(~t_edges)
    dist_to_r = ndimage.distance_transform_edt(~r_edges)
    return float(0.5 * (dist_to_t[r_edges].mean() + dist_to_r[t_edges].mean()))


def _edge_gradient_l1(
    target_srgb: npt.NDArray[Any], rendered_srgb: npt.NDArray[Any]
) -> float:
    t, r = _luma(target_srgb), _luma(rendered_srgb)
    gt_y, gt_x = np.gradient(t)
    gr_y, gr_x = np.gradient(r)
    return float(np.mean(np.abs(gt_x - gr_x)) + np.mean(np.abs(gt_y - gr_y)))


def _salient_crop(
    mask: Optional[npt.NDArray[Any]], shape: Tuple[int, int], min_size: int = 64
) -> Optional[Roi]:
    """Bounding box of the saliency mask, padded to a workable minimum size."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = shape
    if y1 - y0 < min_size:
        pad = (min_size - (y1 - y0) + 1) // 2
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    if x1 - x0 < min_size:
        pad = (min_size - (x1 - x0) + 1) // 2
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
    return (y0, x0, y1, x1)


@dataclass(frozen=True)
class FidelityMetrics:
    """The guarded metric vector from ADR-003."""

    lpips: float  # lower is better (NaN when lpips unavailable)
    psnr_srgb: float  # higher is better
    ssim_srgb: float  # higher is better
    ms_ssim_luma: float  # higher is better
    delta_e_ok_mean: float  # lower is better (native OKLab distance)
    delta_e_ok_p95: float  # lower is better
    edge_chamfer: float  # lower is better (pixels)
    edge_gradient_l1: float  # lower is better
    salient_lpips: float  # lower is better (NaN without mask or lpips)
    worst_roi_error: float  # lower is better (mean OKLab dE in worst ROI)
    splat_count: int
    file_size_bytes: int
    render_method: str

    def as_dict(self) -> Dict[str, Union[float, int, str]]:
        return dict(self.__dict__)


def compute_fidelity_metrics(
    target_linear_rgb: npt.NDArray[Any],
    rendered_linear_rgb: npt.NDArray[Any],
    *,
    fixed_rois: Sequence[Roi] = (),
    saliency_mask: Optional[npt.NDArray[Any]] = None,
    splat_count: int = 0,
    file_size_bytes: int = 0,
    render_method: str = "unknown",
) -> FidelityMetrics:
    target = np.clip(np.asarray(target_linear_rgb, dtype=np.float32)[..., :3], 0, 1)
    rendered = np.clip(np.asarray(rendered_linear_rgb, dtype=np.float32)[..., :3], 0, 1)
    if target.shape != rendered.shape:
        raise ValueError(
            f"shape mismatch: target={target.shape} rendered={rendered.shape}"
        )

    target_srgb = _np_linear_to_srgb(target)
    rendered_srgb = _np_linear_to_srgb(rendered)

    mse_srgb = float(np.mean((target_srgb - rendered_srgb) ** 2))
    psnr_srgb = float(-10.0 * np.log10(max(mse_srgb, 1e-12)))

    from ..quality import _image_ssim

    ssim_srgb = float(_image_ssim(rendered_srgb, target_srgb))

    lab_t = linear_rgb_to_oklab_np(target)
    lab_r = linear_rgb_to_oklab_np(rendered)
    delta_e = np.sqrt(np.sum((lab_t - lab_r) ** 2, axis=-1))

    t_edges, r_edges = _edge_maps(target_srgb, rendered_srgb)

    salient = float("nan")
    crop = _salient_crop(saliency_mask, target.shape[:2])
    if crop is not None:
        y0, x0, y1, x1 = crop
        salient = _lpips_score(target_srgb[y0:y1, x0:x1], rendered_srgb[y0:y1, x0:x1])

    worst_roi = 0.0
    for y0, x0, y1, x1 in fixed_rois:
        worst_roi = max(worst_roi, float(delta_e[y0:y1, x0:x1].mean()))

    return FidelityMetrics(
        lpips=_lpips_score(target_srgb, rendered_srgb),
        psnr_srgb=psnr_srgb,
        ssim_srgb=ssim_srgb,
        ms_ssim_luma=_ms_ssim_luma(target_srgb, rendered_srgb),
        delta_e_ok_mean=float(delta_e.mean()),
        delta_e_ok_p95=float(np.percentile(delta_e, 95)),
        edge_chamfer=_edge_chamfer(t_edges, r_edges),
        edge_gradient_l1=_edge_gradient_l1(target_srgb, rendered_srgb),
        salient_lpips=salient,
        worst_roi_error=worst_roi,
        splat_count=int(splat_count),
        file_size_bytes=int(file_size_bytes),
        render_method=str(render_method),
    )
