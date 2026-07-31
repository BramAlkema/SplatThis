"""Proxy + deployed-artifact evaluation (ADR-003).

Measure the right image: proxy metrics from the numpy forward model reject
cheap losers; deployed metrics from the emitted, actually-rasterized SVG
decide acceptance. A `max`-fidelity gain can never be declared from a
proxy fallback — the gate rejects any render_method starting with "proxy".
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

import numpy as np

from ..splat import GaussianSplat
from .analysis import ResidualAnalysis, analyze_residual, select_fixed_rois
from .config import FidelityConfig
from .metrics import FidelityMetrics, compute_fidelity_metrics
from .stage import FidelityCandidate

logger = logging.getLogger(__name__)

EmitSvgFn = Callable[[List[GaussianSplat]], str]


class FidelityEvaluator:
    """Evaluates candidates against the deployed SVG artifact."""

    def __init__(
        self,
        *,
        target_linear_rgb: np.ndarray,
        background_linear_rgb: Optional[np.ndarray],
        compositing_space: str,
        emit_svg: EmitSvgFn,
        work_dir: str,
        config: FidelityConfig,
        saliency_mask: Optional[np.ndarray] = None,
        keep_candidate_artifacts: bool = False,
    ):
        self.target = np.clip(
            np.asarray(target_linear_rgb, dtype=np.float32)[..., :3], 0.0, 1.0
        )
        self.background = background_linear_rgb
        self.compositing_space = compositing_space
        self.emit_svg = emit_svg
        self.work_dir = work_dir
        self.config = config
        self.saliency_mask = saliency_mask
        self.keep_candidate_artifacts = keep_candidate_artifacts
        os.makedirs(work_dir, exist_ok=True)

        self._fixed_rois = None  # selected from the baseline, then frozen
        self._baseline_proxy_ssim: Optional[float] = None
        self._eval_index = 0

    # -- proxies ---------------------------------------------------------

    def _proxy_render(self, splats: List[GaussianSplat]) -> np.ndarray:
        from ..renderer import render_splats_numpy

        h, w = self.target.shape[:2]
        return render_splats_numpy(
            list(splats),
            width=w,
            height=h,
            background_linear_rgb=self.background,
            compositing_space=self.compositing_space,
        )

    def _deployed_render(
        self, splats: List[GaussianSplat], label: str
    ) -> tuple[Optional[np.ndarray], str, int]:
        from ..browser_capture import render_svg_in_browser_to_linear_rgb

        h, w = self.target.shape[:2]
        svg_content = self.emit_svg(list(splats))
        self._eval_index += 1
        svg_path = os.path.join(
            self.work_dir, f"candidate-{self._eval_index:04d}-{_safe(label)}.svg"
        )
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        size = os.path.getsize(svg_path)
        try:
            rendered, method = render_svg_in_browser_to_linear_rgb(svg_path, w, h)
        except RuntimeError as exc:
            return None, f"proxy-fallback:{exc}", size
        if not self.keep_candidate_artifacts and label != "baseline":
            try:
                os.remove(svg_path)
            except OSError:
                pass
        return rendered, method, size

    # -- public API ------------------------------------------------------

    def evaluate(
        self,
        candidate: FidelityCandidate,
        *,
        label: str,
        baseline: Optional[FidelityMetrics] = None,
    ) -> FidelityMetrics:
        splats = list(candidate.splats)
        proxy = self._proxy_render(splats)

        # Fixed ROIs are selected once, from the baseline proxy residual,
        # and stay frozen for every subsequent comparison.
        if self._fixed_rois is None:
            analysis = analyze_residual(
                self.target,
                proxy,
                saliency=self.saliency_mask,
                roi_size=self.config.roi_size,
                roi_count=self.config.roi_count,
            )
            self._fixed_rois = analysis.fixed_rois

        # Cheap fast-reject before paying emit + rasterize + LPIPS.
        from ..io import _image_ssim
        from .metrics import _np_linear_to_srgb

        proxy_ssim = float(
            _image_ssim(_np_linear_to_srgb(proxy), _np_linear_to_srgb(self.target))
        )
        if self._baseline_proxy_ssim is None:
            self._baseline_proxy_ssim = proxy_ssim
        elif (
            proxy_ssim
            < self._baseline_proxy_ssim
            - self.config.proxy_reject_ssim_factor * self.config.max_ssim_regression
        ):
            return compute_fidelity_metrics(
                self.target,
                proxy,
                fixed_rois=self._fixed_rois,
                saliency_mask=self.saliency_mask,
                splat_count=len(splats),
                file_size_bytes=0,
                render_method="proxy-rejected",
            )

        rendered, method, size = self._deployed_render(splats, label)
        if rendered is None:
            # Browser capture unavailable: record honestly; the gate refuses
            # to accept a candidate or claim a deployed-artifact gain.
            rendered = proxy
        return compute_fidelity_metrics(
            self.target,
            rendered,
            fixed_rois=self._fixed_rois,
            saliency_mask=self.saliency_mask,
            splat_count=len(splats),
            file_size_bytes=size,
            render_method=method,
        )

    def analyze(self, candidate: FidelityCandidate) -> ResidualAnalysis:
        """Residual analysis against the deployed render when possible."""
        splats = list(candidate.splats)
        rendered, method, _ = self._deployed_render(splats, "analysis")
        if rendered is None:
            rendered = self._proxy_render(splats)
        if self._fixed_rois is None:
            self._fixed_rois = select_fixed_rois(
                np.abs(self.target - rendered).sum(axis=-1),
                self.saliency_mask,
                size=self.config.roi_size,
                count=self.config.roi_count,
            )
        return analyze_residual(
            self.target,
            rendered,
            saliency=self.saliency_mask,
            fixed_rois=self._fixed_rois,
        )


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:60]
