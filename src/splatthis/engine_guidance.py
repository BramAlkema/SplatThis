"""Image analysis, regional guidance and coverage helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .engine_state import ConversionEngineState
from .profiles import get_profile_defaults
from .renderer import torch_linear_rgb_to_oklab
from .splat import GaussianSplat


class ConversionGuidanceMixin(ConversionEngineState):
    """Builds spatial priorities and low-level coverage diagnostics."""

    def _normalize_map(self, values: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if max_v <= min_v + 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_v) / (max_v - min_v)).astype(np.float32)

    def _normalize_percentile_map(
        self,
        values: np.ndarray,
        lower: float = 1.0,
        upper: float = 99.0,
    ) -> np.ndarray:
        """Normalize to [0, 1] after clipping extreme percentiles."""
        arr = np.asarray(values, dtype=np.float32)
        lo = float(np.percentile(arr, np.clip(lower, 0.0, 100.0)))
        hi = float(np.percentile(arr, np.clip(upper, 0.0, 100.0)))
        if hi <= lo + 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _estimate_background_color(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate a stable background color from border pixels in linear RGB.

        This avoids SVG transparency defaulting to white when the optimizer
        implicitly relies on a non-white canvas.
        """
        if image.ndim != 3 or image.shape[2] < 3:
            return np.zeros(3, dtype=np.float32)

        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        height, width = rgb.shape[:2]
        border = max(1, int(round(0.04 * float(min(height, width)))))

        top = rgb[:border, :, :].reshape(-1, 3)
        bottom = rgb[max(height - border, 0) :, :, :].reshape(-1, 3)
        left = rgb[:, :border, :].reshape(-1, 3)
        right = rgb[:, max(width - border, 0) :, :].reshape(-1, 3)
        border_pixels = np.concatenate([top, bottom, left, right], axis=0)

        if image.shape[2] >= 4:
            alpha = np.asarray(image[:, :, 3], dtype=np.float32)
            top_a = alpha[:border, :].reshape(-1)
            bottom_a = alpha[max(height - border, 0) :, :].reshape(-1)
            left_a = alpha[:, :border].reshape(-1)
            right_a = alpha[:, max(width - border, 0) :].reshape(-1)
            border_alpha = np.concatenate([top_a, bottom_a, left_a, right_a], axis=0)
            valid = border_alpha > 0.02
            if np.any(valid):
                border_pixels = border_pixels[valid]

        if border_pixels.size == 0:
            border_pixels = rgb.reshape(-1, 3)
        border_std = float(np.mean(np.std(border_pixels, axis=0)))
        max_uniform_std = float(
            self.refinement_config.get("background_uniformity_std_max", 0.18)
        )
        if border_std > max_uniform_std:
            return np.zeros(3, dtype=np.float32)
        background = np.median(border_pixels, axis=0).astype(np.float32)
        if not np.isfinite(background).all():
            return np.zeros(3, dtype=np.float32)
        return np.clip(background, 0.0, 1.0)

    def _estimate_border_median_color(self, image: np.ndarray) -> np.ndarray:
        """Estimate border median without rejecting non-uniform photo borders."""
        if image.ndim != 3 or image.shape[2] < 3:
            return np.zeros(3, dtype=np.float32)
        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        height, width = rgb.shape[:2]
        border = max(1, int(round(0.04 * float(min(height, width)))))
        border_pixels = np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[max(height - border, 0) :, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, max(width - border, 0) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if image.shape[2] >= 4:
            alpha = np.asarray(image[:, :, 3], dtype=np.float32)
            border_alpha = np.concatenate(
                [
                    alpha[:border, :].reshape(-1),
                    alpha[max(height - border, 0) :, :].reshape(-1),
                    alpha[:, :border].reshape(-1),
                    alpha[:, max(width - border, 0) :].reshape(-1),
                ],
                axis=0,
            )
            valid = border_alpha > 0.02
            if np.any(valid):
                border_pixels = border_pixels[valid]
        if border_pixels.size == 0:
            border_pixels = rgb.reshape(-1, 3)
        background = np.median(border_pixels, axis=0).astype(np.float32)
        if not np.isfinite(background).all():
            return np.zeros(3, dtype=np.float32)
        return np.clip(background, 0.0, 1.0)

    def _compute_background_suppressed_priority(
        self,
        lightness: np.ndarray,
        saliency: np.ndarray,
        foreground: np.ndarray,
        edge_strength: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build a pure image-statistics detail prior that discounts border background texture."""
        from scipy import ndimage as ndi
        from skimage.filters import gaussian, sobel
        from skimage.morphology import (
            binary_closing,
            binary_dilation,
            binary_opening,
            disk,
            remove_small_objects,
        )

        light = np.asarray(lightness, dtype=np.float32)
        sal = np.asarray(saliency, dtype=np.float32)
        fg = np.asarray(foreground, dtype=bool)
        edge = np.asarray(edge_strength, dtype=np.float32)
        height, width = sal.shape
        total_pixels = max(1, int(height * width))

        grad = self._normalize_percentile_map(sobel(light), lower=1.0, upper=99.0)
        lap = np.abs(ndi.laplace(gaussian(light, sigma=0.6))).astype(np.float32)
        lap_local = self._normalize_percentile_map(
            gaussian(lap, sigma=1.2), lower=1.0, upper=99.5
        )
        focus = self._normalize_map(0.55 * lap_local + 0.45 * grad)
        focus = self._normalize_map(gaussian(focus, sigma=2.0))

        low_saliency_pct = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_low_percentile", 68.0
                ),
                0.0,
                100.0,
            )
        )
        low_focus_pct = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_focus_low_percentile", 62.0
                ),
                0.0,
                100.0,
            )
        )
        low_saliency = sal < float(np.percentile(sal, low_saliency_pct))
        low_focus = focus < float(np.percentile(focus, low_focus_pct))
        background_corridor = np.asarray(low_saliency | low_focus, dtype=bool)

        border_seed = np.zeros((height, width), dtype=bool)
        border = max(1, int(round(0.035 * float(min(height, width)))))
        border_seed[:border, :] = True
        border_seed[max(height - border, 0) :, :] = True
        border_seed[:, :border] = True
        border_seed[:, max(width - border, 0) :] = True
        border_background = ndi.binary_propagation(
            border_seed, mask=background_corridor
        )
        border_background = binary_dilation(
            np.asarray(border_background, dtype=bool), disk(3)
        )
        background_penalty = self._normalize_map(
            gaussian(border_background.astype(np.float32), sigma=5.0)
        )

        saliency_cut = float(
            np.percentile(
                sal,
                np.clip(
                    self.refinement_config.get(
                        "background_suppressed_subject_saliency_percentile", 72.0
                    ),
                    0.0,
                    100.0,
                ),
            )
        )
        edge_cut = float(
            np.percentile(
                edge,
                np.clip(
                    self.refinement_config.get(
                        "background_suppressed_subject_edge_percentile", 80.0
                    ),
                    0.0,
                    100.0,
                ),
            )
        )
        focus_cut = float(
            np.percentile(
                focus,
                np.clip(
                    self.refinement_config.get(
                        "background_suppressed_subject_focus_percentile", 55.0
                    ),
                    0.0,
                    100.0,
                ),
            )
        )
        candidate = np.asarray(
            (sal >= saliency_cut) | ((edge >= edge_cut) & (focus >= focus_cut)) | fg,
            dtype=bool,
        )
        candidate = binary_closing(candidate, disk(2))
        candidate = binary_opening(candidate, disk(1))
        candidate = remove_small_objects(
            candidate, min_size=max(8, int(total_pixels * 0.0025))
        )
        candidate = np.asarray(candidate, dtype=bool)

        labels, label_count = ndi.label(candidate)
        keep = np.zeros((height, width), dtype=bool)
        min_component_area = max(24, int(total_pixels * 0.0025))
        border_keep_score = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_border_component_keep_score", 0.42
                ),
                0.0,
                1.0,
            )
        )
        component_score_map = self._normalize_map(
            0.45 * sal + 0.35 * focus + 0.20 * edge
        )
        for label_id in range(1, int(label_count) + 1):
            component = labels == label_id
            area = int(np.count_nonzero(component))
            if area < min_component_area:
                continue
            touches_border = bool(
                np.any(component[:border, :])
                or np.any(component[max(height - border, 0) :, :])
                or np.any(component[:, :border])
                or np.any(component[:, max(width - border, 0) :])
            )
            component_score = (
                float(np.mean(component_score_map[component])) if area else 0.0
            )
            if touches_border and component_score < border_keep_score:
                continue
            keep[component] = True

        if not np.any(keep) and np.any(fg):
            keep = np.asarray(fg, dtype=bool)

        center_score = self._normalize_map(gaussian(keep.astype(np.float32), sigma=8.0))
        subject_prior = self._normalize_map(
            0.55 * keep.astype(np.float32)
            + 0.30 * center_score
            + 0.15 * fg.astype(np.float32)
        )
        if float(np.max(subject_prior) - np.min(subject_prior)) <= 1e-8:
            subject_prior = self._normalize_map(
                0.65 * fg.astype(np.float32) + 0.35 * sal
            )

        penalty_strength = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_penalty_strength", 0.65
                ),
                0.0,
                1.0,
            )
        )
        edge_gate = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_edge_gate", 0.80
                ),
                0.0,
                1.0,
            )
        )
        focus_gate = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_focus_gate", 0.80
                ),
                0.0,
                1.0,
            )
        )
        background_keep = np.clip(
            1.0 - penalty_strength * background_penalty, 0.0, 1.0
        ).astype(np.float32)
        gated_edges = self._normalize_map(
            edge
            * background_keep
            * (1.0 - edge_gate + edge_gate * subject_prior)
            * (1.0 - focus_gate + focus_gate * focus)
        )
        gated_saliency = self._normalize_map(
            (0.40 * sal + 0.35 * focus + 0.25 * subject_prior) * background_keep
        )
        detail_priority = self._normalize_map(
            (0.45 * gated_saliency + 0.55 * gated_edges) * (0.35 + 0.65 * subject_prior)
        )
        if float(np.max(detail_priority) - np.min(detail_priority)) <= 1e-8:
            detail_priority = self._normalize_map(
                (0.65 * sal + 0.35 * edge) * background_keep
            )

        return {
            "focus_map": focus.astype(np.float32),
            "background_penalty_map": background_penalty.astype(np.float32),
            "subject_mask": keep,
            "subject_prior_map": subject_prior.astype(np.float32),
            "gated_edge_map": gated_edges.astype(np.float32),
            "gated_saliency_map": gated_saliency.astype(np.float32),
            "detail_priority_map": detail_priority.astype(np.float32),
        }

    def _compute_region_guidance(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Build foreground/background/edge masks plus a spatial loss weight map.

        Edges and salient foreground stay at full weight; safe background gets
        a lower weight so flat regions do not dominate optimization or sampling.
        """
        from skimage.feature import canny
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import (
            binary_closing,
            binary_erosion,
            binary_opening,
            disk,
            remove_small_objects,
        )

        if image.ndim != 3 or image.shape[2] < 3:
            height, width = image.shape[:2]
            ones = np.ones((height, width), dtype=np.float32)
            zeros = np.zeros((height, width), dtype=bool)
            zero_map = np.zeros((height, width), dtype=np.float32)
            return {
                "weight_map": ones,
                "saliency_map": zero_map,
                "detail_priority_map": zero_map,
                "focus_map": zero_map,
                "background_penalty_map": zero_map,
                "gated_edge_map": zero_map,
                "gated_saliency_map": zero_map,
                "subject_prior_map": zero_map,
                "subject_mask": zeros,
                "foreground_mask": zeros,
                "background_safe_mask": zeros,
                "edge_band_mask": zeros,
                "background_linear_rgb": np.zeros(3, dtype=np.float32),
                "summary": {
                    "total_pixels": int(height * width),
                    "foreground_pixels": 0,
                    "background_safe_pixels": 0,
                    "edge_band_pixels": 0,
                    "foreground_ratio": 0.0,
                    "background_safe_ratio": 0.0,
                    "edge_band_ratio": 0.0,
                    "weight_min": 1.0,
                    "weight_max": 1.0,
                    "detail_priority_mean": 0.0,
                    "background_penalty_mean": 0.0,
                    "focus_mean": 0.0,
                    "subject_mask_ratio": 0.0,
                },
            }

        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        height, width = rgb.shape[:2]
        background = self._estimate_border_median_color(image)

        with torch.no_grad():
            lightness = torch_linear_rgb_to_oklab(torch.from_numpy(rgb)).numpy()[
                :, :, 0
            ]

        edge_binary = canny(lightness, sigma=1.4)
        edge_strength = self._normalize_map(
            gaussian(edge_binary.astype(np.float32), sigma=2.0)
        )

        color_distance = self._normalize_map(
            np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=-1)
        )
        dog = self._normalize_map(
            np.abs(gaussian(lightness, sigma=1.0) - gaussian(lightness, sigma=8.0))
        )
        edge_density = self._normalize_map(
            gaussian(edge_binary.astype(np.float32), sigma=6.0)
        )
        saliency = self._normalize_map(
            0.45 * color_distance + 0.35 * dog + 0.20 * edge_density
        )

        if float(np.max(saliency) - np.min(saliency)) <= 1e-8:
            foreground = np.zeros((height, width), dtype=bool)
        else:
            foreground = saliency > float(threshold_otsu(saliency))
            foreground = binary_closing(foreground, disk(3))
            foreground = binary_opening(foreground, disk(2))
            foreground = remove_small_objects(
                foreground,
                min_size=max(1, int(width * height * 0.005)),
            )
            foreground = np.asarray(foreground, dtype=bool)

        background_safe = np.asarray(binary_erosion(~foreground, disk(4)), dtype=bool)
        edge_band = np.asarray(edge_strength > 0.05, dtype=bool)

        base_weight = float(self.refinement_config.get("region_weight_base", 0.70))
        background_weight = float(
            self.refinement_config.get("region_weight_background", 0.25)
        )
        foreground_weight = float(
            self.refinement_config.get("region_weight_foreground", 1.00)
        )
        edge_weight = float(self.refinement_config.get("region_weight_edge", 1.00))
        saliency_boost = float(
            max(0.0, self.refinement_config.get("region_weight_saliency_boost", 0.55))
        )
        saliency_gamma = float(
            max(0.25, self.refinement_config.get("region_weight_saliency_gamma", 0.80))
        )
        weights = np.full(
            (height, width), np.clip(base_weight, 0.0, 10.0), dtype=np.float32
        )
        weights[background_safe & ~edge_band] = np.clip(background_weight, 0.0, 10.0)
        weights[foreground] = np.clip(foreground_weight, 0.0, 10.0)
        weights[edge_band] = np.clip(edge_weight, 0.0, 10.0)
        if saliency_boost > 0.0:
            saliency_prior = np.power(
                np.clip(saliency, 0.0, 1.0), saliency_gamma
            ).astype(np.float32)
            weights = weights * (1.0 + saliency_boost * saliency_prior)

        if bool(
            self.refinement_config.get("background_suppressed_saliency_enabled", False)
        ):
            suppressed = self._compute_background_suppressed_priority(
                lightness=lightness,
                saliency=saliency,
                foreground=foreground,
                edge_strength=edge_strength,
            )
            if bool(
                self.refinement_config.get(
                    "background_suppressed_saliency_use_for_weights", False
                )
            ):
                detail_priority = suppressed["detail_priority_map"]
                background_penalty = suppressed["background_penalty_map"]
                weights = weights * (0.35 + 0.65 * detail_priority)
                weights = weights * np.clip(1.0 - 0.35 * background_penalty, 0.0, 1.0)
                weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        else:
            suppressed = {
                "detail_priority_map": saliency.astype(np.float32),
                "focus_map": np.zeros((height, width), dtype=np.float32),
                "background_penalty_map": np.zeros((height, width), dtype=np.float32),
                "gated_edge_map": edge_strength.astype(np.float32),
                "gated_saliency_map": saliency.astype(np.float32),
                "subject_prior_map": foreground.astype(np.float32),
                "subject_mask": foreground,
            }

        total_pixels = max(1, int(width * height))
        foreground_pixels = int(np.count_nonzero(foreground))
        background_safe_pixels = int(np.count_nonzero(background_safe))
        edge_band_pixels = int(np.count_nonzero(edge_band))
        subject_mask = np.asarray(suppressed["subject_mask"], dtype=bool)
        detail_priority_map = np.asarray(
            suppressed["detail_priority_map"], dtype=np.float32
        )
        background_penalty_map = np.asarray(
            suppressed["background_penalty_map"], dtype=np.float32
        )
        focus_map = np.asarray(suppressed["focus_map"], dtype=np.float32)

        return {
            "weight_map": weights.astype(np.float32),
            "saliency_map": saliency.astype(np.float32),
            "detail_priority_map": detail_priority_map,
            "focus_map": focus_map,
            "background_penalty_map": background_penalty_map,
            "gated_edge_map": np.asarray(
                suppressed["gated_edge_map"], dtype=np.float32
            ),
            "gated_saliency_map": np.asarray(
                suppressed["gated_saliency_map"], dtype=np.float32
            ),
            "subject_prior_map": np.asarray(
                suppressed["subject_prior_map"], dtype=np.float32
            ),
            "subject_mask": subject_mask,
            "foreground_mask": foreground,
            "background_safe_mask": background_safe,
            "edge_band_mask": edge_band,
            "background_linear_rgb": background,
            "summary": {
                "total_pixels": int(total_pixels),
                "foreground_pixels": foreground_pixels,
                "background_safe_pixels": background_safe_pixels,
                "edge_band_pixels": edge_band_pixels,
                "foreground_ratio": float(foreground_pixels / total_pixels),
                "background_safe_ratio": float(background_safe_pixels / total_pixels),
                "edge_band_ratio": float(edge_band_pixels / total_pixels),
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                "weight_mean": float(np.mean(weights)),
                "saliency_min": float(np.min(saliency)),
                "saliency_max": float(np.max(saliency)),
                "saliency_mean": float(np.mean(saliency)),
                "saliency_p90": float(np.percentile(saliency, 90)),
                "saliency_p95": float(np.percentile(saliency, 95)),
                "detail_priority_mean": float(np.mean(detail_priority_map)),
                "detail_priority_p90": float(np.percentile(detail_priority_map, 90)),
                "detail_priority_p95": float(np.percentile(detail_priority_map, 95)),
                "background_penalty_mean": float(np.mean(background_penalty_map)),
                "focus_mean": float(np.mean(focus_map)),
                "subject_mask_pixels": int(np.count_nonzero(subject_mask)),
                "subject_mask_ratio": float(
                    np.count_nonzero(subject_mask) / total_pixels
                ),
                "background_linear_rgb": [
                    float(background[0]),
                    float(background[1]),
                    float(background[2]),
                ],
            },
        }

    def _loss_weight_tensor(self, width: int, height: int) -> Optional[torch.Tensor]:
        """Return region weighting as a tensor when enabled and shape-compatible."""
        if not self.region_weighting_enabled or self._region_weight_map is None:
            return None
        if self._region_weight_map.shape != (int(height), int(width)):
            return None
        return torch.from_numpy(
            self._region_weight_map.astype(np.float32, copy=False)
        ).to(self.device)

    def _sample_candidate_positions(
        self,
        score_map: np.ndarray,
        percentile: float,
        max_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample top-scoring coordinates with probability proportional to score."""
        if max_samples <= 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        if (
            self.region_weighting_enabled
            and self._region_weight_map is not None
            and self._region_weight_map.shape == score_map.shape
        ):
            score_map = (
                np.asarray(score_map, dtype=np.float32) * self._region_weight_map
            )

        threshold = float(np.percentile(score_map, percentile))
        mask = score_map >= threshold
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0:
            flat = score_map.reshape(-1)
            if flat.size == 0:
                return (
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32),
                )
            topk = min(max_samples, flat.size)
            top_idx = np.argpartition(flat, -topk)[-topk:]
            y_indices, x_indices = np.unravel_index(top_idx, score_map.shape)

        sample_count = min(int(max_samples), len(x_indices))
        if sample_count <= 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        weights = score_map[y_indices, x_indices].astype(np.float64)
        if float(weights.sum()) > 1e-12:
            weights = weights / float(weights.sum())
        else:
            weights = None

        selected = rng.choice(
            len(x_indices), size=sample_count, replace=False, p=weights
        )
        selected_x = x_indices[selected].astype(np.int32)
        selected_y = y_indices[selected].astype(np.int32)
        selected_scores = score_map[selected_y, selected_x].astype(np.float32)
        return selected_x, selected_y, selected_scores

    def _build_alpha_coverage_map(
        self, splats: List[GaussianSplat], width: int, height: int
    ) -> np.ndarray:
        """Build alpha coverage map where 1 means fully covered by accumulated opacity."""
        transmittance = np.ones((height, width), dtype=np.float32)
        self._apply_splats_to_transmittance(
            transmittance=transmittance,
            splats=splats,
            width=width,
            height=height,
        )
        coverage = 1.0 - transmittance
        return np.clip(coverage, 0.0, 1.0).astype(np.float32)

    def _apply_splats_to_transmittance(
        self,
        transmittance: np.ndarray,
        splats: List[GaussianSplat],
        width: int,
        height: int,
    ) -> None:
        """Apply splat alpha-over attenuation into a transmittance map in place.

        Uses the shared footprint kernel in renderer.py so the coverage math
        cannot drift from the numpy validator.
        """
        from .renderer import iter_splat_footprints

        footprint_sigma = float(
            max(self.refinement_config.get("coverage_footprint_sigma", 3.0), 1.0)
        )
        for _, y0, y1, x0, x1, layer_alpha in iter_splat_footprints(
            splats, width, height, footprint_sigma
        ):
            transmittance[y0:y1, x0:x1] *= np.clip(
                1.0 - layer_alpha.astype(np.float32), 0.0, 1.0
            )

    def _compute_coverage_ratio(self, coverage_map: np.ndarray) -> float:
        """Compute covered-pixel ratio under configured alpha threshold."""
        threshold = float(
            np.clip(self.refinement_config.get("coverage_threshold", 0.03), 0.0, 1.0)
        )
        return float(np.mean(coverage_map >= threshold))

    def _resolve_target_size(self, input_path: str) -> Tuple[int, int]:
        """Resolve effective target size after applying resolution scale."""
        if self.target_size is not None:
            base_w, base_h = self.target_size
        else:
            with Image.open(input_path) as img:
                base_w, base_h = img.size

        scaled_w = max(1, int(round(base_w * self.resolution_scale)))
        scaled_h = max(1, int(round(base_h * self.resolution_scale)))
        return (scaled_w, scaled_h)

    def _get_profile_defaults(self, profile: str) -> Dict[str, Any]:
        """Return tuned defaults for quality profile (see profiles.py)."""
        return get_profile_defaults(profile)

    def _sha256_file(self, path: str) -> str:
        """Compute SHA256 of input file."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
