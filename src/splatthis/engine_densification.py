"""Error-driven densification and residual detail passes."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .features import edge_tangent_angle, estimate_local_color
from .renderer import splats_to_tensor
from .splat import (
    LAYER_DETAIL,
    LAYER_EDGE,
    GaussianSplat,
    create_anisotropic_splat,
    create_isotropic_splat,
)

logger = logging.getLogger(__name__)


class ConversionDensificationMixin:
    """Adds detail splats from residual and saliency evidence."""

    def _add_error_driven_splats(  # noqa: C901
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        target: torch.Tensor,
        renderer: torch.nn.Module,
        rng: np.random.Generator,
        edge_map: Optional[np.ndarray] = None,
        stage_idx: int = 0,
        precomputed_rendered: Optional[torch.Tensor] = None,
        precomputed_coverage_map: Optional[np.ndarray] = None,
        structure_primary: Optional[np.ndarray] = None,
        structure_anisotropy: Optional[np.ndarray] = None,
        max_splats_cap: Optional[int] = None,
    ) -> Tuple[List[GaussianSplat], Optional[np.ndarray]]:
        """Add new splats using residual, uncovered-opacity, and edge cues."""
        cap = int(
            self.max_splats
            if max_splats_cap is None
            else np.clip(max_splats_cap, 0, self.max_splats)
        )
        if len(splats) >= cap:
            return splats, precomputed_coverage_map

        if precomputed_rendered is None:
            splats_tensor = splats_to_tensor(splats, device=self.device)
            with torch.no_grad():
                rendered = renderer(splats_tensor)
        else:
            rendered = precomputed_rendered
        with torch.no_grad():
            residual_map = target - rendered
            error_map = torch.mean(residual_map**2, dim=-1)
        error_np = error_map.cpu().numpy()
        residual_np = residual_map.cpu().numpy()
        error_norm = self._normalize_map(error_np)
        height, width = image.shape[:2]
        if edge_map is None or edge_map.shape != (height, width):
            edge_map = self._build_edge_map(image)

        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (
            height,
            width,
        ):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(
                splats=splats,
                width=width,
                height=height,
            )
        uncovered_map = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32)

        coverage_ratio = self._compute_coverage_ratio(coverage_map)
        target_coverage = float(
            np.clip(self.refinement_config.get("coverage_target", 0.985), 0.0, 1.0)
        )
        coverage_deficit = max(target_coverage - coverage_ratio, 0.0)

        weight_error = float(
            max(self.refinement_config.get("densify_weight_error", 0.50), 0.0)
        )
        weight_uncovered = float(
            max(self.refinement_config.get("densify_weight_uncovered", 0.40), 0.0)
        )
        weight_edge = float(
            max(self.refinement_config.get("densify_weight_edge", 0.10), 0.0)
        )
        weight_sum = max(weight_error + weight_uncovered + weight_edge, 1e-8)
        sampling_map = (
            (weight_error / weight_sum) * error_norm
            + (weight_uncovered / weight_sum) * uncovered_map
            + (weight_edge / weight_sum) * edge_map
        )
        sampling_map = self._apply_saliency_sampling_bias(
            sampling_map,
            strength=float(self.refinement_config.get("densify_weight_saliency", 0.45)),
        )
        sampling_map = np.clip(sampling_map, 0.0, 1.0).astype(np.float32)
        if float(np.sum(sampling_map)) <= 1e-12:
            sampling_map = np.maximum(error_norm, uncovered_map)

        base_percentile = float(
            np.clip(self.refinement_config["densify_percentile"], 0.0, 100.0)
        )
        stage_scale = max(len(self.stages) - stage_idx, 1) / max(len(self.stages), 1)
        adaptive_percentile = float(
            np.clip(base_percentile - 35.0 * coverage_deficit * stage_scale, 45.0, 99.8)
        )

        densify_fraction = float(
            np.clip(self.refinement_config["densify_fraction"], 0.01, 1.0)
        )
        deficit_boost = (
            1.0
            + float(self.refinement_config.get("coverage_densify_boost", 2.0))
            * coverage_deficit
        )
        max_new = min(
            cap - len(splats),
            int(np.ceil(len(splats) * densify_fraction * deficit_boost)),
        )
        if max_new <= 0:
            return splats, coverage_map

        x_indices, y_indices, sample_weights = self._sample_candidate_positions(
            score_map=sampling_map,
            percentile=adaptive_percentile,
            max_samples=max_new,
            rng=rng,
        )
        if len(x_indices) == 0:
            return splats, coverage_map

        new_splats: List[GaussianSplat] = []
        residual_color_gain = float(
            self.refinement_config.get("residual_color_gain", 0.75)
        )
        sigma_minor_min = float(self.refinement_config.get("sigma_minor_min", 0.35))
        sigma_min = float(self.refinement_config.get("sigma_min", 0.45))
        sigma_max = float(self.refinement_config.get("sigma_max", 4.0))
        sigma_scale = float(self.refinement_config.get("sigma_scale", 2.0))
        sigma_fill_max = float(
            max(
                self.refinement_config.get("coverage_sigma_max", sigma_max * 1.8),
                sigma_max,
            )
        )
        for idx, (x, y) in enumerate(zip(x_indices, y_indices)):
            base_color = estimate_local_color(image, x, y)
            residual_rgb = residual_np[y, x, :3].astype(np.float32)
            color = np.clip(
                base_color + residual_color_gain * residual_rgb, 0.0, 1.0
            ).astype(np.float32)
            if not np.isfinite(color).all():
                color = base_color

            detail_need = float(error_norm[y, x])
            fill_need = float(uncovered_map[y, x])
            edge_need = float(edge_map[y, x])

            sigma_detail = float(
                np.clip(sigma_max - sigma_scale * detail_need, sigma_min, sigma_max)
            )
            sigma = float(
                np.clip(
                    (1.0 - fill_need) * sigma_detail + fill_need * sigma_fill_max,
                    sigma_min,
                    sigma_fill_max,
                )
            )
            alpha = float(
                np.clip(
                    self.refinement_config["alpha_base"]
                    + self.refinement_config["alpha_scale"]
                    * (0.55 * detail_need + 0.45 * fill_need),
                    self.refinement_config["alpha_min"],
                    self.refinement_config["alpha_max"],
                )
            )
            x_center = float(np.clip(x + rng.uniform(-0.5, 0.5), 0.0, width - 1.0))
            y_center = float(np.clip(y + rng.uniform(-0.5, 0.5), 0.0, height - 1.0))

            local_structure_edge_threshold = float(
                np.clip(
                    self.refinement_config.get("structure_local_edge_threshold", 0.18),
                    0.0,
                    1.0,
                )
            )
            local_structure_detail_threshold = float(
                np.clip(
                    self.refinement_config.get(
                        "structure_local_detail_threshold", 0.22
                    ),
                    0.0,
                    1.0,
                )
            )
            prefer_local_structure = bool(
                edge_need >= local_structure_edge_threshold
                or detail_need >= local_structure_detail_threshold
            )
            if (
                structure_primary is not None
                and structure_anisotropy is not None
                and structure_primary.shape[:2] == (height, width)
                and structure_primary.shape[-1] == 2
                and structure_anisotropy.shape == (height, width)
                and not prefer_local_structure
            ):
                primary_direction = structure_primary[y, x]
                anisotropy = float(structure_anisotropy[y, x])
            else:
                primary_direction, anisotropy = self._analyze_local_structure(
                    image, x, y
                )
            anisotropy_threshold = float(
                max(
                    1.0,
                    self.refinement_config.get("densify_anisotropy_threshold", 1.30),
                )
            )
            anisotropy_edge_threshold = float(
                np.clip(
                    self.refinement_config.get(
                        "densify_anisotropy_edge_threshold", 0.14
                    ),
                    0.0,
                    1.0,
                )
            )
            strong_edge_threshold = float(
                np.clip(
                    self.refinement_config.get("densify_strong_edge_threshold", 0.38),
                    0.0,
                    1.0,
                )
            )
            make_anisotropic = (
                anisotropy >= anisotropy_threshold
                and edge_need >= anisotropy_edge_threshold
            ) or (
                anisotropy >= max(1.0, anisotropy_threshold - 0.08)
                and edge_need >= strong_edge_threshold
            )
            if make_anisotropic:
                angle = edge_tangent_angle(primary_direction)
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                sigma_major = sigma * (1.0 + 0.5 * fill_need)
                anisotropy_cap = max(
                    1.0,
                    min(
                        float(anisotropy),
                        float(
                            self.refinement_config.get(
                                "local_structure_anisotropy_clip", 4.0
                            )
                        ),
                    ),
                )
                sigma_minor = max(sigma_major / anisotropy_cap, sigma_minor_min)
                if edge_need > 0.5 and fill_need < 0.5:
                    sigma_minor = max(sigma_minor * 0.75, sigma_minor_min)
                new_splat = create_anisotropic_splat(
                    center=np.array([x_center, y_center], dtype=np.float32),
                    eigenvals=np.array(
                        [sigma_major**2, sigma_minor**2], dtype=np.float32
                    ),
                    eigenvecs=rotation_matrix,
                    color=color,
                    alpha=alpha,
                )
            else:
                new_splat = create_isotropic_splat(
                    center=np.array([x_center, y_center], dtype=np.float32),
                    sigma=sigma,
                    color=color,
                    alpha=alpha,
                )
            layer, local_importance = self._saliency_layer_for_pixel(x, y, LAYER_DETAIL)
            local_importance = float(
                max(
                    local_importance,
                    0.20 + 0.79 * float(np.clip(sample_weights[idx], 0.0, 1.0)),
                )
            )
            self._assign_splat_layer(new_splat, layer, local_importance)
            new_splats.append(new_splat)

        logger.info(
            "Added %s splats (coverage %.1f%% -> target %.1f%%)",
            len(new_splats),
            coverage_ratio * 100.0,
            target_coverage * 100.0,
        )
        if not new_splats:
            return splats, coverage_map

        # Incremental coverage update: apply only newly inserted splats to current transmittance.
        transmittance = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(
            np.float32, copy=True
        )
        self._apply_splats_to_transmittance(
            transmittance=transmittance,
            splats=new_splats,
            width=width,
            height=height,
        )
        updated_coverage = np.clip(1.0 - transmittance, 0.0, 1.0).astype(np.float32)
        return splats + new_splats, updated_coverage

    def _run_residual_detail_passes(  # noqa: C901
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: torch.nn.Module,
        rng: np.random.Generator,
        edge_map: np.ndarray,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], List[Dict[str, Any]]]:
        """Run late residual-focused densification with small isotropic splats."""
        if not bool(self.refinement_config.get("residual_detail_enabled", False)):
            return splats, []

        passes = int(max(1, self.refinement_config.get("residual_detail_passes", 1)))
        residual_metrics: List[Dict[str, Any]] = []
        current_splats = splats
        height, width = image.shape[:2]

        for pass_idx in range(passes):
            pass_t0 = time.perf_counter()
            if self._time_budget_exhausted():
                break
            if len(current_splats) >= self.max_splats:
                break

            with torch.no_grad():
                if current_splats:
                    rendered = renderer(
                        splats_to_tensor(current_splats, device=self.device)
                    )
                else:
                    rendered = torch.zeros(
                        (height, width, 3),
                        dtype=target.dtype,
                        device=target.device,
                    )
                residual_map = target - rendered
                error_map = torch.mean(residual_map**2, dim=-1)

            error_norm = self._normalize_map(error_map.cpu().numpy())
            residual_np = residual_map.cpu().numpy()

            edge_weight = float(
                np.clip(
                    self.refinement_config.get("residual_detail_edge_weight", 0.30),
                    0.0,
                    2.0,
                )
            )
            score_map = self._normalize_map(error_norm * (1.0 + edge_weight * edge_map))
            score_map = self._apply_saliency_sampling_bias(
                score_map,
                strength=float(
                    self.refinement_config.get("residual_detail_saliency_weight", 0.55)
                ),
            )
            percentile = float(
                np.clip(
                    self.refinement_config.get("residual_detail_percentile", 90.0),
                    0.0,
                    100.0,
                )
            )
            fraction = float(
                np.clip(
                    self.refinement_config.get("residual_detail_fraction", 0.12),
                    0.01,
                    1.0,
                )
            )
            max_new = min(
                self.max_splats - len(current_splats),
                int(np.ceil(max(1, len(current_splats)) * fraction)),
            )
            if max_new <= 0:
                break

            edge_fraction = float(
                np.clip(
                    self.refinement_config.get("residual_detail_edge_fraction", 0.45),
                    0.0,
                    0.95,
                )
            )
            edge_count = min(max_new, int(round(max_new * edge_fraction)))
            residual_count = max_new - edge_count
            edge_gamma = float(
                max(
                    0.20, self.refinement_config.get("residual_detail_edge_gamma", 0.70)
                )
            )
            edge_error_floor = float(
                np.clip(
                    self.refinement_config.get(
                        "residual_detail_edge_error_floor", 0.20
                    ),
                    0.0,
                    1.0,
                )
            )
            edge_score = self._normalize_map(
                np.power(np.clip(edge_map, 0.0, 1.0), edge_gamma).astype(np.float32)
                * np.clip(edge_error_floor + error_norm, 0.0, None)
            )
            edge_score = self._apply_saliency_sampling_bias(
                edge_score,
                strength=float(
                    self.refinement_config.get(
                        "residual_detail_edge_saliency_weight", 0.85
                    )
                ),
            )
            edge_percentile = float(
                np.clip(
                    self.refinement_config.get(
                        "residual_detail_edge_percentile", max(55.0, percentile - 18.0)
                    ),
                    0.0,
                    100.0,
                )
            )

            candidates: List[Tuple[int, int, float, bool]] = []
            seen: set[Tuple[int, int]] = set()

            def add_candidates(
                x_values: np.ndarray,
                y_values: np.ndarray,
                weights: np.ndarray,
                *,
                is_edge: bool,
            ) -> None:
                for x_raw, y_raw, weight_raw in zip(x_values, y_values, weights):
                    x = int(np.clip(int(x_raw), 0, width - 1))
                    y = int(np.clip(int(y_raw), 0, height - 1))
                    key = (x, y)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        (x, y, float(np.clip(weight_raw, 0.0, 1.0)), bool(is_edge))
                    )
                    if len(candidates) >= max_new:
                        break

            if edge_count > 0 and float(np.max(edge_score)) > 1e-8:
                edge_x, edge_y, edge_weights = self._sample_candidate_positions(
                    score_map=edge_score,
                    percentile=edge_percentile,
                    max_samples=edge_count,
                    rng=rng,
                )
                add_candidates(edge_x, edge_y, edge_weights, is_edge=True)

            if residual_count > 0 and len(candidates) < max_new:
                res_x, res_y, res_weights = self._sample_candidate_positions(
                    score_map=score_map,
                    percentile=percentile,
                    max_samples=residual_count,
                    rng=rng,
                )
                add_candidates(res_x, res_y, res_weights, is_edge=False)

            if len(candidates) < max_new:
                fill_x, fill_y, fill_weights = self._sample_candidate_positions(
                    score_map=score_map,
                    percentile=max(0.0, percentile - 12.0),
                    max_samples=max_new - len(candidates),
                    rng=rng,
                )
                add_candidates(fill_x, fill_y, fill_weights, is_edge=False)

            if not candidates:
                break

            sigma_min = float(
                max(0.10, self.refinement_config.get("residual_detail_sigma_min", 0.28))
            )
            sigma_max = float(
                max(
                    sigma_min,
                    self.refinement_config.get("residual_detail_sigma_max", 1.20),
                )
            )
            edge_sigma_min = float(
                max(
                    0.04,
                    self.refinement_config.get(
                        "residual_detail_edge_sigma_min", sigma_min * 0.60
                    ),
                )
            )
            edge_sigma_max = float(
                max(
                    edge_sigma_min,
                    self.refinement_config.get(
                        "residual_detail_edge_sigma_max", min(sigma_max, 0.70)
                    ),
                )
            )
            edge_sigma_major_max = float(
                max(
                    edge_sigma_max,
                    self.refinement_config.get(
                        "residual_detail_edge_sigma_major_max", edge_sigma_max * 1.8
                    ),
                )
            )
            alpha_min = float(
                np.clip(
                    self.refinement_config.get("residual_detail_alpha_min", 0.16),
                    0.0,
                    1.0,
                )
            )
            alpha_max = float(
                np.clip(
                    self.refinement_config.get("residual_detail_alpha_max", 0.70),
                    alpha_min,
                    1.0,
                )
            )
            residual_color_gain = float(
                self.refinement_config.get("residual_detail_color_gain", 0.95)
            )
            edge_color_gain = float(
                self.refinement_config.get(
                    "residual_detail_edge_color_gain", residual_color_gain
                )
            )
            edge_alpha_boost = float(
                max(
                    0.0,
                    self.refinement_config.get(
                        "residual_detail_edge_alpha_boost", 0.06
                    ),
                )
            )
            edge_make_threshold = float(
                np.clip(
                    self.refinement_config.get(
                        "residual_detail_edge_make_threshold", 0.20
                    ),
                    0.0,
                    1.0,
                )
            )
            edge_anisotropic = bool(
                self.refinement_config.get("residual_detail_edge_anisotropic", True)
            )
            edge_anisotropy_threshold = float(
                max(
                    1.0,
                    self.refinement_config.get(
                        "residual_detail_edge_anisotropy_threshold", 1.20
                    ),
                )
            )
            edge_aspect = float(
                max(1.0, self.refinement_config.get("residual_detail_edge_aspect", 2.0))
            )

            new_splats: List[GaussianSplat] = []
            edge_candidates_used = 0
            anisotropic_edge_splats = 0
            for x, y, sample_weight, sampled_from_edge_pool in candidates:
                base_color = estimate_local_color(image, x, y)
                residual_rgb = residual_np[y, x, :3].astype(np.float32)
                edge_need = float(edge_map[y, x])
                is_edge_candidate = bool(
                    sampled_from_edge_pool or edge_need >= edge_make_threshold
                )
                color_gain = (
                    edge_color_gain if is_edge_candidate else residual_color_gain
                )
                color = np.clip(
                    base_color + color_gain * residual_rgb, 0.0, 1.0
                ).astype(np.float32)
                if not np.isfinite(color).all():
                    color = base_color

                detail_need = float(error_norm[y, x])
                if is_edge_candidate:
                    edge_candidates_used += 1
                    edge_detail_need = float(
                        np.clip(0.65 * detail_need + 0.35 * edge_need, 0.0, 1.0)
                    )
                    sigma = float(
                        np.clip(
                            edge_sigma_max
                            - (edge_sigma_max - edge_sigma_min) * edge_detail_need,
                            edge_sigma_min,
                            edge_sigma_max,
                        )
                    )
                    alpha = float(
                        np.clip(
                            alpha_min
                            + (alpha_max - alpha_min) * (0.35 + 0.65 * edge_detail_need)
                            + edge_alpha_boost,
                            alpha_min,
                            alpha_max,
                        )
                    )
                else:
                    sigma = float(
                        np.clip(
                            sigma_max - (sigma_max - sigma_min) * detail_need,
                            sigma_min,
                            sigma_max,
                        )
                    )
                    alpha = float(
                        np.clip(
                            alpha_min
                            + (alpha_max - alpha_min) * (0.30 + 0.70 * detail_need),
                            alpha_min,
                            alpha_max,
                        )
                    )
                x_center = float(
                    np.clip(x + rng.uniform(-0.35, 0.35), 0.0, width - 1.0)
                )
                y_center = float(
                    np.clip(y + rng.uniform(-0.35, 0.35), 0.0, height - 1.0)
                )

                splat: GaussianSplat
                if is_edge_candidate and edge_anisotropic:
                    primary_direction, anisotropy = self._analyze_local_structure(
                        image, x, y
                    )
                    if float(anisotropy) >= edge_anisotropy_threshold:
                        angle = edge_tangent_angle(primary_direction)
                        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                        rotation_matrix = np.array(
                            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                        )
                        sigma_major = float(
                            np.clip(sigma * edge_aspect, sigma, edge_sigma_major_max)
                        )
                        sigma_minor = float(max(edge_sigma_min, sigma / edge_aspect))
                        splat = create_anisotropic_splat(
                            center=np.array([x_center, y_center], dtype=np.float32),
                            eigenvals=np.array(
                                [sigma_major**2, sigma_minor**2], dtype=np.float32
                            ),
                            eigenvecs=rotation_matrix,
                            color=color,
                            alpha=alpha,
                        )
                        anisotropic_edge_splats += 1
                    else:
                        splat = create_isotropic_splat(
                            center=np.array([x_center, y_center], dtype=np.float32),
                            sigma=sigma,
                            color=color,
                            alpha=alpha,
                        )
                else:
                    splat = create_isotropic_splat(
                        center=np.array([x_center, y_center], dtype=np.float32),
                        sigma=sigma,
                        color=color,
                        alpha=alpha,
                    )
                layer, local_importance = self._saliency_layer_for_pixel(
                    x,
                    y,
                    LAYER_EDGE if is_edge_candidate else LAYER_DETAIL,
                )
                local_importance = float(
                    max(local_importance, 0.65 + 0.35 * sample_weight)
                )
                if is_edge_candidate:
                    local_importance = float(
                        max(local_importance, 0.80 + 0.19 * edge_need)
                    )
                self._assign_splat_layer(splat, layer, local_importance)
                new_splats.append(splat)

            if not new_splats:
                break

            if verbose:
                logger.info(
                    "Residual detail pass %s: adding %s small splats (%s edge candidates)",
                    pass_idx + 1,
                    len(new_splats),
                    edge_candidates_used,
                )

            current_splats = current_splats + new_splats
            residual_iters = int(
                max(0, self.refinement_config.get("residual_detail_iters", 8))
            )
            current_splats, stage_metric, stage_rendered = self._optimize_stage(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                num_iters=residual_iters,
                verbose=verbose,
            )

            quality, _, _ = self._compute_quality_metrics_cached(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                precomputed_rendered=stage_rendered,
            )
            stage_metric.update(quality)
            stage_metric["stage"] = -1
            stage_metric["stage_type"] = "residual_detail"
            stage_metric["residual_pass"] = pass_idx + 1
            stage_metric["splat_count"] = len(current_splats)
            stage_metric["residual_detail_added"] = len(new_splats)
            stage_metric["residual_detail_edge_candidates"] = int(edge_candidates_used)
            stage_metric["residual_detail_anisotropic_edge_splats"] = int(
                anisotropic_edge_splats
            )
            stage_metric["residual_detail_edge_fraction"] = float(edge_fraction)
            stage_metric["residual_detail_edge_percentile"] = float(edge_percentile)
            stage_metric["residual_detail_elapsed_sec"] = float(
                time.perf_counter() - pass_t0
            )
            remaining = self._time_budget_seconds_remaining()
            if remaining is not None:
                stage_metric["time_budget_remaining_sec"] = max(0.0, float(remaining))
                stage_metric["time_budget_exhausted"] = bool(
                    self._time_budget_exhausted()
                )
            residual_metrics.append(stage_metric)
            if verbose:
                logger.info(
                    "Residual detail pass %s done in %.2fs: SSIM_sRGB=%.4f, splats=%s",
                    pass_idx + 1,
                    stage_metric["residual_detail_elapsed_sec"],
                    float(stage_metric.get("ssim_srgb", 0.0)),
                    len(current_splats),
                )

        return current_splats, residual_metrics
