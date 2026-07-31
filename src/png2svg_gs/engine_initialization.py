"""Content-adaptive splat initialization and seed sampling."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .features import (
    analyze_local_structure,
    compute_gradient_magnitude,
    edge_tangent_angle,
    estimate_local_color,
    init_seeds_content_adaptive,
    poisson_disk_sampling,
)
from .splat import (
    LAYER_BASE,
    LAYER_EDGE,
    LAYER_MASS,
    GaussianSplat,
    create_anisotropic_splat,
    create_isotropic_splat,
)

logger = logging.getLogger(__name__)


class ConversionInitializationMixin:
    """Constructs the initial stratified, edge-aware splat population."""

    def _initialize_splats(  # noqa: C901
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        structure_primary: Optional[np.ndarray] = None,
        structure_anisotropy: Optional[np.ndarray] = None,
    ) -> List[GaussianSplat]:
        """
        Initialize splats with a guaranteed-coverage base layer plus detail layer.

        The base layer is stratified over the full canvas to avoid early empty regions.
        The detail layer is content-adaptive and edge-biased.
        """
        height, width = image.shape[:2]
        initial_count = self._initial_splat_count()
        if initial_count <= 0:
            return []

        base_fraction = float(
            np.clip(self.refinement_config.get("base_layer_fraction", 0.35), 0.10, 0.80)
        )
        base_count = max(4, int(round(initial_count * base_fraction)))
        detail_count = max(1, initial_count - base_count)

        base_positions = self._make_stratified_positions(
            width=width,
            height=height,
            count=base_count,
            rng=rng,
            jitter_ratio=0.65,
        )

        adaptive_count = max(
            1, int(round(detail_count * (1.0 - self.init_random_ratio)))
        )
        random_count = max(0, detail_count - adaptive_count)
        sampling_prior = self._sampling_prior_map()
        edge_count = 0
        edge_map: Optional[np.ndarray] = None
        edge_positions: List[Tuple[float, float]] = []
        edge_init_fraction = float(
            np.clip(self.refinement_config.get("edge_init_fraction", 0.0), 0.0, 0.85)
        )
        if edge_init_fraction > 0.0:
            edge_count = min(
                adaptive_count, int(round(adaptive_count * edge_init_fraction))
            )
            if edge_count > 0:
                edge_map = self._build_edge_map(image)
                edge_score = np.asarray(edge_map, dtype=np.float32)
                if sampling_prior is not None and sampling_prior.shape == (
                    height,
                    width,
                ):
                    edge_score = self._apply_saliency_sampling_bias(
                        edge_score,
                        strength=float(
                            self.refinement_config.get(
                                "edge_init_saliency_weight", 0.70
                            )
                        ),
                    )
                edge_positions = self._sample_map_positions(
                    score_map=edge_score,
                    count=edge_count,
                    rng=rng,
                    percentile=float(
                        self.refinement_config.get("edge_init_percentile", 68.0)
                    ),
                    jitter=0.28,
                )
                edge_count = len(edge_positions)
        saliency_count = 0
        if sampling_prior is not None and sampling_prior.shape == (height, width):
            saliency_fraction = float(
                np.clip(
                    self.refinement_config.get("saliency_init_fraction", 0.35),
                    0.0,
                    0.85,
                )
            )
            saliency_count = min(
                max(0, adaptive_count - edge_count),
                int(round(adaptive_count * saliency_fraction)),
            )
        content_adaptive_count = max(0, adaptive_count - edge_count - saliency_count)

        seed_positions = (
            init_seeds_content_adaptive(
                image=image,
                target_count=content_adaptive_count,
                gradient_weight=self.init_gradient_weight,
                method=self.gradient_method,
                rng=rng,
            )
            if content_adaptive_count > 0
            else []
        )
        saliency_positions: List[Tuple[float, float]] = []
        if saliency_count > 0 and sampling_prior is not None:
            saliency_positions = self._sample_map_positions(
                score_map=sampling_prior,
                count=saliency_count,
                rng=rng,
                percentile=float(
                    self.refinement_config.get("saliency_init_percentile", 62.0)
                ),
                jitter=0.45,
            )
        random_positions: List[Tuple[float, float]] = []
        if random_count > 0:
            random_saliency_fraction = float(
                np.clip(
                    self.refinement_config.get("saliency_random_fraction", 0.35),
                    0.0,
                    1.0,
                )
            )
            random_saliency_count = (
                min(random_count, int(round(random_count * random_saliency_fraction)))
                if sampling_prior is not None
                else 0
            )
            if random_saliency_count > 0 and sampling_prior is not None:
                random_positions.extend(
                    self._sample_map_positions(
                        score_map=sampling_prior,
                        count=random_saliency_count,
                        rng=rng,
                        percentile=float(
                            self.refinement_config.get(
                                "saliency_random_percentile", 55.0
                            )
                        ),
                        jitter=0.65,
                    )
                )
            uniform_count = random_count - random_saliency_count
            if uniform_count > 0:
                random_x = rng.uniform(0.0, float(width), size=uniform_count)
                random_y = rng.uniform(0.0, float(height), size=uniform_count)
                random_positions.extend(
                    (float(x), float(y)) for x, y in zip(random_x, random_y)
                )

        filled_count = (
            len(base_positions)
            + len(seed_positions)
            + len(edge_positions)
            + len(saliency_positions)
            + len(random_positions)
        )
        poisson_count = max(
            0, min(initial_count - filled_count, max(1, detail_count // 5))
        )
        if poisson_count > 0:
            min_distance = max(
                2.0, min(width, height) / max(np.sqrt(max(detail_count, 1.0)), 1.0)
            )
            poisson_positions = poisson_disk_sampling(
                width=width,
                height=height,
                min_distance=min_distance,
                rng=rng,
            )[:poisson_count]
        else:
            poisson_positions = []

        all_positions = (
            base_positions
            + seed_positions
            + edge_positions
            + saliency_positions
            + random_positions
            + poisson_positions
        )
        splats: List[GaussianSplat] = []
        edge_start = len(base_positions) + len(seed_positions)
        edge_end = edge_start + len(edge_positions)

        base_sigma = float(
            np.clip(
                np.sqrt((float(width) * float(height)) / max(base_count, 1)) * 0.85,
                self.refinement_config.get("sigma_min", 1.0),
                self.refinement_config.get("coverage_sigma_max", 8.0),
            )
        )
        base_alpha = float(
            np.clip(self.refinement_config.get("base_layer_alpha", 0.42), 0.08, 0.95)
        )
        sigma_minor_min = float(self.refinement_config.get("sigma_minor_min", 0.35))

        for idx, (x, y) in enumerate(all_positions):
            x_int = int(np.clip(x, 0, width - 1))
            y_int = int(np.clip(y, 0, height - 1))

            if (
                structure_primary is not None
                and structure_anisotropy is not None
                and structure_primary.shape[:2] == (height, width)
                and structure_primary.shape[-1] == 2
                and structure_anisotropy.shape == (height, width)
            ):
                primary_direction = structure_primary[y_int, x_int]
                anisotropy = float(structure_anisotropy[y_int, x_int])
            else:
                primary_direction, anisotropy = self._analyze_local_structure(
                    image, x_int, y_int
                )
            color = estimate_local_color(image, x_int, y_int)

            is_base = idx < len(base_positions)
            is_edge_init = edge_start <= idx < edge_end
            if is_base:
                sigma = base_sigma
                alpha = base_alpha
            elif is_edge_init:
                edge_need = (
                    float(edge_map[y_int, x_int]) if edge_map is not None else 1.0
                )
                edge_sigma_min = float(
                    max(0.10, self.refinement_config.get("edge_init_sigma_min", 0.45))
                )
                edge_sigma_max = float(
                    max(
                        edge_sigma_min,
                        self.refinement_config.get("edge_init_sigma_max", 1.25),
                    )
                )
                sigma = float(
                    np.clip(
                        edge_sigma_max - (edge_sigma_max - edge_sigma_min) * edge_need,
                        edge_sigma_min,
                        edge_sigma_max,
                    )
                )
                edge_alpha_min = float(
                    np.clip(
                        self.refinement_config.get("edge_init_alpha_min", 0.30),
                        0.0,
                        1.0,
                    )
                )
                edge_alpha_max = float(
                    np.clip(
                        self.refinement_config.get("edge_init_alpha_max", 0.72),
                        edge_alpha_min,
                        1.0,
                    )
                )
                alpha = float(
                    np.clip(
                        edge_alpha_min
                        + (edge_alpha_max - edge_alpha_min) * (0.35 + 0.65 * edge_need),
                        edge_alpha_min,
                        edge_alpha_max,
                    )
                )
            else:
                sigma = float(
                    np.clip(
                        base_sigma * 0.65,
                        self.refinement_config.get("sigma_min", 1.0),
                        6.0,
                    )
                )
                alpha = float(np.clip(base_alpha + 0.18, 0.15, 0.95))

            init_anisotropy_threshold = float(
                max(1.0, self.refinement_config.get("init_anisotropy_threshold", 1.55))
            )
            edge_init_anisotropy_threshold = float(
                max(
                    1.0,
                    self.refinement_config.get("edge_init_anisotropy_threshold", 1.15),
                )
            )
            if is_edge_init and anisotropy >= edge_init_anisotropy_threshold:
                angle = edge_tangent_angle(primary_direction)
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                sigma_major_scale = float(
                    max(
                        1.0,
                        self.refinement_config.get("edge_init_sigma_major_scale", 2.20),
                    )
                )
                sigma_major_max = float(
                    max(
                        sigma,
                        self.refinement_config.get("edge_init_sigma_major_max", 3.00),
                    )
                )
                sigma_major = float(
                    np.clip(sigma * sigma_major_scale, sigma, sigma_major_max)
                )
                sigma_minor = max(sigma, sigma_minor_min)
                splat = create_anisotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    eigenvals=np.array(
                        [sigma_major**2, sigma_minor**2], dtype=np.float32
                    ),
                    eigenvecs=rotation_matrix,
                    color=color,
                    alpha=alpha,
                )
            elif (not is_base) and anisotropy >= init_anisotropy_threshold:
                angle = edge_tangent_angle(primary_direction)
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                sigma_major = sigma
                sigma_minor = max(
                    sigma_major
                    / min(
                        float(anisotropy),
                        float(
                            self.refinement_config.get(
                                "local_structure_anisotropy_clip", 4.0
                            )
                        ),
                    ),
                    sigma_minor_min,
                )
                splat = create_anisotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    eigenvals=np.array(
                        [sigma_major**2, sigma_minor**2], dtype=np.float32
                    ),
                    eigenvecs=rotation_matrix,
                    color=color,
                    alpha=alpha,
                )
            else:
                splat = create_isotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    sigma=sigma,
                    color=color,
                    alpha=alpha,
                )

            if is_base:
                layer, local_importance = LAYER_BASE, 0.10
            elif is_edge_init:
                layer, local_importance = self._saliency_layer_for_pixel(
                    x_int, y_int, LAYER_EDGE
                )
                edge_need = (
                    float(edge_map[y_int, x_int]) if edge_map is not None else 1.0
                )
                local_importance = float(max(local_importance, 0.78 + 0.21 * edge_need))
            else:
                layer, local_importance = self._saliency_layer_for_pixel(
                    x_int, y_int, LAYER_MASS
                )
            self._assign_splat_layer(splat, layer, local_importance)
            splats.append(splat)

        logger.info(
            "Initialized %s splats (%s base + %s detail)",
            len(splats),
            len(base_positions),
            len(splats) - len(base_positions),
        )
        return splats

    def _make_stratified_positions(
        self,
        width: int,
        height: int,
        count: int,
        rng: np.random.Generator,
        jitter_ratio: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Generate approximately uniform stratified points over image space."""
        if count <= 0:
            return []

        aspect = float(width) / max(float(height), 1.0)
        cols = max(1, int(np.ceil(np.sqrt(float(count) * aspect))))
        rows = max(1, int(np.ceil(float(count) / float(cols))))
        cell_w = float(width) / float(cols)
        cell_h = float(height) / float(rows)
        jitter = float(np.clip(jitter_ratio, 0.0, 1.0))

        positions: List[Tuple[float, float]] = []
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= count:
                    break
                cx = (float(col) + 0.5) * cell_w
                cy = (float(row) + 0.5) * cell_h
                jx = (rng.random() - 0.5) * jitter * cell_w
                jy = (rng.random() - 0.5) * jitter * cell_h
                x = float(np.clip(cx + jx, 0.0, max(float(width) - 1.0, 0.0)))
                y = float(np.clip(cy + jy, 0.0, max(float(height) - 1.0, 0.0)))
                positions.append((x, y))
            if len(positions) >= count:
                break
        return positions

    def _sample_map_positions(
        self,
        score_map: np.ndarray,
        count: int,
        rng: np.random.Generator,
        percentile: float,
        jitter: float,
    ) -> List[Tuple[float, float]]:
        """Sample image positions from a continuous saliency/score map."""
        if count <= 0:
            return []
        score = np.asarray(score_map, dtype=np.float32)
        if score.ndim != 2 or score.size == 0:
            return []
        x_indices, y_indices, _ = self._sample_candidate_positions(
            score_map=score,
            percentile=float(np.clip(percentile, 0.0, 100.0)),
            max_samples=int(count),
            rng=rng,
        )
        height, width = score.shape
        jitter_amount = float(np.clip(jitter, 0.0, 1.0))
        positions: List[Tuple[float, float]] = []
        for x, y in zip(x_indices, y_indices):
            x_center = float(
                np.clip(
                    float(x) + rng.uniform(-jitter_amount, jitter_amount),
                    0.0,
                    width - 1.0,
                )
            )
            y_center = float(
                np.clip(
                    float(y) + rng.uniform(-jitter_amount, jitter_amount),
                    0.0,
                    height - 1.0,
                )
            )
            positions.append((x_center, y_center))
        return positions

    def _build_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Build normalized edge-energy map used by densification."""
        grad_mag = compute_gradient_magnitude(image, method=self.gradient_method)
        return self._normalize_map(grad_mag)

    def _analyze_local_structure(
        self, image: np.ndarray, x: int, y: int
    ) -> Tuple[np.ndarray, float]:
        """Analyze local orientation with conservative anisotropy gating."""
        return analyze_local_structure(
            image=image,
            x=x,
            y=y,
            window_size=int(
                max(3, self.refinement_config.get("structure_local_window", 7))
            ),
            anisotropy_clip=float(
                max(
                    1.0,
                    self.refinement_config.get("local_structure_anisotropy_clip", 4.0),
                )
            ),
            min_coherence=float(
                np.clip(
                    self.refinement_config.get("local_structure_min_coherence", 0.12),
                    0.0,
                    1.0,
                )
            ),
            min_energy=float(
                max(0.0, self.refinement_config.get("local_structure_min_energy", 1e-4))
            ),
        )
