"""Shared state and cross-mixin surface of the conversion engine.

``ConversionEngine`` composes seven mixins that split one stateful algorithm by
reason to change (see ``docs/ARCHITECTURE.md``). Each mixin reads attributes and
calls methods that a *sibling* mixin owns. Python resolves that at runtime
through the composed instance, but until this module existed no single class
declared the shared surface, so a type checker saw 339 spurious
``has no attribute`` errors and could not verify any cross-mixin call.

This base declares that surface once, and nothing else. It defines no behaviour
and assigns no values: ``engine_configuration.py`` remains the only place engine
state is initialized, and each mixin remains the only place its own methods are
implemented. Every mixin inherits from this class, which places it last in the
MRO, after every real implementation.

Adding shared state or a cross-mixin call means declaring it here. That is the
point: the declaration list is the engine's state contract, and it is meant to
be read.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from .adaptive_compute import OnlineAdaptiveConfig
    from .fidelity.config import FidelityConfig
    from .splat import GaussianSplat


class ConversionEngineState:
    """Declares the state and cross-mixin methods shared by the engine mixins."""

    # -- Population and geometry budgets -----------------------------------
    max_splats: int
    memory_guard_percent: Optional[float]
    _memory_guard: Dict[str, Any]
    k_sigma: float
    stages: List[int]
    target_size: Optional[Tuple[int, int]]
    resolution_scale: float

    # -- Initialization ----------------------------------------------------
    gradient_method: str
    init_random_ratio: float
    init_gradient_weight: float

    # -- Compositing -------------------------------------------------------
    blend_mode: str
    compositing_space: str

    # -- Devices and backends ----------------------------------------------
    device: torch.device
    renderer_backend: str
    optimizer_backend: str

    # -- Training configuration --------------------------------------------
    loss_weights: Dict[str, float]
    learning_rates: Dict[str, float]
    schedule_config: Dict[str, Any]
    refinement_config: Dict[str, Any]
    training_export_target: str

    # -- MLX backend -------------------------------------------------------
    mlx_loss: str
    mlx_tile_plan: str
    mlx_tile_plan_rebuild_interval: int
    mlx_trainable_groups: Tuple[str, ...]

    # -- Deployment targets ------------------------------------------------
    svg_export_recipe: str
    svg_gradient_quality: str
    svg_embed_population: bool
    svg_painter_order: str
    pptx_splat_style: str
    pptx_painter_order: str

    # -- Optional stages ---------------------------------------------------
    fidelity_config: FidelityConfig
    adaptive_compute_config: OnlineAdaptiveConfig

    # -- Region guidance ---------------------------------------------------
    # Installed by pipeline_phases._install_guidance, not by the constructor.
    region_weighting_enabled: bool
    _region_weight_map: Optional[npt.NDArray[Any]]
    _region_foreground_mask: Optional[npt.NDArray[Any]]
    _region_background_safe_mask: Optional[npt.NDArray[Any]]
    _region_edge_band_mask: Optional[npt.NDArray[Any]]
    _background_linear_rgb: npt.NDArray[Any]

    if TYPE_CHECKING:
        # Cross-mixin calls. Declared for the type checker only; the owning
        # mixin listed against each one provides the implementation, and this
        # class is last in the MRO, so none of these is ever reached.

        # engine_densification.py
        def _add_error_driven_splats(
            self,
            splats: List[GaussianSplat],
            image: npt.NDArray[Any],
            target: torch.Tensor,
            renderer: torch.nn.Module,
            rng: np.random.Generator,
            edge_map: Optional[npt.NDArray[Any]] = None,
            stage_idx: int = 0,
            precomputed_rendered: Optional[torch.Tensor] = None,
            precomputed_coverage_map: Optional[npt.NDArray[Any]] = None,
            structure_primary: Optional[npt.NDArray[Any]] = None,
            structure_anisotropy: Optional[npt.NDArray[Any]] = None,
            max_splats_cap: Optional[int] = None,
        ) -> Tuple[List[GaussianSplat], Optional[npt.NDArray[Any]]]: ...

        # engine_initialization.py
        def _analyze_local_structure(
            self, image: npt.NDArray[Any], x: int, y: int
        ) -> Tuple[npt.NDArray[Any], float]: ...

        # engine_configuration.py
        def _apply_saliency_sampling_bias(
            self, score_map: npt.NDArray[Any], strength: float
        ) -> npt.NDArray[Any]: ...

        # engine_guidance.py
        def _apply_splats_to_transmittance(
            self,
            transmittance: npt.NDArray[Any],
            splats: List[GaussianSplat],
            width: int,
            height: int,
        ) -> None: ...

        # engine_configuration.py
        def _assign_splat_layer(
            self, splat: GaussianSplat, layer: int, local_importance: float
        ) -> None: ...

        # engine_guidance.py
        def _build_alpha_coverage_map(
            self, splats: List[GaussianSplat], width: int, height: int
        ) -> npt.NDArray[Any]: ...

        # engine_initialization.py
        def _build_edge_map(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]: ...

        # engine_configuration.py
        def _clear_renderer_cache(self, renderer: torch.nn.Module) -> None: ...

        # engine_guidance.py
        def _compute_coverage_ratio(self, coverage_map: npt.NDArray[Any]) -> float: ...

        # engine_optimization.py
        def _compute_quality_metrics_cached(
            self,
            splats: List[GaussianSplat],
            target: torch.Tensor,
            renderer: torch.nn.Module,
            loss_fn: torch.nn.Module,
            precomputed_rendered: Optional[torch.Tensor] = None,
            precomputed_coverage_map: Optional[npt.NDArray[Any]] = None,
        ) -> Tuple[Dict[str, float], torch.Tensor, npt.NDArray[Any]]: ...

        # engine_configuration.py
        def _copy_splat_layers(
            self, source: List[GaussianSplat], optimized: List[GaussianSplat]
        ) -> List[GaussianSplat]: ...

        # engine_configuration.py
        def _create_training_loss(
            self, target: torch.Tensor, width: int, height: int
        ) -> torch.nn.Module: ...

        # engine_configuration.py
        def _create_training_renderer(
            self, width: int, height: int
        ) -> torch.nn.Module: ...

        # engine_configuration.py
        def _deployed_compositing_space(self) -> str: ...

        # engine_guidance.py
        def _get_profile_defaults(self, profile: str) -> Dict[str, Any]: ...

        # engine_configuration.py
        def _initial_splat_count(self) -> int: ...

        # engine_guidance.py
        def _loss_weight_tensor(
            self, width: int, height: int
        ) -> Optional[torch.Tensor]: ...

        # engine_guidance.py
        def _normalize_map(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]: ...

        # engine_optimization.py
        def _optimize_stage(
            self,
            splats: List[GaussianSplat],
            target: torch.Tensor,
            renderer: torch.nn.Module,
            loss_fn: torch.nn.Module,
            num_iters: int,
            verbose: bool,
        ) -> Tuple[List[GaussianSplat], Dict[str, Any], torch.Tensor]: ...

        # engine_postfit.py
        def _prefer_canvas_checkpoint(
            self,
            *,
            candidate: Dict[str, float],
            candidate_count: int,
            incumbent: Dict[str, float],
            incumbent_count: int,
        ) -> bool: ...

        # engine_postfit.py
        def _prune_splats(
            self,
            splats: List[GaussianSplat],
            max_count: int,
            target: Optional[torch.Tensor] = None,
            renderer: Optional[torch.nn.Module] = None,
            precomputed_coverage_map: Optional[npt.NDArray[Any]] = None,
        ) -> List[GaussianSplat]: ...

        # engine_configuration.py
        def _renderer_cache_stats(
            self, renderer: torch.nn.Module
        ) -> Dict[str, Any]: ...

        # engine_densification.py
        def _run_residual_detail_passes(
            self,
            splats: List[GaussianSplat],
            image: npt.NDArray[Any],
            target: torch.Tensor,
            renderer: torch.nn.Module,
            loss_fn: torch.nn.Module,
            rng: np.random.Generator,
            edge_map: npt.NDArray[Any],
            verbose: bool,
        ) -> Tuple[List[GaussianSplat], List[Dict[str, Any]]]: ...

        # engine_configuration.py
        def _saliency_layer_for_pixel(
            self, x: int, y: int, default_layer: int
        ) -> Tuple[int, float]: ...

        # engine_guidance.py
        def _sample_candidate_positions(
            self,
            score_map: npt.NDArray[Any],
            percentile: float,
            max_samples: int,
            rng: np.random.Generator,
        ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]: ...

        # engine_configuration.py
        def _sampling_prior_map(self) -> Optional[npt.NDArray[Any]]: ...

        # engine_postfit.py
        def _score_canvas_runtime_model(
            self, splats: List[GaussianSplat], image: npt.NDArray[Any]
        ) -> Dict[str, float]: ...

        # engine_configuration.py
        def _time_budget_exhausted(self) -> bool: ...

        # engine_configuration.py
        def _time_budget_seconds_remaining(self) -> Optional[float]: ...

        # engine_configuration.py
        def _use_mlx_spatial_weights(self) -> bool: ...

        # engine_configuration.py
        def _use_pptx_proxy_training(self) -> bool: ...

        # engine_artifacts.py
        def _write_stage_artifact(
            self,
            artifacts_dir: Optional[Path],
            stage_name: str,
            splats: List[GaussianSplat],
            metrics: Optional[Dict[str, Any]] = None,
        ) -> None: ...
