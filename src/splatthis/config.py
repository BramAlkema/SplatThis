"""Immutable public configuration and per-conversion request models."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Tuple

SUPPORTED_OUTPUT_FORMATS = frozenset(
    {"svg", "drawingml", "pptx", "canvas", "css", "pixel-runtime"}
)


def _freeze(value: Any) -> Any:
    """Recursively detach mutable caller-owned configuration."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return copy.deepcopy(value)


def thaw(value: Any) -> Any:
    """Return a JSON-friendly mutable representation of frozen config."""

    if isinstance(value, Mapping):
        return {str(key): thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(thaw(item) for item in value)
    return copy.deepcopy(value)


def _json_default(value: Any) -> Any:
    """Canonicalize common scientific/path values without breaking conversion."""

    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    value_type = type(value)
    return {
        "type": f"{value_type.__module__}.{value_type.__qualname__}",
        "repr": repr(value),
    }


@dataclass(frozen=True)
class ConversionRequest:
    """All values that can vary between calls to one converter instance."""

    input_path: str
    output_path: Optional[str] = None
    save_json: bool = False
    verbose: bool = True
    output_format: str = "svg"
    seed: Optional[int] = None
    artifacts_dir: Optional[str] = None
    acceptance_criteria: Mapping[str, float] = field(default_factory=dict)
    acceptance_overridden: bool = False
    validate_roundtrip: bool = False
    side_by_side_html: Optional[str] = None
    preview_png_path: Optional[str] = None

    def __post_init__(self) -> None:
        normalized = str(self.output_format).strip().lower()
        if normalized not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        object.__setattr__(self, "output_format", normalized)
        object.__setattr__(self, "input_path", str(self.input_path))
        for name in (
            "output_path",
            "artifacts_dir",
            "side_by_side_html",
            "preview_png_path",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else str(value))
        object.__setattr__(
            self,
            "acceptance_criteria",
            _freeze(dict(self.acceptance_criteria)),
        )


@dataclass(frozen=True)
class ConverterConfig:
    """Detached immutable snapshot of a converter's constructed configuration.

    The optimizer still has a transitional mutable execution object, but every
    conversion starts from this stable value object.  This makes run identity
    explicit and provides a trustworthy basis for future content-addressed
    caches.
    """

    requested_max_splats: int
    max_splats: int
    k_sigma: float
    stages: Tuple[int, ...]
    target_size: Optional[Tuple[int, int]]
    gradient_method: str
    device: str
    seed: Optional[int]
    quality_profile: str
    resolution_scale: float
    renderer_backend: str
    resolved_renderer_backend: str
    optimizer_backend: str
    blend_mode: str
    compositing_space: str
    loss_color_space: str
    time_budget: Optional[str]
    layered_saliency: bool
    pptx_splat_style: str
    pptx_painter_order: str
    training_export_target: str
    training_export_target_explicit: bool
    loss_weights: Mapping[str, Any]
    learning_rates: Mapping[str, Any]
    refinement: Mapping[str, Any]
    schedule: Mapping[str, Any]
    acceptance: Mapping[str, Any]

    @classmethod
    def from_converter(cls, converter: Any) -> "ConverterConfig":
        target_size = converter.target_size
        return cls(
            requested_max_splats=int(converter.requested_max_splats),
            max_splats=int(converter.max_splats),
            k_sigma=float(converter.k_sigma),
            stages=tuple(int(item) for item in converter.stages),
            target_size=(
                None
                if target_size is None
                else (int(target_size[0]), int(target_size[1]))
            ),
            gradient_method=str(converter.gradient_method),
            device=str(converter.device),
            seed=converter.seed,
            quality_profile=str(converter.quality_profile),
            resolution_scale=float(converter.resolution_scale),
            renderer_backend=str(converter.renderer_backend),
            resolved_renderer_backend=str(converter.resolved_renderer_backend),
            optimizer_backend=str(converter.optimizer_backend),
            blend_mode=str(converter.blend_mode),
            compositing_space=str(converter.compositing_space),
            loss_color_space=str(converter.loss_color_space),
            time_budget=converter.time_budget,
            layered_saliency=bool(converter.layered_saliency),
            pptx_splat_style=str(converter.pptx_splat_style),
            pptx_painter_order=str(converter.pptx_painter_order),
            training_export_target=str(converter.training_export_target),
            training_export_target_explicit=bool(
                converter._training_export_target_explicit
            ),
            loss_weights=_freeze(converter.loss_weights),
            learning_rates=_freeze(converter.learning_rates),
            refinement=_freeze(converter.refinement_config),
            schedule=_freeze(converter.schedule_config),
            acceptance=_freeze(converter.acceptance_criteria),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_max_splats": self.requested_max_splats,
            "max_splats": self.max_splats,
            "k_sigma": self.k_sigma,
            "stages": list(self.stages),
            "target_size": None if self.target_size is None else list(self.target_size),
            "gradient_method": self.gradient_method,
            "device": self.device,
            "seed": self.seed,
            "quality_profile": self.quality_profile,
            "resolution_scale": self.resolution_scale,
            "renderer_backend": self.renderer_backend,
            "resolved_renderer_backend": self.resolved_renderer_backend,
            "optimizer_backend": self.optimizer_backend,
            "blend_mode": self.blend_mode,
            "compositing_space": self.compositing_space,
            "loss_color_space": self.loss_color_space,
            "time_budget": self.time_budget,
            "layered_saliency": self.layered_saliency,
            "pptx_splat_style": self.pptx_splat_style,
            "pptx_painter_order": self.pptx_painter_order,
            "training_export_target": self.training_export_target,
            "training_export_target_explicit": self.training_export_target_explicit,
            "loss_weights": thaw(self.loss_weights),
            "learning_rates": thaw(self.learning_rates),
            "refinement": thaw(self.refinement),
            "schedule": thaw(self.schedule),
            "acceptance": thaw(self.acceptance),
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
