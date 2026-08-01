"""Core values shared by optimization, export, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .splat import GaussianSplat


class EvidenceLevel(str, Enum):
    """Strength of the render used to grade an artifact."""

    DEPLOYED = "deployed"
    PARITY_MODEL = "parity-model"
    PROXY = "proxy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SplatScene:
    """A complete exportable 2D splat scene."""

    width: int
    height: int
    splats: Sequence[GaussianSplat]
    background_linear_rgb: np.ndarray
    compositing_space: str = "linear"

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("scene width and height must be positive")
        background = np.asarray(self.background_linear_rgb, dtype=np.float32).reshape(
            -1
        )
        if background.size != 3:
            raise ValueError("scene background must have exactly three components")
        detached_background = np.array(background, dtype=np.float32, copy=True)
        detached_background.setflags(write=False)
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "splats", tuple(self.splats))
        object.__setattr__(self, "background_linear_rgb", detached_background)


@dataclass(frozen=True)
class ArtifactPayload:
    """An emitted primary artifact before it is persisted."""

    output_format: str
    content: str
    media_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ArtifactEvaluation:
    """Typed provenance for the render governing acceptance."""

    evidence_level: EvidenceLevel
    render_kind: str
    renderer: str
    metric_source: str
    quality: Mapping[str, Any]
    acceptance_eligible: bool
    artifact_path: Optional[Path] = None

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "render_kind": self.render_kind,
            "renderer": self.renderer,
            "is_deployed_artifact": self.evidence_level == EvidenceLevel.DEPLOYED,
            "metric_source": self.metric_source,
        }
