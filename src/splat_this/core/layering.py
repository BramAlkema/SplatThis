"""Depth scoring and layer assignment for splats."""

from typing import Dict, List
import numpy as np

if False:  # TYPE_CHECKING
    from .extract import Gaussian


class ImportanceScorer:
    """Score splats by visual importance for depth assignment."""

    def __init__(
        self,
        area_weight: float = 0.3,
        edge_weight: float = 0.5,
        color_weight: float = 0.2,
    ):
        self.area_weight = area_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

    def score_splats(self, splats: List["Gaussian"], image: np.ndarray) -> None:
        """Update splat scores based on importance factors."""
        # TODO: Implement importance scoring
        # Placeholder: assign random scores
        for splat in splats:
            splat.score = np.random.random()


class LayerAssigner:
    """Assign splats to depth layers based on importance scores."""

    def __init__(self, n_layers: int = 4):
        self.n_layers = n_layers

    def assign_layers(self, splats: List["Gaussian"]) -> Dict[int, List["Gaussian"]]:
        """Assign splats to depth layers based on scores."""
        # TODO: Implement layer assignment
        # Placeholder: put all splats in layer 0
        return {0: splats}


class QualityController:
    """Apply quality control and filtering to splat collections."""

    def __init__(self, target_count: int, k_multiplier: float = 2.5):
        self.target_count = target_count
        self.k_multiplier = k_multiplier

    def optimize_splats(self, splats: List["Gaussian"]) -> List["Gaussian"]:
        """Apply quality control and filtering to splat collection."""
        # TODO: Implement quality control
        # Placeholder: return first target_count splats
        return splats[: self.target_count]
