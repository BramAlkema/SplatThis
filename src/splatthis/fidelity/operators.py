"""Candidate operators (ADR-003 §5).

Phase-1 ships only the safest operator — bounded single-splat recolor —
because every proposal still passes the deployed-artifact accept-or-revert
gate. The wider portfolio (move/reshape/split/merge/reorder/recipe-tune)
lands operator-by-operator with ablations per the ADR delivery plan.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Sequence

import numpy as np
import numpy.typing as npt

from ..splat import GaussianSplat
from .analysis import ResidualAnalysis
from .config import FidelityConfig
from .stage import FidelityCandidate


class RecolorOperator:
    """Recolor the strongest splat inside each worst ROI toward the residual.

    Bounded: one splat per candidate, color shifted by a fraction of the
    ROI's mean signed linear residual, clipped to [0, 1]. Deterministic.
    """

    name = "recolor"

    def __init__(self, step: float = 0.8):
        self.step = float(step)

    def propose(
        self,
        best: FidelityCandidate,
        analysis: ResidualAnalysis,
        limit: int,
    ) -> Sequence[FidelityCandidate]:
        splats = list(best.splats)
        if not splats:
            return []
        centers = np.array([[s.mu[0], s.mu[1]] for s in splats], dtype=np.float32)
        alphas = np.array([s.alpha for s in splats], dtype=np.float32)

        proposals: List[FidelityCandidate] = []
        for roi_index, (y0, x0, y1, x1) in enumerate(analysis.fixed_rois):
            if len(proposals) >= limit:
                break
            mean_residual = analysis.residual_linear[y0:y1, x0:x1].mean(axis=(0, 1))
            if float(np.abs(mean_residual).max()) < 1e-3:
                continue
            inside = (
                (centers[:, 0] >= x0)
                & (centers[:, 0] < x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] < y1)
                & (alphas > 0.05)
            )
            candidates_idx = np.nonzero(inside)[0]
            if candidates_idx.size == 0:
                continue
            # Strongest contributor: highest alpha inside the ROI.
            target_idx = int(candidates_idx[np.argmax(alphas[candidates_idx])])
            original = splats[target_idx]
            new_color = np.clip(
                np.asarray(original.color[:3], dtype=np.float32)
                + self.step * mean_residual,
                0.0,
                1.0,
            )
            if float(np.abs(new_color - original.color[:3]).max()) < 1e-4:
                continue
            mutated = _with_color(original, new_color)
            new_splats = list(splats)
            new_splats[target_idx] = mutated
            proposals.append(
                FidelityCandidate(
                    name=f"recolor:roi{roi_index}:splat{target_idx}",
                    splats=tuple(new_splats),
                )
            )
        return proposals


def _with_color(splat: GaussianSplat, color: npt.NDArray[Any]) -> GaussianSplat:
    raw = splat.to_raw_splat()
    raw = replace(raw, r=float(color[0]), g=float(color[1]), b=float(color[2]))
    updated = GaussianSplat.from_raw_splat(raw)
    return updated


def build_operators(config: FidelityConfig) -> List[object]:
    """Operator portfolio per mode. Balanced = shell only (no proposals)."""
    if config.mode == "max":
        return [RecolorOperator()]
    return []
