"""Per-run orchestration boundary for the public converter facade."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from .config import ConversionRequest, ConverterConfig, thaw

if TYPE_CHECKING:
    from .converter import PNG2SVGConverter


@dataclass
class RunContext:
    """Mutable state owned by exactly one conversion run."""

    request: ConversionRequest
    config: ConverterConfig
    start_time: float
    rng: np.random.Generator
    timings: Dict[str, float] = field(default_factory=dict)
    artifacts_path: Path | None = None

    @classmethod
    def create(
        cls, request: ConversionRequest, config: ConverterConfig
    ) -> "RunContext":
        run_seed = config.seed if request.seed is None else request.seed
        rng = np.random.default_rng(run_seed)
        if run_seed is not None:
            torch.manual_seed(int(run_seed))
        artifacts_path = (
            None if request.artifacts_dir is None else Path(request.artifacts_dir)
        )
        if artifacts_path is not None:
            artifacts_path.mkdir(parents=True, exist_ok=True)
        return cls(
            request=request,
            config=config,
            start_time=time.perf_counter(),
            rng=rng,
            artifacts_path=artifacts_path,
        )

    @property
    def run_seed(self) -> int | None:
        return self.config.seed if self.request.seed is None else self.request.seed


class ConversionPipeline:
    """Creates isolated execution state and invokes the conversion phases."""

    def __init__(self, converter: "PNG2SVGConverter") -> None:
        self._converter = converter
        self._config = ConverterConfig.from_converter(converter)

    @property
    def config(self) -> ConverterConfig:
        return self._config

    def run(self, request: ConversionRequest) -> str:
        context = RunContext.create(request=request, config=self._config)
        runner = copy.copy(self._converter)

        # The algorithms currently tune these values during a run.  They are
        # detached here, never restored later, so the public converter remains
        # reentrant and concurrent calls cannot ratchet each other's budgets.
        runner.max_splats = int(self._config.max_splats)
        runner.stages = list(self._config.stages)
        runner.loss_weights = thaw(self._config.loss_weights)
        runner.learning_rates = thaw(self._config.learning_rates)
        runner.refinement_config = thaw(self._config.refinement)
        runner.schedule_config = thaw(self._config.schedule)
        runner.acceptance_criteria = thaw(self._config.acceptance)
        runner.training_export_target = self._config.training_export_target
        runner.time_budget_plan = None
        runner._time_budget_deadline = None
        runner._image_width = 1000
        runner._image_height = 1000
        runner._region_weight_map = None
        runner._region_saliency_map = None
        runner._region_detail_priority_map = None
        runner._region_background_penalty_map = None
        runner._region_foreground_mask = None
        runner._region_background_safe_mask = None
        runner._region_edge_band_mask = None

        return runner._convert_impl(request=request, context=context)
