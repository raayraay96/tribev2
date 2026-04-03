# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end cognitive scaffolding pipeline.

Unified interface that wires together all scaffolding components:
    TribeModel → PFC ROI → Scoring → Smoothing → State Machine

Usage:
    from tribev2.scaffolding_pipeline import ScaffoldingPipeline

    pipeline = ScaffoldingPipeline()
    result = pipeline.run(predictions)
    print(result.summary())
"""

import logging
import time
import typing as tp
from dataclasses import dataclass, field

import numpy as np

from tribev2.pfc_roi import (
    PFCRegion,
    extract_pfc_vertices,
    get_all_region_masks,
    get_pfc_mask,
    summarize_pfc_regions,
)
from tribev2.scoring import (
    ScoringConfig,
    SessionBaseline,
    score_pfc_stability,
)
from tribev2.state_machine import (
    CognitiveState,
    ScaffoldingStateMachine,
    StateMachineConfig,
    StateTransition,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full scaffolding pipeline.

    Parameters
    ----------
    pfc_region : PFCRegion
        Which PFC region to score. Default: all PFC.
    scoring_method : str
        Scoring method. Default: "inverse_variance".
    scoring_config : ScoringConfig
        Scoring parameters.
    state_machine_config : StateMachineConfig
        State machine parameters.
    enable_subregion_analysis : bool
        If True, also compute scores for each PFC sub-region.
    """

    pfc_region: PFCRegion = PFCRegion.ALL
    scoring_method: str = "inverse_variance"
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)
    state_machine_config: StateMachineConfig = field(
        default_factory=StateMachineConfig
    )
    enable_subregion_analysis: bool = False


@dataclass
class PipelineResult:
    """Results from a scaffolding pipeline run."""

    # Core outputs
    scores: np.ndarray  # (n_timesteps,) stability scores
    states: list[CognitiveState]  # Per-timestep states
    transitions: list[StateTransition]  # State transitions
    baseline: SessionBaseline  # Session baseline used

    # Sub-region analysis (optional)
    subregion_scores: dict[str, np.ndarray] | None = None

    # Metadata
    n_timesteps: int = 0
    n_pfc_vertices: int = 0
    runtime_seconds: float = 0.0
    config: PipelineConfig | None = None

    def summary(self) -> dict[str, tp.Any]:
        """Generate a human-readable summary of pipeline results.

        Returns
        -------
        summary : dict
            Pipeline result summary with all key metrics.
        """
        state_counts = {s.name: 0 for s in CognitiveState}
        for s in self.states:
            state_counts[s.name] += 1

        n = max(len(self.states), 1)
        return {
            "n_timesteps": self.n_timesteps,
            "n_pfc_vertices": self.n_pfc_vertices,
            "scoring_method": self.config.scoring_method if self.config else "unknown",
            "mean_score": float(np.mean(self.scores)),
            "std_score": float(np.std(self.scores)),
            "min_score": float(np.min(self.scores)),
            "max_score": float(np.max(self.scores)),
            "score_range": float(np.max(self.scores) - np.min(self.scores)),
            "state_distribution": {
                k: round(v / n, 3) for k, v in state_counts.items()
            },
            "total_transitions": len(self.transitions),
            "baseline": {
                "mean": self.baseline.mean,
                "std": self.baseline.std,
                "calibrated": self.baseline.is_calibrated,
            },
            "runtime_seconds": round(self.runtime_seconds, 3),
        }


class ScaffoldingPipeline:
    """End-to-end cognitive scaffolding pipeline.

    Wires together PFC extraction, scoring, smoothing, and state machine
    into a single callable interface.

    Parameters
    ----------
    config : PipelineConfig or None
        Pipeline configuration. Uses defaults if None.
    on_transition : callable or None
        Callback for state transitions.

    Examples
    --------
    >>> pipeline = ScaffoldingPipeline()
    >>> predictions = model.predict(events)  # (T, 20484)
    >>> result = pipeline.run(predictions)
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        on_transition: tp.Callable[[StateTransition], None] | None = None,
    ):
        self.config = config or PipelineConfig()
        self._pfc_mask = None
        self._subregion_masks = None
        self._state_machine = ScaffoldingStateMachine(
            config=self.config.state_machine_config,
            on_transition=on_transition,
        )

    def _ensure_masks(self) -> None:
        """Lazily initialize PFC masks."""
        if self._pfc_mask is None:
            self._pfc_mask = get_pfc_mask(region=self.config.pfc_region)
            LOGGER.info(
                "PFC mask initialized: %d vertices", self._pfc_mask.sum()
            )

        if self.config.enable_subregion_analysis and self._subregion_masks is None:
            self._subregion_masks = get_all_region_masks()

    def run(
        self,
        predictions: np.ndarray,
        whole_brain: np.ndarray | None = None,
    ) -> PipelineResult:
        """Run the full scaffolding pipeline.

        Parameters
        ----------
        predictions : np.ndarray of shape (n_timesteps, 20484)
            Full cortical surface predictions from TRIBE v2.
        whole_brain : np.ndarray or None
            Full brain predictions for activation_ratio method.
            If None, uses ``predictions`` directly.

        Returns
        -------
        result : PipelineResult
            Complete pipeline results.
        """
        t0 = time.monotonic()
        self._ensure_masks()

        # Step 1: Extract PFC vertices
        pfc = extract_pfc_vertices(predictions, mask=self._pfc_mask)
        n_pfc = pfc.shape[1]

        # Step 2: Score stability
        wb = whole_brain if whole_brain is not None else predictions
        if self.config.scoring_method == "activation_ratio":
            scores, baseline = score_pfc_stability(
                pfc,
                method=self.config.scoring_method,
                whole_brain=wb,
                config=self.config.scoring_config,
            )
        else:
            scores, baseline = score_pfc_stability(
                pfc,
                method=self.config.scoring_method,
                config=self.config.scoring_config,
            )

        # Step 3: State machine
        self._state_machine.reset()
        states, transitions = self._state_machine.process_batch(scores)

        # Step 4: Optional sub-region analysis
        subregion_scores = None
        if self.config.enable_subregion_analysis and self._subregion_masks:
            subregion_scores = {}
            for region, mask in self._subregion_masks.items():
                sub_pfc = predictions[:, mask]
                sub_scores, _ = score_pfc_stability(
                    sub_pfc,
                    method=self.config.scoring_method,
                    config=self.config.scoring_config,
                    calibrate=False,
                    smooth=False,
                )
                subregion_scores[region.value] = sub_scores

        runtime = time.monotonic() - t0

        return PipelineResult(
            scores=scores,
            states=states,
            transitions=transitions,
            baseline=baseline,
            subregion_scores=subregion_scores,
            n_timesteps=predictions.shape[0],
            n_pfc_vertices=n_pfc,
            runtime_seconds=runtime,
            config=self.config,
        )
