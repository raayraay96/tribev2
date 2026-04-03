# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Karpathy loop — automated parameter search for PFC stability scoring.

Implements an iterative experiment loop that systematically searches
the parameter space for optimal scoring configuration:

1. Define parameter grid (scoring method, variance_scale, thresholds, etc.)
2. For each configuration, run inference on test stimuli
3. Evaluate scoring quality via differentiation metrics
4. Rank configurations and select the best
5. Export optimal config to YAML

Designed for Scholar cluster SLURM execution with 10-minute budget per config.
"""

import itertools
import json
import logging
import time
import typing as tp
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tribev2.pfc_roi import PFCRegion, extract_pfc_vertices, get_pfc_mask
from tribev2.scoring import ScoringConfig, score_pfc_stability
from tribev2.state_machine import (
    CognitiveState,
    ScaffoldingStateMachine,
    StateMachineConfig,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single Karpathy loop experiment.

    Parameters
    ----------
    name : str
        Human-readable experiment name.
    scoring_method : str
        Scoring method: "inverse_variance", "activation_ratio", or "combined".
    scoring_config : ScoringConfig
        Scoring parameters for this experiment.
    state_machine_config : StateMachineConfig
        State machine parameters.
    pfc_region : PFCRegion
        PFC region to use for scoring.
    """

    name: str
    scoring_method: str = "inverse_variance"
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)
    state_machine_config: StateMachineConfig = field(
        default_factory=StateMachineConfig
    )
    pfc_region: PFCRegion = PFCRegion.ALL


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config_name: str
    scoring_method: str

    # Quality metrics
    state_differentiation: float  # Higher = better state separation
    stability_variance: float  # Lower = more stable scores
    transition_count: int  # Lower = less oscillation
    mean_score: float
    score_range: float  # Dynamic range of scores

    # State distribution
    pct_stable: float
    pct_edge: float
    pct_scaffolding: float

    # Composite ranking score (higher = better)
    rank_score: float = 0.0

    # Timing
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, tp.Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def generate_parameter_grid(
    compact: bool = False,
) -> list[ExperimentConfig]:
    """Generate a grid of experiment configurations to search.

    Parameters
    ----------
    compact : bool
        If True, use a smaller grid for quick testing.

    Returns
    -------
    configs : list[ExperimentConfig]
        List of experiment configurations.
    """
    if compact:
        variance_scales = [50, 100]
        combined_alphas = [0.6]
        smoothing_kernels = [5]
        lower_thresholds = [0.40]
        upper_thresholds = [0.55]
        methods = ["inverse_variance", "combined"]
        sustain_timesteps = [3]
    else:
        variance_scales = [10, 50, 100, 200, 500]
        combined_alphas = [0.4, 0.6, 0.8]
        smoothing_kernels = [0, 3, 5, 7]
        lower_thresholds = [0.30, 0.35, 0.40, 0.45]
        upper_thresholds = [0.50, 0.55, 0.60, 0.65]
        methods = ["inverse_variance", "activation_ratio", "combined"]
        sustain_timesteps = [2, 3, 5]

    configs = []
    config_id = 0

    for method in methods:
        for vs in variance_scales:
            for alpha in combined_alphas:
                for sk in smoothing_kernels:
                    for lt in lower_thresholds:
                        for ut in upper_thresholds:
                            if lt >= ut:
                                continue
                            for st in sustain_timesteps:
                                config_id += 1
                                name = (
                                    f"exp_{config_id:04d}_{method[:3]}"
                                    f"_vs{vs}_a{alpha}_sk{sk}"
                                    f"_th{lt}-{ut}_st{st}"
                                )
                                configs.append(
                                    ExperimentConfig(
                                        name=name,
                                        scoring_method=method,
                                        scoring_config=ScoringConfig(
                                            variance_scale=vs,
                                            combined_alpha=alpha,
                                            smoothing_kernel=sk,
                                        ),
                                        state_machine_config=StateMachineConfig(
                                            lower_threshold=lt,
                                            upper_threshold=ut,
                                            sustain_timesteps=st,
                                        ),
                                    )
                                )

    LOGGER.info("Generated %d experiment configurations", len(configs))
    return configs


def evaluate_experiment(
    predictions: np.ndarray,
    config: ExperimentConfig,
    stimulus_labels: list[str] | None = None,
) -> ExperimentResult:
    """Run a single experiment configuration and evaluate its quality.

    Parameters
    ----------
    predictions : np.ndarray of shape (n_timesteps, 20484)
        Full cortical predictions from TRIBE v2.
    config : ExperimentConfig
        Experiment configuration to evaluate.
    stimulus_labels : list[str] or None
        Labels for each timestep to evaluate state differentiation.

    Returns
    -------
    result : ExperimentResult
        Evaluation metrics for this configuration.
    """
    t0 = time.monotonic()
    mask = get_pfc_mask(region=config.pfc_region)
    pfc = extract_pfc_vertices(predictions, mask=mask)

    # Score
    if config.scoring_method == "activation_ratio":
        scores, baseline = score_pfc_stability(
            pfc,
            method=config.scoring_method,
            whole_brain=predictions,
            config=config.scoring_config,
        )
    else:
        scores, baseline = score_pfc_stability(
            pfc,
            method=config.scoring_method,
            config=config.scoring_config,
        )

    # State machine
    sm = ScaffoldingStateMachine(config.state_machine_config)
    states, transitions = sm.process_batch(scores)

    # Compute metrics
    state_counts = {s: 0 for s in CognitiveState}
    for s in states:
        state_counts[s] += 1
    n_total = len(states)

    pct_stable = state_counts[CognitiveState.STABLE] / n_total
    pct_edge = state_counts[CognitiveState.EDGE] / n_total
    pct_scaffolding = state_counts[CognitiveState.SCAFFOLDING] / n_total

    # State differentiation: entropy across states (higher = more diverse)
    probs = np.array([pct_stable, pct_edge, pct_scaffolding])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    # Score quality metrics
    stability_variance = float(np.var(scores))
    score_range = float(scores.max() - scores.min())
    mean_score = float(np.mean(scores))

    # Composite ranking: balance differentiation, stability, and range
    rank_score = (
        entropy * 0.4              # Want diverse states
        + score_range * 0.3        # Want dynamic range
        - stability_variance * 0.1  # Don't want too noisy
        - len(transitions) * 0.01   # Penalize excessive transitions
    )

    runtime = time.monotonic() - t0

    return ExperimentResult(
        config_name=config.name,
        scoring_method=config.scoring_method,
        state_differentiation=float(entropy),
        stability_variance=stability_variance,
        transition_count=len(transitions),
        mean_score=mean_score,
        score_range=score_range,
        pct_stable=pct_stable,
        pct_edge=pct_edge,
        pct_scaffolding=pct_scaffolding,
        rank_score=float(rank_score),
        runtime_seconds=runtime,
    )


def run_karpathy_loop(
    predictions: np.ndarray,
    configs: list[ExperimentConfig] | None = None,
    compact: bool = True,
    output_dir: str | Path = ".planning/karpathy_results",
    stimulus_labels: list[str] | None = None,
) -> tuple[ExperimentConfig, list[ExperimentResult]]:
    """Run the full Karpathy loop: sweep parameter grid, rank, select best.

    Parameters
    ----------
    predictions : np.ndarray of shape (n_timesteps, 20484)
        Full cortical predictions from TRIBE v2.
    configs : list[ExperimentConfig] or None
        Custom configurations. If None, generates from parameter grid.
    compact : bool
        If True, use smaller parameter grid.
    output_dir : str or Path
        Directory to save results.
    stimulus_labels : list[str] or None
        Labels for differentiation analysis.

    Returns
    -------
    best_config : ExperimentConfig
        The configuration with the highest rank score.
    results : list[ExperimentResult]
        All experiment results, sorted by rank_score descending.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = generate_parameter_grid(compact=compact)

    LOGGER.info("Starting Karpathy loop with %d configurations", len(configs))
    results = []
    t_start = time.monotonic()

    for i, config in enumerate(configs):
        LOGGER.info(
            "[%d/%d] Running: %s", i + 1, len(configs), config.name
        )
        try:
            result = evaluate_experiment(
                predictions, config, stimulus_labels
            )
            results.append(result)
            LOGGER.info(
                "  rank=%.4f, diff=%.3f, range=%.3f, transitions=%d (%.1fs)",
                result.rank_score,
                result.state_differentiation,
                result.score_range,
                result.transition_count,
                result.runtime_seconds,
            )
        except Exception as e:
            LOGGER.error("  FAILED: %s", e)
            continue

    # Sort by rank score (descending)
    results.sort(key=lambda r: r.rank_score, reverse=True)

    total_time = time.monotonic() - t_start
    LOGGER.info(
        "Karpathy loop complete: %d/%d succeeded in %.1fs",
        len(results),
        len(configs),
        total_time,
    )

    if not results:
        raise RuntimeError("All experiments failed — no results to rank")

    best_config = configs[0]
    for config in configs:
        if config.name == results[0].config_name:
            best_config = config
            break

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "best_config": results[0].config_name,
                "total_configs": len(configs),
                "total_time_seconds": total_time,
                "results": [r.to_dict() for r in results[:20]],  # Top 20
            },
            f,
            indent=2,
        )

    # Save best config
    best_file = output_path / "best_config.json"
    with open(best_file, "w") as f:
        json.dump(
            {
                "name": best_config.name,
                "scoring_method": best_config.scoring_method,
                "scoring_config": {
                    "variance_scale": best_config.scoring_config.variance_scale,
                    "combined_alpha": best_config.scoring_config.combined_alpha,
                    "smoothing_kernel": best_config.scoring_config.smoothing_kernel,
                    "smoothing_sigma": best_config.scoring_config.smoothing_sigma,
                    "baseline_window": best_config.scoring_config.baseline_window,
                    "asymmetric_smoothing": best_config.scoring_config.asymmetric_smoothing,
                },
                "state_machine_config": {
                    "lower_threshold": best_config.state_machine_config.lower_threshold,
                    "upper_threshold": best_config.state_machine_config.upper_threshold,
                    "sustain_timesteps": best_config.state_machine_config.sustain_timesteps,
                },
                "rank_score": results[0].rank_score,
            },
            f,
            indent=2,
        )

    LOGGER.info("Best config: %s (rank=%.4f)", best_config.name, results[0].rank_score)
    LOGGER.info("Results saved to %s", output_path)

    return best_config, results
