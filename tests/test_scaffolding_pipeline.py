# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for the end-to-end scaffolding pipeline."""

import numpy as np
import pytest

from tribev2.pfc_roi import PFCRegion
from tribev2.scaffolding_pipeline import (
    PipelineConfig,
    PipelineResult,
    ScaffoldingPipeline,
)
from tribev2.scoring import ScoringConfig
from tribev2.state_machine import CognitiveState, StateMachineConfig

N_TOTAL_VERTICES = 20484


def _make_predictions(
    n_timesteps: int = 100, seed: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n_timesteps, N_TOTAL_VERTICES).astype(np.float32)


def _make_state_varied_predictions(n_per_state: int = 30) -> np.ndarray:
    """Create predictions with clearly different variance regimes.

    Low variance -> STABLE, Medium -> EDGE, High -> SCAFFOLDING
    """
    rng = np.random.RandomState(42)
    stable = rng.randn(n_per_state, N_TOTAL_VERTICES).astype(np.float32) * 0.01
    edge = rng.randn(n_per_state, N_TOTAL_VERTICES).astype(np.float32) * 0.5
    scaffolding = rng.randn(n_per_state, N_TOTAL_VERTICES).astype(np.float32) * 5.0
    return np.concatenate([stable, edge, scaffolding])


class TestScaffoldingPipelineBasic:
    """Basic pipeline tests."""

    def test_default_config_runs(self):
        predictions = _make_predictions()
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            )
        )
        result = pipeline.run(predictions)
        assert isinstance(result, PipelineResult)

    def test_result_shapes(self):
        n_timesteps = 50
        predictions = _make_predictions(n_timesteps=n_timesteps)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            )
        )
        result = pipeline.run(predictions)
        assert result.scores.shape == (n_timesteps,)
        assert len(result.states) == n_timesteps
        assert result.n_timesteps == n_timesteps
        assert result.n_pfc_vertices > 0

    def test_scores_in_range(self):
        predictions = _make_predictions()
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            )
        )
        result = pipeline.run(predictions)
        assert np.all(result.scores >= 0.0)
        assert np.all(result.scores <= 1.0)

    def test_all_states_are_cognitive_states(self):
        predictions = _make_predictions()
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            )
        )
        result = pipeline.run(predictions)
        assert all(isinstance(s, CognitiveState) for s in result.states)


class TestScoringMethods:
    """Test all scoring methods through the pipeline."""

    @pytest.mark.parametrize(
        "method",
        ["inverse_variance", "activation_ratio", "combined"],
    )
    def test_method_runs(self, method):
        predictions = _make_predictions(n_timesteps=50)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                scoring_method=method,
                state_machine_config=StateMachineConfig(sustain_timesteps=2),
            )
        )
        result = pipeline.run(predictions)
        assert result.scores.shape == (50,)
        assert np.all(result.scores >= 0.0)
        assert np.all(result.scores <= 1.0)

    def test_different_methods_produce_different_scores(self):
        predictions = _make_predictions(n_timesteps=30)
        cfg = StateMachineConfig(sustain_timesteps=2)
        scores = {}
        for method in ["inverse_variance", "combined"]:
            pipeline = ScaffoldingPipeline(
                PipelineConfig(scoring_method=method, state_machine_config=cfg)
            )
            result = pipeline.run(predictions)
            scores[method] = result.scores
        assert not np.allclose(scores["inverse_variance"], scores["combined"])


class TestSubregionAnalysis:
    def test_subregions_enabled(self):
        predictions = _make_predictions(n_timesteps=30)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                enable_subregion_analysis=True,
                state_machine_config=StateMachineConfig(sustain_timesteps=2),
            )
        )
        result = pipeline.run(predictions)
        assert result.subregion_scores is not None
        assert len(result.subregion_scores) == 4  # dlPFC, vlPFC, aPFC, vmPFC
        for name, scores in result.subregion_scores.items():
            assert scores.shape == (30,), f"{name} has wrong shape"

    def test_subregions_disabled(self):
        predictions = _make_predictions(n_timesteps=30)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                enable_subregion_analysis=False,
                state_machine_config=StateMachineConfig(sustain_timesteps=2),
            )
        )
        result = pipeline.run(predictions)
        assert result.subregion_scores is None


class TestPipelineSummary:
    def test_summary_keys(self):
        predictions = _make_predictions(n_timesteps=30)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            )
        )
        result = pipeline.run(predictions)
        summary = result.summary()

        required_keys = [
            "n_timesteps",
            "n_pfc_vertices",
            "scoring_method",
            "mean_score",
            "std_score",
            "min_score",
            "max_score",
            "score_range",
            "state_distribution",
            "total_transitions",
            "baseline",
            "runtime_seconds",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_state_distribution_sums_to_one(self):
        predictions = _make_predictions(n_timesteps=30)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            )
        )
        result = pipeline.run(predictions)
        summary = result.summary()
        dist = summary["state_distribution"]
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01, f"State distribution sums to {total}"


class TestPipelineWithTransitions:
    """Test transition callback integration."""

    def test_callback_receives_transitions(self):
        transitions = []
        predictions = _make_predictions(n_timesteps=100)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                state_machine_config=StateMachineConfig(sustain_timesteps=2)
            ),
            on_transition=lambda t: transitions.append(t),
        )
        result = pipeline.run(predictions)
        # Callback transitions should match result transitions
        assert len(transitions) == len(result.transitions)


class TestFullSuite:
    """Run all tests through the full pipeline to verify integration."""

    def test_full_suite_all_tests(self):
        """Run the full test suite to verify all modules integrate correctly."""
        predictions = _make_predictions(n_timesteps=50)
        pipeline = ScaffoldingPipeline(
            PipelineConfig(
                scoring_method="inverse_variance",
                scoring_config=ScoringConfig(
                    variance_scale=100,
                    smoothing_kernel=5,
                    asymmetric_smoothing=True,
                    baseline_window=10,
                ),
                state_machine_config=StateMachineConfig(
                    lower_threshold=0.40,
                    upper_threshold=0.55,
                    sustain_timesteps=3,
                ),
                enable_subregion_analysis=True,
            )
        )
        result = pipeline.run(predictions)

        # Verify all outputs
        assert result.scores.shape == (50,)
        assert len(result.states) == 50
        assert all(isinstance(s, CognitiveState) for s in result.states)
        assert np.all(result.scores >= 0.0)
        assert np.all(result.scores <= 1.0)
        assert result.baseline.is_calibrated
        assert result.subregion_scores is not None
        assert result.n_pfc_vertices > 3000  # ~3810 expected
        assert result.runtime_seconds > 0

        summary = result.summary()
        assert summary["n_timesteps"] == 50
        assert summary["scoring_method"] == "inverse_variance"
        assert 0 <= summary["mean_score"] <= 1
