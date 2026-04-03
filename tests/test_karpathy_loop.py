# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Karpathy loop automation module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tribev2.karpathy_loop import (
    ExperimentConfig,
    ExperimentResult,
    evaluate_experiment,
    generate_parameter_grid,
    run_karpathy_loop,
)
from tribev2.scoring import ScoringConfig
from tribev2.state_machine import StateMachineConfig

N_TOTAL_VERTICES = 20484
N_TIMESTEPS = 50


def _make_predictions(n_timesteps: int = N_TIMESTEPS) -> np.ndarray:
    """Generate synthetic cortical predictions."""
    rng = np.random.RandomState(42)
    return rng.randn(n_timesteps, N_TOTAL_VERTICES).astype(np.float32)


class TestParameterGrid:
    def test_compact_grid_size(self):
        configs = generate_parameter_grid(compact=True)
        assert len(configs) > 0
        assert len(configs) < 100  # Compact should be small

    def test_full_grid_larger(self):
        compact = generate_parameter_grid(compact=True)
        full = generate_parameter_grid(compact=False)
        assert len(full) > len(compact)

    def test_no_invalid_thresholds(self):
        configs = generate_parameter_grid(compact=False)
        for cfg in configs:
            assert (
                cfg.state_machine_config.lower_threshold
                < cfg.state_machine_config.upper_threshold
            )

    def test_unique_names(self):
        configs = generate_parameter_grid(compact=True)
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), "Duplicate config names"


class TestEvaluateExperiment:
    def test_basic_evaluation(self):
        predictions = _make_predictions()
        config = ExperimentConfig(
            name="test_basic",
            scoring_method="inverse_variance",
            state_machine_config=StateMachineConfig(sustain_timesteps=2),
        )
        result = evaluate_experiment(predictions, config)
        assert isinstance(result, ExperimentResult)
        assert result.config_name == "test_basic"

    def test_result_ranges(self):
        predictions = _make_predictions()
        config = ExperimentConfig(
            name="test_ranges",
            state_machine_config=StateMachineConfig(sustain_timesteps=2),
        )
        result = evaluate_experiment(predictions, config)
        assert 0 <= result.pct_stable <= 1
        assert 0 <= result.pct_edge <= 1
        assert 0 <= result.pct_scaffolding <= 1
        assert abs(result.pct_stable + result.pct_edge + result.pct_scaffolding - 1.0) < 1e-6
        assert result.runtime_seconds >= 0

    def test_activation_ratio_method(self):
        predictions = _make_predictions()
        config = ExperimentConfig(
            name="test_activation",
            scoring_method="activation_ratio",
            state_machine_config=StateMachineConfig(sustain_timesteps=2),
        )
        result = evaluate_experiment(predictions, config)
        assert result.scoring_method == "activation_ratio"

    def test_combined_method(self):
        predictions = _make_predictions()
        config = ExperimentConfig(
            name="test_combined",
            scoring_method="combined",
            state_machine_config=StateMachineConfig(sustain_timesteps=2),
        )
        result = evaluate_experiment(predictions, config)
        assert result.scoring_method == "combined"


class TestKarpathyLoop:
    def test_loop_finds_best(self):
        predictions = _make_predictions()
        with tempfile.TemporaryDirectory() as tmpdir:
            best_config, results = run_karpathy_loop(
                predictions,
                compact=True,
                output_dir=tmpdir,
            )
            assert isinstance(best_config, ExperimentConfig)
            assert len(results) > 0
            # Results should be sorted by rank_score descending
            for i in range(len(results) - 1):
                assert results[i].rank_score >= results[i + 1].rank_score

    def test_output_files_created(self):
        predictions = _make_predictions()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_karpathy_loop(
                predictions,
                compact=True,
                output_dir=tmpdir,
            )
            assert (Path(tmpdir) / "results.json").exists()
            assert (Path(tmpdir) / "best_config.json").exists()

    def test_results_json_valid(self):
        predictions = _make_predictions()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_karpathy_loop(
                predictions,
                compact=True,
                output_dir=tmpdir,
            )
            with open(Path(tmpdir) / "results.json") as f:
                data = json.load(f)
            assert "best_config" in data
            assert "results" in data
            assert len(data["results"]) > 0

    def test_best_config_json_valid(self):
        predictions = _make_predictions()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_karpathy_loop(
                predictions,
                compact=True,
                output_dir=tmpdir,
            )
            with open(Path(tmpdir) / "best_config.json") as f:
                data = json.load(f)
            assert "scoring_method" in data
            assert "scoring_config" in data
            assert "state_machine_config" in data
            assert "rank_score" in data

    def test_custom_configs(self):
        predictions = _make_predictions()
        custom = [
            ExperimentConfig(
                name="custom_1",
                scoring_method="inverse_variance",
                state_machine_config=StateMachineConfig(sustain_timesteps=2),
            ),
            ExperimentConfig(
                name="custom_2",
                scoring_method="combined",
                state_machine_config=StateMachineConfig(sustain_timesteps=2),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            best, results = run_karpathy_loop(
                predictions,
                configs=custom,
                output_dir=tmpdir,
            )
            assert len(results) == 2


class TestExperimentResult:
    def test_to_dict(self):
        result = ExperimentResult(
            config_name="test",
            scoring_method="inverse_variance",
            state_differentiation=1.5,
            stability_variance=0.02,
            transition_count=5,
            mean_score=0.5,
            score_range=0.4,
            pct_stable=0.3,
            pct_edge=0.4,
            pct_scaffolding=0.3,
            rank_score=1.2,
            runtime_seconds=0.5,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["config_name"] == "test"
        assert d["rank_score"] == 1.2
