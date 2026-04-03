# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for PFC stability scoring module."""

import numpy as np
import pytest

from tribev2.scoring import (
    ScoringConfig,
    SessionBaseline,
    apply_baseline,
    compute_baseline,
    score_activation_ratio,
    score_combined,
    score_inverse_variance,
    score_pfc_stability,
)

# Fake PFC data dimensions
N_TIMESTEPS = 50
N_PFC_VERTICES = 3810
N_TOTAL_VERTICES = 20484


def _make_pfc_data(
    n_timesteps: int = N_TIMESTEPS,
    n_vertices: int = N_PFC_VERTICES,
    variance: float = 1.0,
) -> np.ndarray:
    """Generate synthetic PFC vertex data with controlled variance."""
    rng = np.random.RandomState(42)
    return rng.randn(n_timesteps, n_vertices).astype(np.float32) * variance


def _make_brain_data(
    n_timesteps: int = N_TIMESTEPS,
) -> np.ndarray:
    """Generate synthetic whole-brain data."""
    rng = np.random.RandomState(123)
    return rng.randn(n_timesteps, N_TOTAL_VERTICES).astype(np.float32)


class TestScoringConfig:
    def test_defaults(self):
        cfg = ScoringConfig()
        assert cfg.variance_scale == 100.0
        assert cfg.combined_alpha == 0.6
        assert cfg.baseline_window == 10
        assert cfg.clip_range == (0.0, 1.0)
        assert cfg.temporal_window == 5

    def test_custom(self):
        cfg = ScoringConfig(variance_scale=50, combined_alpha=0.8)
        assert cfg.variance_scale == 50
        assert cfg.combined_alpha == 0.8


class TestSessionBaseline:
    def test_uncalibrated(self):
        bl = SessionBaseline()
        assert not bl.is_calibrated

    def test_calibrated(self):
        bl = SessionBaseline(mean=0.5, std=0.1, n_samples=10)
        assert bl.is_calibrated

    def test_zero_std_not_calibrated(self):
        bl = SessionBaseline(mean=0.5, std=0.0, n_samples=10)
        assert not bl.is_calibrated


class TestInverseVariance:
    def test_output_shape(self):
        pfc = _make_pfc_data()
        scores = score_inverse_variance(pfc)
        assert scores.shape == (N_TIMESTEPS,)

    def test_output_range(self):
        pfc = _make_pfc_data()
        scores = score_inverse_variance(pfc)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_low_variance_high_score(self):
        low_var = _make_pfc_data(variance=0.01)
        high_var = _make_pfc_data(variance=10.0)
        scores_low = score_inverse_variance(low_var)
        scores_high = score_inverse_variance(high_var)
        assert np.mean(scores_low) > np.mean(scores_high)

    def test_variance_scale_effect(self):
        pfc = _make_pfc_data()
        cfg_sensitive = ScoringConfig(variance_scale=1000)
        cfg_insensitive = ScoringConfig(variance_scale=1)
        s1 = score_inverse_variance(pfc, cfg_sensitive)
        s2 = score_inverse_variance(pfc, cfg_insensitive)
        # Higher scale → lower scores for same variance
        assert np.mean(s1) < np.mean(s2)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="2D array"):
            score_inverse_variance(np.random.randn(100))


class TestActivationRatio:
    def test_output_shape(self):
        pfc = _make_pfc_data()
        brain = _make_brain_data()
        scores = score_activation_ratio(pfc, brain)
        assert scores.shape == (N_TIMESTEPS,)

    def test_output_range(self):
        pfc = _make_pfc_data()
        brain = _make_brain_data()
        scores = score_activation_ratio(pfc, brain)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_timestep_mismatch_raises(self):
        pfc = _make_pfc_data(n_timesteps=10)
        brain = _make_brain_data(n_timesteps=20)
        with pytest.raises(ValueError, match="Timestep mismatch"):
            score_activation_ratio(pfc, brain)


class TestCombined:
    def test_output_shape(self):
        pfc = _make_pfc_data()
        scores = score_combined(pfc)
        assert scores.shape == (N_TIMESTEPS,)

    def test_output_range(self):
        pfc = _make_pfc_data()
        scores = score_combined(pfc)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_alpha_weighting(self):
        pfc = _make_pfc_data()
        cfg_inv_heavy = ScoringConfig(combined_alpha=0.9)
        cfg_temp_heavy = ScoringConfig(combined_alpha=0.1)
        s1 = score_combined(pfc, cfg_inv_heavy)
        s2 = score_combined(pfc, cfg_temp_heavy)
        # Different weights should produce different scores
        assert not np.allclose(s1, s2)


class TestBaseline:
    def test_compute_baseline(self):
        scores = np.linspace(0.3, 0.7, 50)
        bl = compute_baseline(scores, window=10)
        assert bl.is_calibrated
        assert bl.n_samples == 10
        assert 0.0 < bl.mean < 1.0
        assert bl.std > 0

    def test_apply_baseline(self):
        scores = np.linspace(0.3, 0.7, 50)
        bl = compute_baseline(scores, window=10)
        calibrated = apply_baseline(scores, bl)
        assert calibrated.shape == scores.shape
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_uncalibrated_passthrough(self):
        scores = np.array([0.3, 0.5, 0.7])
        bl = SessionBaseline()  # uncalibrated
        calibrated = apply_baseline(scores, bl)
        np.testing.assert_array_equal(calibrated, scores)

    def test_zero_window_disables(self):
        scores = np.linspace(0.3, 0.7, 50)
        bl = compute_baseline(scores, window=0)
        assert not bl.is_calibrated


class TestUnifiedScoring:
    def test_inverse_variance_method(self):
        pfc = _make_pfc_data()
        scores, bl = score_pfc_stability(pfc, method="inverse_variance")
        assert scores.shape == (N_TIMESTEPS,)
        assert bl.is_calibrated

    def test_activation_ratio_method(self):
        pfc = _make_pfc_data()
        brain = _make_brain_data()
        scores, bl = score_pfc_stability(
            pfc, method="activation_ratio", whole_brain=brain
        )
        assert scores.shape == (N_TIMESTEPS,)

    def test_combined_method(self):
        pfc = _make_pfc_data()
        scores, bl = score_pfc_stability(pfc, method="combined")
        assert scores.shape == (N_TIMESTEPS,)

    def test_no_calibration(self):
        pfc = _make_pfc_data()
        scores, bl = score_pfc_stability(pfc, calibrate=False)
        assert not bl.is_calibrated

    def test_invalid_method_raises(self):
        pfc = _make_pfc_data()
        with pytest.raises(ValueError, match="Unknown method"):
            score_pfc_stability(pfc, method="nonexistent")

    def test_activation_ratio_without_brain_raises(self):
        pfc = _make_pfc_data()
        with pytest.raises(ValueError, match="whole_brain"):
            score_pfc_stability(pfc, method="activation_ratio")
