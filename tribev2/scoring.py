# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PFC stability scoring functions for cognitive scaffolding.

Provides three complementary methods to quantify prefrontal cortex stability
from TRIBE v2 cortical predictions:

1. **inverse_variance** — Lower PFC variance → higher stability.
   Simple and interpretable. Default scoring method.

2. **activation_ratio** — PFC activation relative to whole-brain baseline.
   Higher ratio indicates stronger PFC engagement.

3. **combined** — Weighted combination of inverse_variance and temporal
   stability (low frame-to-frame jitter). Most robust but requires tuning.

All methods include per-session baseline calibration via z-score normalization
to handle the poor test-retest reliability of fMRI-level predictions.
"""

import logging
import typing as tp
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for PFC stability scoring.

    Parameters
    ----------
    variance_scale : float
        Scaling factor for inverse_variance method. Higher values increase
        sensitivity to small variance changes. Default: 100.
    combined_alpha : float
        Weight for inverse_variance in the combined method (0-1).
        combined = alpha * inv_var + (1-alpha) * temporal_stability.
        Default: 0.6.
    baseline_window : int
        Number of initial timesteps used for per-session baseline calibration.
        Set to 0 to disable baseline calibration. Default: 10.
    clip_range : tuple[float, float]
        Min/max range for output scores after all transformations.
        Default: (0.0, 1.0).
    temporal_window : int
        Number of timesteps for rolling temporal stability calculation.
        Default: 5.
    smoothing_kernel : int
        Gaussian smoothing kernel size (number of timesteps). Must be odd.
        Set to 0 to disable smoothing. Default: 5.
    smoothing_sigma : float
        Standard deviation for Gaussian kernel. Default: 1.0.
    asymmetric_smoothing : bool
        If True, smooth increases but preserve rapid drops. This prevents
        smoothing from masking acute stability decreases. Default: True.
    """

    variance_scale: float = 100.0
    combined_alpha: float = 0.6
    baseline_window: int = 10
    clip_range: tuple[float, float] = (0.0, 1.0)
    temporal_window: int = 5
    smoothing_kernel: int = 5
    smoothing_sigma: float = 1.0
    asymmetric_smoothing: bool = True


@dataclass
class SessionBaseline:
    """Per-session baseline statistics for z-score normalization.

    Computed from the first ``baseline_window`` timesteps of a session.
    Applied to raw scores before clipping.
    """

    mean: float = 0.0
    std: float = 1.0
    n_samples: int = 0

    @property
    def is_calibrated(self) -> bool:
        """Whether the baseline has been calibrated from actual data."""
        return self.n_samples > 0 and self.std > 0


def _clip_scores(scores: np.ndarray, clip_range: tuple[float, float]) -> np.ndarray:
    """Clip scores to the specified range."""
    return np.clip(scores, clip_range[0], clip_range[1])


def _normalize_to_unit(
    scores: np.ndarray, clip_range: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Normalize scores to [0, 1] range using min-max scaling, then clip."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-10:
        # Constant signal — return 0.5
        return np.full_like(scores, 0.5)
    normalized = (scores - s_min) / (s_max - s_min)
    return _clip_scores(normalized, clip_range)


def compute_baseline(
    scores: np.ndarray, window: int = 10
) -> SessionBaseline:
    """Compute per-session baseline statistics from initial timesteps.

    Parameters
    ----------
    scores : np.ndarray of shape (n_timesteps,)
        Raw stability scores from the beginning of a session.
    window : int
        Number of timesteps to use for baseline computation.

    Returns
    -------
    baseline : SessionBaseline
        Mean and standard deviation for z-score normalization.
    """
    if window <= 0 or len(scores) == 0:
        return SessionBaseline()

    calibration = scores[:window]
    return SessionBaseline(
        mean=float(np.mean(calibration)),
        std=float(max(np.std(calibration), 1e-8)),
        n_samples=len(calibration),
    )


def apply_baseline(
    scores: np.ndarray,
    baseline: SessionBaseline,
    clip_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Apply per-session z-score normalization to stability scores.

    Transforms raw scores to z-scores using the session baseline, then
    maps to [0, 1] via sigmoid-like rescaling.

    Parameters
    ----------
    scores : np.ndarray of shape (n_timesteps,)
        Raw stability scores.
    baseline : SessionBaseline
        Baseline statistics computed from session start.
    clip_range : tuple[float, float]
        Output range.

    Returns
    -------
    calibrated : np.ndarray of shape (n_timesteps,)
        Baseline-calibrated scores in clip_range.
    """
    if not baseline.is_calibrated:
        return _clip_scores(scores, clip_range)

    # Z-score normalization
    z = (scores - baseline.mean) / baseline.std

    # Sigmoid mapping: z ∈ (-∞, ∞) → (0, 1)
    # Centers at baseline mean with unit std
    calibrated = 1.0 / (1.0 + np.exp(-z))

    return _clip_scores(calibrated, clip_range)


def score_inverse_variance(
    pfc_vertices: np.ndarray,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    """Compute stability score based on inverse PFC variance.

    Lower variance across PFC vertices at each timestep indicates
    more uniform (stable) prefrontal activity.

    Parameters
    ----------
    pfc_vertices : np.ndarray of shape (n_timesteps, n_pfc_vertices)
        PFC vertex predictions from pfc_roi.extract_pfc_vertices().
    config : ScoringConfig or None
        Scoring configuration. Uses defaults if None.

    Returns
    -------
    scores : np.ndarray of shape (n_timesteps,)
        Stability scores ∈ [0, 1]. Higher = more stable.
    """
    if config is None:
        config = ScoringConfig()

    if pfc_vertices.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_timesteps, n_vertices), "
            f"got shape {pfc_vertices.shape}"
        )

    # Per-timestep variance across PFC vertices
    variance = np.var(pfc_vertices, axis=1)

    # Inverse variance with scaling
    scores = 1.0 / (1.0 + variance * config.variance_scale)

    return _clip_scores(scores, config.clip_range)


def score_activation_ratio(
    pfc_vertices: np.ndarray,
    whole_brain: np.ndarray,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    """Compute stability score based on PFC-to-whole-brain activation ratio.

    Higher PFC activation relative to whole-brain baseline indicates
    stronger prefrontal engagement.

    Parameters
    ----------
    pfc_vertices : np.ndarray of shape (n_timesteps, n_pfc_vertices)
        PFC vertex predictions.
    whole_brain : np.ndarray of shape (n_timesteps, n_total_vertices)
        Full cortical predictions (20484 vertices).
    config : ScoringConfig or None
        Scoring configuration.

    Returns
    -------
    scores : np.ndarray of shape (n_timesteps,)
        Activation ratio scores normalized to [0, 1].
    """
    if config is None:
        config = ScoringConfig()

    if pfc_vertices.ndim != 2 or whole_brain.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays")

    if pfc_vertices.shape[0] != whole_brain.shape[0]:
        raise ValueError(
            f"Timestep mismatch: pfc={pfc_vertices.shape[0]}, "
            f"brain={whole_brain.shape[0]}"
        )

    # Mean activation per timestep
    pfc_mean = np.mean(np.abs(pfc_vertices), axis=1)
    brain_mean = np.mean(np.abs(whole_brain), axis=1)

    # Avoid division by zero
    brain_mean = np.maximum(brain_mean, 1e-10)

    # Raw ratio
    ratio = pfc_mean / brain_mean

    # Normalize to [0, 1]
    return _normalize_to_unit(ratio, config.clip_range)


def _compute_temporal_stability(
    pfc_vertices: np.ndarray, window: int = 5
) -> np.ndarray:
    """Compute frame-to-frame temporal stability of PFC activity.

    Lower jitter (frame-to-frame change) = higher temporal stability.

    Parameters
    ----------
    pfc_vertices : np.ndarray of shape (n_timesteps, n_pfc_vertices)
        PFC vertex predictions.
    window : int
        Rolling window size for stability calculation.

    Returns
    -------
    stability : np.ndarray of shape (n_timesteps,)
        Temporal stability scores ∈ [0, 1]. Higher = more stable.
    """
    n_timesteps = pfc_vertices.shape[0]
    if n_timesteps < 2:
        return np.ones(n_timesteps)

    # Frame-to-frame absolute difference (jitter)
    diffs = np.mean(np.abs(np.diff(pfc_vertices, axis=0)), axis=1)

    # Pad first timestep (no previous frame to compare)
    jitter = np.concatenate([[diffs[0]], diffs])

    # Rolling average of jitter
    if window > 1 and n_timesteps >= window:
        kernel = np.ones(window) / window
        jitter_smooth = np.convolve(jitter, kernel, mode="same")
    else:
        jitter_smooth = jitter

    # Inverse jitter → stability
    max_jitter = jitter_smooth.max()
    if max_jitter < 1e-10:
        return np.ones(n_timesteps)

    stability = 1.0 - (jitter_smooth / max_jitter)
    return np.clip(stability, 0.0, 1.0)


def _gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a 1D Gaussian kernel for temporal smoothing.

    Parameters
    ----------
    size : int
        Kernel size (must be odd).
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    kernel : np.ndarray of shape (size,)
        Normalized Gaussian kernel.
    """
    if size % 2 == 0:
        size += 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def temporal_smooth(
    scores: np.ndarray,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    """Apply Gaussian temporal smoothing to stability scores.

    Reduces per-timestep noise while preserving significant state changes.
    With asymmetric smoothing enabled (default), rapid drops are preserved
    to ensure the state machine can detect acute stability decreases.

    Parameters
    ----------
    scores : np.ndarray of shape (n_timesteps,)
        Raw or calibrated stability scores.
    config : ScoringConfig or None
        Configuration with smoothing parameters.

    Returns
    -------
    smoothed : np.ndarray of shape (n_timesteps,)
        Smoothed stability scores.
    """
    if config is None:
        config = ScoringConfig()

    if config.smoothing_kernel <= 0:
        return scores.copy()

    kernel_size = config.smoothing_kernel
    if kernel_size % 2 == 0:
        kernel_size += 1

    if len(scores) < kernel_size:
        return scores.copy()

    kernel = _gaussian_kernel(kernel_size, config.smoothing_sigma)
    smoothed = np.convolve(scores, kernel, mode="same")

    if config.asymmetric_smoothing:
        # Preserve rapid drops: where raw score drops below smoothed,
        # use the raw score instead (keeps acute decreases sharp)
        drops = scores < smoothed
        smoothed[drops] = scores[drops]

    return np.clip(smoothed, config.clip_range[0], config.clip_range[1])



def score_combined(
    pfc_vertices: np.ndarray,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    """Compute combined stability score from inverse variance + temporal stability.

    Weighted combination: alpha * inverse_variance + (1 - alpha) * temporal_stability.

    Parameters
    ----------
    pfc_vertices : np.ndarray of shape (n_timesteps, n_pfc_vertices)
        PFC vertex predictions.
    config : ScoringConfig or None
        Scoring configuration. Uses defaults if None.

    Returns
    -------
    scores : np.ndarray of shape (n_timesteps,)
        Combined stability scores ∈ [0, 1].
    """
    if config is None:
        config = ScoringConfig()

    inv_var = score_inverse_variance(pfc_vertices, config)
    temporal = _compute_temporal_stability(pfc_vertices, config.temporal_window)

    combined = config.combined_alpha * inv_var + (1 - config.combined_alpha) * temporal
    return _clip_scores(combined, config.clip_range)


# Convenience mapping for string-based method selection
SCORING_METHODS: dict[str, tp.Callable] = {
    "inverse_variance": score_inverse_variance,
    "activation_ratio": score_activation_ratio,
    "combined": score_combined,
}


def score_pfc_stability(
    pfc_vertices: np.ndarray,
    method: str = "inverse_variance",
    whole_brain: np.ndarray | None = None,
    config: ScoringConfig | None = None,
    calibrate: bool = True,
    smooth: bool = True,
) -> tuple[np.ndarray, SessionBaseline]:
    """Unified scoring interface with automatic baseline calibration.

    This is the primary entry point for scoring PFC stability. It:
    1. Computes raw scores using the specified method
    2. Calibrates using per-session baseline (if enabled)
    3. Returns calibrated scores and baseline for downstream use

    Parameters
    ----------
    pfc_vertices : np.ndarray of shape (n_timesteps, n_pfc_vertices)
        PFC vertex predictions from pfc_roi.extract_pfc_vertices().
    method : str
        Scoring method: "inverse_variance", "activation_ratio", or "combined".
    whole_brain : np.ndarray or None
        Full cortical predictions. Required for "activation_ratio" method.
    config : ScoringConfig or None
        Scoring configuration.
    calibrate : bool
        Whether to apply per-session baseline calibration.
    smooth : bool
        Whether to apply temporal smoothing.

    Returns
    -------
    scores : np.ndarray of shape (n_timesteps,)
        Calibrated (and optionally smoothed) stability scores ∈ [0, 1].
    baseline : SessionBaseline
        Baseline statistics used for calibration.
    """
    if config is None:
        config = ScoringConfig()

    if method not in SCORING_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(SCORING_METHODS.keys())}"
        )

    # Compute raw scores
    if method == "activation_ratio":
        if whole_brain is None:
            raise ValueError(
                "activation_ratio method requires whole_brain argument"
            )
        raw_scores = score_activation_ratio(pfc_vertices, whole_brain, config)
    else:
        raw_scores = SCORING_METHODS[method](pfc_vertices, config)

    # Baseline calibration
    baseline = SessionBaseline()
    if calibrate and config.baseline_window > 0:
        baseline = compute_baseline(raw_scores, config.baseline_window)
        scores = apply_baseline(raw_scores, baseline, config.clip_range)
        LOGGER.info(
            "Baseline calibration: mean=%.4f, std=%.4f, window=%d",
            baseline.mean,
            baseline.std,
            baseline.n_samples,
        )
    else:
        scores = raw_scores

    # Temporal smoothing
    if smooth and config.smoothing_kernel > 0:
        scores = temporal_smooth(scores, config)

    return scores, baseline
