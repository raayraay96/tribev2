# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for hysteresis state machine module."""

import numpy as np
import pytest

from tribev2.state_machine import (
    CognitiveState,
    ScaffoldingStateMachine,
    StateMachineConfig,
    StateTransition,
)


class TestStateMachineConfig:
    def test_defaults(self):
        cfg = StateMachineConfig()
        assert cfg.lower_threshold == 0.40
        assert cfg.upper_threshold == 0.55
        assert cfg.sustain_seconds == 3.0

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="lower_threshold"):
            ScaffoldingStateMachine(
                StateMachineConfig(lower_threshold=0.6, upper_threshold=0.4)
            )


class TestBasicTransitions:
    """Test raw transition logic without sustain."""

    def _make_sm(self, sustain_timesteps: int = 1) -> ScaffoldingStateMachine:
        """Create state machine with minimal sustain for testing."""
        return ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=sustain_timesteps)
        )

    def test_initial_state_is_edge(self):
        sm = self._make_sm()
        assert sm.current_state == CognitiveState.EDGE

    def test_high_score_transitions_to_stable(self):
        sm = self._make_sm(sustain_timesteps=1)
        state = sm.update(0.7)
        assert state == CognitiveState.STABLE

    def test_low_score_transitions_to_scaffolding(self):
        sm = self._make_sm()
        state = sm.update(0.2)
        assert state == CognitiveState.SCAFFOLDING

    def test_mid_score_stays_edge(self):
        sm = self._make_sm()
        state = sm.update(0.48)
        assert state == CognitiveState.EDGE

    def test_downward_is_immediate(self):
        sm = self._make_sm(sustain_timesteps=10)
        # Get to STABLE first
        for _ in range(15):
            sm.update(0.7)
        assert sm.current_state == CognitiveState.STABLE

        # Single downward drop should be immediate
        state = sm.update(0.2)
        assert state == CognitiveState.SCAFFOLDING

    def test_transition_callback(self):
        transitions = []
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=1),
            on_transition=lambda t: transitions.append(t),
        )
        sm.update(0.7)  # EDGE -> STABLE
        assert len(transitions) == 1
        assert transitions[0].from_state == CognitiveState.EDGE
        assert transitions[0].to_state == CognitiveState.STABLE


class TestHysteresis:
    """Test sustain-based oscillation prevention."""

    def test_sustain_prevents_immediate_upward(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=3)
        )
        # Drop to SCAFFOLDING
        sm.update(0.2)
        assert sm.current_state == CognitiveState.SCAFFOLDING

        # Only 1 timestep above threshold — should NOT transition
        sm.update(0.5)
        assert sm.current_state == CognitiveState.SCAFFOLDING

    def test_sustain_allows_after_enough_timesteps(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=3)
        )
        # Drop to SCAFFOLDING
        sm.update(0.2)
        assert sm.current_state == CognitiveState.SCAFFOLDING

        # 3 consecutive timesteps in EDGE territory
        sm.update(0.45)
        sm.update(0.48)
        state = sm.update(0.50)
        assert state == CognitiveState.EDGE

    def test_no_oscillation_with_noisy_signal(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=5)
        )
        # Alternating signal that should NOT cause rapid transitions
        transitions_before = len(sm.transitions)
        for score in [0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3]:
            sm.update(score)

        # Should have very few transitions due to hysteresis
        new_transitions = len(sm.transitions) - transitions_before
        assert new_transitions < 8, (
            f"Too many transitions ({new_transitions}) for oscillating signal"
        )

    def test_sustain_resets_on_interruption(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=3)
        )
        sm.update(0.2)  # -> SCAFFOLDING
        assert sm.current_state == CognitiveState.SCAFFOLDING

        # Start sustain for EDGE
        sm.update(0.45)
        sm.update(0.48)

        # Interrupt with a drop
        sm.update(0.2)
        assert sm.current_state == CognitiveState.SCAFFOLDING

        # Sustain should have reset — need full 3 again
        sm.update(0.45)
        assert sm.current_state == CognitiveState.SCAFFOLDING


class TestBatchProcessing:
    def test_batch_returns_states(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=3)
        )
        scores = np.array([0.6, 0.55, 0.45, 0.3, 0.35, 0.5, 0.6, 0.65])
        states, transitions = sm.process_batch(scores)
        assert len(states) == len(scores)
        assert all(isinstance(s, CognitiveState) for s in states)

    def test_batch_deterministic(self):
        cfg = StateMachineConfig(sustain_timesteps=2)
        scores = np.random.RandomState(42).rand(100)

        sm1 = ScaffoldingStateMachine(cfg)
        states1, _ = sm1.process_batch(scores)

        sm2 = ScaffoldingStateMachine(cfg)
        states2, _ = sm2.process_batch(scores)

        assert states1 == states2

    def test_batch_transitions_recorded(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=1)
        )
        # Clear signal: high -> low -> high
        scores = np.array(
            [0.7] * 5 + [0.2] * 5 + [0.7] * 5
        )
        states, transitions = sm.process_batch(scores)
        assert len(transitions) > 0


class TestSummary:
    def test_summary_structure(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=1)
        )
        sm.update(0.7)
        sm.update(0.2)
        sm.update(0.7)

        summary = sm.get_summary()
        assert "current_state" in summary
        assert "total_transitions" in summary
        assert "timesteps_processed" in summary
        assert "state_durations" in summary
        assert "transitions" in summary
        assert summary["timesteps_processed"] == 3

    def test_state_durations(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=1)
        )
        # 5 timesteps in STABLE
        for _ in range(5):
            sm.update(0.7)
        # 3 timesteps in SCAFFOLDING
        for _ in range(3):
            sm.update(0.2)

        summary = sm.get_summary()
        durations = summary["state_durations"]
        # EDGE initial + transitions
        assert durations["STABLE"] >= 4
        assert durations["SCAFFOLDING"] >= 2


class TestReset:
    def test_reset_clears_state(self):
        sm = ScaffoldingStateMachine(
            StateMachineConfig(sustain_timesteps=1)
        )
        sm.update(0.7)
        sm.update(0.2)
        assert sm.timestep == 2
        assert len(sm.transitions) > 0

        sm.reset()
        assert sm.current_state == CognitiveState.EDGE
        assert sm.timestep == 0
        assert len(sm.transitions) == 0
