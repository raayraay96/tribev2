# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hysteresis state machine for cognitive scaffolding interventions.

Implements a three-state controller that converts PFC stability scores into
cognitive scaffolding decisions with oscillation prevention via sustain timers.

States:
- STABLE (score > upper threshold): No scaffolding needed
- EDGE (score between thresholds): Monitoring zone, light intervention
- SCAFFOLDING (score < lower threshold): Full scaffolding active

The hysteresis mechanism prevents rapid oscillation between states by requiring
a "sustain" period before transitioning to a more stable state (upward only).
Downward transitions (toward SCAFFOLDING) are immediate to ensure prompt
intervention when stability drops.
"""

import logging
import time
import typing as tp
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

LOGGER = logging.getLogger(__name__)


class CognitiveState(Enum):
    """Cognitive scaffolding states."""

    STABLE = auto()
    EDGE = auto()
    SCAFFOLDING = auto()


# Priority ordering: higher index = more scaffolding needed
_STATE_PRIORITY = {
    CognitiveState.STABLE: 0,
    CognitiveState.EDGE: 1,
    CognitiveState.SCAFFOLDING: 2,
}


@dataclass
class StateTransition:
    """Record of a state transition event."""

    from_state: CognitiveState
    to_state: CognitiveState
    score: float
    timestamp: float
    timestep: int


@dataclass
class StateMachineConfig:
    """Configuration for the hysteresis state machine.

    Parameters
    ----------
    lower_threshold : float
        Scores below this trigger SCAFFOLDING state. Default: 0.40.
    upper_threshold : float
        Scores above this trigger STABLE state. Default: 0.55.
    sustain_seconds : float
        Minimum duration score must remain above threshold before
        transitioning to a more stable state. Default: 3.0.
    sustain_timesteps : int
        Alternative to sustain_seconds: minimum number of consecutive
        timesteps. If > 0, this takes priority over sustain_seconds.
        Default: 0 (use sustain_seconds).
    """

    lower_threshold: float = 0.40
    upper_threshold: float = 0.55
    sustain_seconds: float = 3.0
    sustain_timesteps: int = 0


@dataclass
class StateMachineState:
    """Internal state of the hysteresis state machine."""

    current_state: CognitiveState = CognitiveState.EDGE
    pending_state: CognitiveState | None = None
    sustain_start: float | None = None
    sustain_start_timestep: int | None = None
    consecutive_above: int = 0
    transitions: list[StateTransition] = field(default_factory=list)
    timestep: int = 0


class ScaffoldingStateMachine:
    """Hysteresis state machine for cognitive scaffolding.

    Converts a stream of PFC stability scores into cognitive state decisions
    with oscillation prevention. Downward transitions (toward SCAFFOLDING)
    are immediate; upward transitions require sustain confirmation.

    Parameters
    ----------
    config : StateMachineConfig or None
        Configuration. Uses defaults if None.
    on_transition : callable or None
        Callback function called on each state transition.
        Signature: ``(transition: StateTransition) -> None``

    Examples
    --------
    >>> sm = ScaffoldingStateMachine()
    >>> scores = [0.6, 0.55, 0.45, 0.3, 0.35, 0.5, 0.6, 0.65]
    >>> for score in scores:
    ...     state = sm.update(score)
    ...     print(f"{score:.2f} -> {state.name}")
    """

    def __init__(
        self,
        config: StateMachineConfig | None = None,
        on_transition: tp.Callable[[StateTransition], None] | None = None,
    ):
        self.config = config or StateMachineConfig()
        self.on_transition = on_transition
        self._state = StateMachineState()

        if self.config.lower_threshold >= self.config.upper_threshold:
            raise ValueError(
                f"lower_threshold ({self.config.lower_threshold}) must be < "
                f"upper_threshold ({self.config.upper_threshold})"
            )

    @property
    def current_state(self) -> CognitiveState:
        """Current cognitive state."""
        return self._state.current_state

    @property
    def transitions(self) -> list[StateTransition]:
        """History of all state transitions."""
        return self._state.transitions

    @property
    def timestep(self) -> int:
        """Current timestep count."""
        return self._state.timestep

    def _classify_score(self, score: float) -> CognitiveState:
        """Classify a score into a raw (non-hysteresis) state."""
        if score >= self.config.upper_threshold:
            return CognitiveState.STABLE
        elif score >= self.config.lower_threshold:
            return CognitiveState.EDGE
        else:
            return CognitiveState.SCAFFOLDING

    def _is_upward_transition(
        self, from_state: CognitiveState, to_state: CognitiveState
    ) -> bool:
        """Check if a transition moves toward more stability."""
        return _STATE_PRIORITY[to_state] < _STATE_PRIORITY[from_state]

    def _is_downward_transition(
        self, from_state: CognitiveState, to_state: CognitiveState
    ) -> bool:
        """Check if a transition moves toward less stability."""
        return _STATE_PRIORITY[to_state] > _STATE_PRIORITY[from_state]

    def _sustain_met(self) -> bool:
        """Check if the sustain criteria are met for an upward transition."""
        if self.config.sustain_timesteps > 0:
            # Timestep-based sustain
            return self._state.consecutive_above >= self.config.sustain_timesteps

        # Time-based sustain
        if self._state.sustain_start is None:
            return False
        elapsed = time.monotonic() - self._state.sustain_start
        return elapsed >= self.config.sustain_seconds

    def _emit_transition(
        self,
        from_state: CognitiveState,
        to_state: CognitiveState,
        score: float,
    ) -> None:
        """Record and emit a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            score=score,
            timestamp=time.monotonic(),
            timestep=self._state.timestep,
        )
        self._state.transitions.append(transition)

        LOGGER.info(
            "State transition: %s -> %s (score=%.3f, t=%d)",
            from_state.name,
            to_state.name,
            score,
            self._state.timestep,
        )

        if self.on_transition is not None:
            self.on_transition(transition)

    def update(self, score: float) -> CognitiveState:
        """Process a new stability score and return the current state.

        Parameters
        ----------
        score : float
            PFC stability score ∈ [0, 1].

        Returns
        -------
        state : CognitiveState
            The current cognitive state after processing this score.
        """
        self._state.timestep += 1
        raw_state = self._classify_score(score)
        current = self._state.current_state

        if raw_state == current:
            # Same state — reset pending
            self._state.pending_state = None
            self._state.sustain_start = None
            self._state.sustain_start_timestep = None
            self._state.consecutive_above = 0
            return current

        if self._is_downward_transition(current, raw_state):
            # Downward transitions are IMMEDIATE (no sustain required)
            self._emit_transition(current, raw_state, score)
            self._state.current_state = raw_state
            self._state.pending_state = None
            self._state.sustain_start = None
            self._state.sustain_start_timestep = None
            self._state.consecutive_above = 0
            return raw_state

        # Upward transition — requires sustain
        if self._state.pending_state != raw_state:
            # New upward target — start sustain timer
            self._state.pending_state = raw_state
            self._state.sustain_start = time.monotonic()
            self._state.sustain_start_timestep = self._state.timestep
            self._state.consecutive_above = 1
        else:
            # Continuing toward same target
            self._state.consecutive_above += 1

        if self._sustain_met():
            # Sustain period complete — commit transition
            self._emit_transition(current, raw_state, score)
            self._state.current_state = raw_state
            self._state.pending_state = None
            self._state.sustain_start = None
            self._state.sustain_start_timestep = None
            self._state.consecutive_above = 0

        return self._state.current_state

    def process_batch(
        self, scores: np.ndarray
    ) -> tuple[list[CognitiveState], list[StateTransition]]:
        """Process a batch of stability scores.

        For batch processing, uses timestep-based sustain instead of
        time-based to ensure deterministic behavior.

        Parameters
        ----------
        scores : np.ndarray of shape (n_timesteps,)
            Sequence of stability scores.

        Returns
        -------
        states : list[CognitiveState]
            State for each timestep.
        transitions : list[StateTransition]
            New transitions that occurred during this batch.
        """
        # Force timestep-based sustain for batch mode
        if self.config.sustain_timesteps == 0:
            # Convert seconds to timesteps (assume ~1 score/second)
            self.config.sustain_timesteps = max(
                1, int(self.config.sustain_seconds)
            )

        initial_transitions = len(self._state.transitions)
        states = []

        for score in scores:
            state = self.update(float(score))
            states.append(state)

        new_transitions = self._state.transitions[initial_transitions:]
        return states, new_transitions

    def reset(self) -> None:
        """Reset the state machine to initial conditions."""
        self._state = StateMachineState()

    def get_summary(self) -> dict[str, tp.Any]:
        """Get a summary of the state machine's history.

        Returns
        -------
        summary : dict
            Keys: 'current_state', 'total_transitions', 'timesteps_processed',
            'state_durations', 'transitions'.
        """
        # Calculate time spent in each state
        durations: dict[str, int] = {s.name: 0 for s in CognitiveState}
        prev_timestep = 0
        prev_state = CognitiveState.EDGE

        for t in self._state.transitions:
            durations[prev_state.name] += t.timestep - prev_timestep
            prev_timestep = t.timestep
            prev_state = t.to_state

        # Add remaining time to current state
        durations[prev_state.name] += self._state.timestep - prev_timestep

        return {
            "current_state": self._state.current_state.name,
            "total_transitions": len(self._state.transitions),
            "timesteps_processed": self._state.timestep,
            "state_durations": durations,
            "transitions": [
                {
                    "from": t.from_state.name,
                    "to": t.to_state.name,
                    "score": round(t.score, 4),
                    "timestep": t.timestep,
                }
                for t in self._state.transitions
            ],
        }
