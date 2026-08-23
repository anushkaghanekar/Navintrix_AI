"""Signal safety state machine — the authoritative core of the whole system.

Nothing else in this codebase — not the adaptive controller, not emergency
priority, not a future RL agent — may set a signal phase directly. Every
phase change goes through here, which enforces min/max green, yellow, and
all-red durations and refuses any transition that would create a
conflicting green.

This is the single most important module to get right and to test
thoroughly (see tests/) — everything else is optimization on top of it.
"""

from enum import Enum, auto


class SignalPhase(Enum):
    GREEN = auto()
    YELLOW = auto()
    ALL_RED = auto()


class SafetyStateMachine:
    def __init__(self, signal_config: dict):
        """signal_config: the `signal:` block from configs/signal.yaml
        (min_green_seconds, max_green_seconds, yellow_seconds, all_red_seconds).

        TODO: initialize current phase/road, phase start time, and whatever
        state you need to enforce min/max green server-side (not just
        trusting the caller).
        """
        raise NotImplementedError

    def request_phase_change(self, requested_road: str, timestamp: float, emergency: bool = False) -> bool:
        """Request that `requested_road` get the next green.

        TODO: refuse the request outright if it would conflict with the
        current phase's safety constraints (min green not yet met, unless
        `emergency` and even then only if switching now doesn't skip
        yellow/all-red). Otherwise begin the
        GREEN -> YELLOW -> ALL_RED -> new GREEN sequence. Return whether
        the request was accepted.
        """
        raise NotImplementedError

    def tick(self, timestamp: float) -> SignalPhase:
        """Advance internal timers; call every control loop iteration.

        TODO: enforce max_green_seconds (force a transition if exceeded),
        advance yellow_seconds -> all_red_seconds -> next green
        automatically once a transition has started.
        """
        raise NotImplementedError
