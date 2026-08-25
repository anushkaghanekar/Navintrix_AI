"""Signal safety state machine — the authoritative core of the whole system.

Nothing else in this codebase — not the adaptive controller, not emergency
priority, not a future RL agent — may set a signal phase directly. Every
phase change goes through here, which enforces min/max green, yellow, and
all-red durations and refuses any transition that would create a
conflicting green.

The model
---------
Green is granted to exactly ONE road at a time, and the only legal state
flow is

    GREEN(road A) -> YELLOW -> ALL_RED -> GREEN(road B)

so a conflicting green (two roads' greens overlapping in time) is
*structurally impossible*, not merely "checked": the transition sequence is
the only path between green grants, and only after all-red does any road
become green. This is the strongest safety guarantee we can offer and the
easiest to defend in front of a reviewer.

Semantics
---------
* request_phase_change(A): refused (returns False) if A is already green, if
  the current green has been up for less than min_green_seconds (unless
  emergency=True, which bypasses min but never the yellow/all-red sequence),
  or if a transition sequence is already committed. Otherwise it commits and
  starts GREEN -> YELLOW -> ALL_RED -> GREEN(A).
* tick(): advances timers; the max_green timeout FORCES a transition even
  with no incoming request (rotating to the next approach road). YELLOW
  lasts yellow_seconds, ALL_RED all_red_seconds, then the pending road goes
  green with its timers reset.
"""

from __future__ import annotations

from enum import Enum, auto

DEFAULT_ROADS = ("north", "east", "south", "west")


class SignalPhase(Enum):
    GREEN = auto()
    YELLOW = auto()
    ALL_RED = auto()


def _validate_timers(signal_config: dict) -> None:
    keys = (
        "min_green_seconds",
        "max_green_seconds",
        "yellow_seconds",
        "all_red_seconds",
    )
    missing = [k for k in keys if k not in signal_config]
    if missing:
        raise ValueError(f"signal_config missing required key(s): {', '.join(missing)}")
    for key in keys:
        value = signal_config[key]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"signal_config.{key} must be a positive number, got {value!r}")
class SafetyStateMachine:
    def __init__(
        self,
        signal_config: dict,
        roads: tuple[str, ...] | None = None,
    ):
        """signal_config: the `signal:` block from configs/signal.yaml
        (min_green_seconds, max_green_seconds, yellow_seconds, all_red_seconds).

        ``roads`` optionally overrides the approach set (used by tests and by
        SUMO-built intersections); defaults to (north, east, south, west).
        The machine starts with the first road green at t=0.
        """
        _validate_timers(signal_config)
        self.min_green_seconds = float(signal_config["min_green_seconds"])
        self.max_green_seconds = float(signal_config["max_green_seconds"])
        self.yellow_seconds = float(signal_config["yellow_seconds"])
        self.all_red_seconds = float(signal_config["all_red_seconds"])

        self._roads = tuple(roads) if roads is not None else DEFAULT_ROADS
        if not self._roads:
            raise ValueError("roads must be non-empty")
        self._rotation = list(self._roads)

        # State: start with the first approach road green at t=0.
        self._phase = SignalPhase.GREEN
        self._current_road = self._roads[0]
        self._pending_road = None
        self.phase_start_t = 0.0
        self.green_start_t = 0.0

    # -- OBSERVERS --

    @property
    def phase(self) -> SignalPhase:
        """The current global signal phase (GREEN/YELLOW/ALL_RED)."""
        return self._phase

    def green_roads(self) -> list[str]:
        """Roads currently showing green (0 or 1 entries — the invariant
        that makes conflicting greens structurally impossible)."""
        if self._phase == SignalPhase.GREEN:
            return [self._current_road]
        return []

    def current_green_road(self) -> str | None:
        return self._current_road if self._phase == SignalPhase.GREEN else None

    def pending_road(self) -> str | None:
        """The road locked in to receive green after the current transition."""
        return self._pending_road

    def outgoing_road(self) -> str | None:
        """The approach now exiting green, or the current green road during
        GREEN. During YELLOW/ALL_RED this is the road that held green before
        the transition — used to pick which SUMO transition phase to show."""
        return self._current_road

    # -- REQUEST API --

    def request_phase_change(
        self,
        requested_road: str,
        timestamp: float,
        emergency: bool = False,
    ) -> bool:
        """Request that ``requested_road`` receive the next green.

        Refused (returns False) when: the road is currently green; we are
        mid-transition (a committed yellow/all-red cannot be reversed);
        or min_green_seconds has not yet elapsed and this is not an
        emergency. Emergency skips only the min-green wait — never yellow
        or all-red. On acceptance, the fixed
        GREEN -> YELLOW -> ALL_RED -> GREEN(requested) sequence begins.
        """
        if requested_road not in self._roads:
            raise ValueError(f"unknown road {requested_road!r} (known: {self._roads})")

        if self._phase == SignalPhase.GREEN and requested_road == self._current_road:
            return False

        if self._phase != SignalPhase.GREEN:
            return False

        elapsed = float(timestamp) - self.green_start_t
        if not emergency and elapsed < self.min_green_seconds:
            return False

        self._pending_road = requested_road
        self._phase = SignalPhase.YELLOW
        self.phase_start_t = float(timestamp)
        return True

    def tick(self, timestamp: float) -> SignalPhase:
        """Advance internal timers; call every control loop iteration.

        Enforces max_green_seconds (forcing a rotation once exceeded even
        with no request), then yellow_seconds -> all_red_seconds -> next
        green once a transition has started. Returns the current phase.
        """
        self._advance(float(timestamp))
        return self._phase

    def _advance(self, now: float) -> None:
        elapsed = now - self.phase_start_t
        if self._phase == SignalPhase.GREEN:
            if elapsed >= self.max_green_seconds:
                self._pending_road = self._next_rotation(self._current_road)
                self._set_phase(SignalPhase.YELLOW, now)
        elif self._phase == SignalPhase.YELLOW and elapsed >= self.yellow_seconds:
            self._set_phase(SignalPhase.ALL_RED, now)
        elif self._phase == SignalPhase.ALL_RED and elapsed >= self.all_red_seconds:
            target = self._pending_road or self._next_rotation(self._current_road)
            self._current_road = target
            self._pending_road = None
            self._set_phase(SignalPhase.GREEN, now)

    def _set_phase(self, phase: SignalPhase, now: float) -> None:
        self._phase = phase
        self.phase_start_t = now
        if phase == SignalPhase.GREEN:
            self.green_start_t = now

    def _next_rotation(self, road: str) -> str:
        idx = self._rotation.index(road)
        return self._rotation[(idx + 1) % len(self._rotation)]
