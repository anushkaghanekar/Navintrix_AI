"""Anti-starvation tracking: a low-traffic road must not wait forever just
because its priority_score is consistently lower than the other roads.

The controller consults FairnessTracker every cycle: a road that has been
waiting longer than ``max_wait_seconds`` since its last green is forced to
the front of the queue, overriding the raw adaptive score. This is the
anti-starvation ceiling referenced in configs/signal.yaml.
"""

from __future__ import annotations

DEFAULT_ROADS = ("north", "east", "south", "west")


class FairnessTracker:
    def __init__(self, max_wait_seconds: float, roads: tuple[str, ...] | None = None):
        if max_wait_seconds <= 0:
            raise ValueError(f"max_wait_seconds must be > 0, got {max_wait_seconds}")
        self.max_wait_seconds = float(max_wait_seconds)
        self._roads = tuple(roads) if roads is not None else DEFAULT_ROADS
        self._last_green_time: dict[str, float] = {road: 0.0 for road in self._roads}

    def on_green_granted(self, road: str, timestamp: float) -> None:
        """Record that ``road`` received green at ``timestamp``."""
        self._last_green_time[road] = float(timestamp)

    def road_forcing_green(self, timestamp: float) -> str | None:
        """Return the road whose wait since last green exceeds
        max_wait_seconds, if any (pick the worst offender if more than one).

        A road is "forcing" when (timestamp - last_green_time[road]) >
        max_wait_seconds. Among multiple, the one that has overshot by the
        most is returned. None if nobody has exceeded the ceiling.
        """
        worst_road: str | None = None
        worst_overshoot = 0.0
        now = float(timestamp)
        for road in self._roads:
            waited = now - self._last_green_time[road]
            overshoot = waited - self.max_wait_seconds
            if overshoot > worst_overshoot:
                worst_overshoot = overshoot
                worst_road = road
        return worst_road if worst_overshoot > 0 else None

    def reset(self) -> None:
        self._last_green_time = {road: 0.0 for road in self._roads}
