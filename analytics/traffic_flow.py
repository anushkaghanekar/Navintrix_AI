"""Traffic flow: vehicles crossing the counting line per unit time, per road.

FlowTracker keeps per-road timestamps of line crossings and reports
vehicles-per-minute over a sliding window. Crossing events come from
counting/counter.py + counting/roi.py (the line-crossing hook); this class
only does the time-windowing bookkeeping.
"""

from __future__ import annotations

ROADS = ("north", "south", "east", "west")


class FlowTracker:
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._crossing_timestamps: dict[str, list[float]] = {
            road: [] for road in ROADS
        }

    def on_crossing(self, road: str, timestamp: float) -> None:
        """Append a crossing timestamp, drop anything older than
        window_seconds (so current_flow reflects a true sliding window)."""
        if road not in self._crossing_timestamps:
            # Crossings on unknown roads still count; they just aren't
            # aggregated into the standard road keys.
            self._crossing_timestamps[road] = []
        ts = float(timestamp)
        cutoff = ts - self.window_seconds
        timestamps = self._crossing_timestamps[road]
        timestamps.append(ts)
        # Timestamps arrive in order; drop from the front while stale.
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

    def current_flow(self, road: str, now: float | None = None) -> float:
        """Return crossings-per-minute over the current window for ``road``.

        ``now`` is the query time used to define the sliding window:
        only crossings in ``[now - window_seconds, now]`` count. If omitted,
        the window is anchored at the most recent crossing. Either way, stale
        timestamps are pruned so the list does not grow unboundedly.

        Empty window -> 0.0. With a part-filled window this is the
        extrapolated per-minute rate. Unknown road -> 0.0.
        """
        timestamps = self._crossing_timestamps.get(road, [])
        if not timestamps or self.window_seconds <= 0:
            return 0.0
        reference = float(now) if now is not None else timestamps[-1]
        cutoff = reference - self.window_seconds
        # Mutate the list in place: drop events that fell out of the window.
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
        return len(timestamps) * (60.0 / self.window_seconds)
