"""Per-vehicle and per-road waiting time.

A vehicle's waiting time starts accumulating once it enters the queue
region and stops when it crosses the counting line (departs). Aggregate
into average / max / per-road numbers for the controller and the report.

Waiting time is recorded on *departure* (the total a queued vehicle spent
waiting) and also kept as a live snapshot while it is still waiting, which
is what the controller reads for anti-starvation.
"""

from __future__ import annotations


class WaitingTimeTracker:
    def __init__(self):
        self._queue_entry_times: dict[int, float] = {}   # track_id -> timestamp
        self._tracked_roads: dict[int, str] = {}          # track_id -> road
        self._departed: dict[str, list[float]] = {        # road -> [wait durations]
            "north": [], "south": [], "east": [], "west": [],
        }

    def on_enter_queue(self, track_id: int, timestamp: float, road: str | None = None) -> None:
        """Record queue entry time if not already recorded.

        ``road`` is an optional hint used for road-scoped aggregates; it is
        safe to pass or omit (it is not required by the classic signature).
        """
        if track_id in self._queue_entry_times:
            return  # already queued
        self._queue_entry_times[track_id] = timestamp
        if road is not None:
            self._tracked_roads[track_id] = road

    def on_depart(self, track_id: int, timestamp: float) -> float:
        """Return this vehicle's total waiting time, then clear its entry.

        Vehicles that were never queued (e.g. a pass-through) have 0 wait.
        """
        entry = self._queue_entry_times.pop(track_id, None)
        road = self._tracked_roads.pop(track_id, None)
        if entry is None:
            return 0.0
        wait = max(0.0, float(timestamp) - entry)
        if road is not None:
            self._departed[road].append(wait)
        return wait

    def current_wait(self, track_id: int, timestamp: float) -> float:
        """Live wait for a still-queued vehicle; 0 if not currently queued."""
        entry = self._queue_entry_times.get(track_id)
        if entry is None:
            return 0.0
        return max(0.0, float(timestamp) - entry)

    def road_average(self, road: str) -> float:
        """Average completed waiting time for ``road`` (0 if none yet)."""
        waits = self._departed.get(road, [])
        return sum(waits) / len(waits) if waits else 0.0

    def road_max(self, road: str) -> float:
        """Max completed waiting time for ``road`` (0 if none yet)."""
        waits = self._departed.get(road, [])
        return max(waits) if waits else 0.0
