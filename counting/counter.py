"""Trajectory-based vehicle counting.

IMPORTANT: count on counting-line crossing, not on per-frame detections —
a vehicle present in 50 frames must add 1 to the count, not 50. Keep a set
of track_ids already counted per road to prevent duplicates.

The crossing signal comes in as the ``crossed_counting_line`` boolean; the
geometry itself (previous/current position vs. the road's counting line)
lives in counting/roi.py so this module stays pure bookkeeping.
"""

from __future__ import annotations


class VehicleCounter:
    def __init__(self):
        """Fresh counter: no roads or classes counted yet, empty tracking
        set — a track_id already counted on a road is never counted again
        unless reset() is called."""
        self._counted_track_ids: set[int] = set()
        # Road whose counting line each counted track performed its first
        # (and only) crossing on — captures "arrival" direction.
        self._counted_roads: dict[int, str] = {}
        self.counts_by_road = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.counts_by_class: dict[str, int] = {}

    def update(self, tracked_vehicle, crossed_counting_line: bool) -> dict:
        """Record one tracked vehicle for one frame.

        If ``crossed_counting_line`` is True and this track_id has not been
        counted yet, increment counts_by_road[vehicle.road] (defaulting to
        an "unassigned" key if road is None) and counts_by_class[cls], and
        mark the ID counted so 50 frames of the same vehicle count as 1.

        Returns self.counts_by_road so callers can read the running totals
        without reaching into internals.
        """
        if not crossed_counting_line or tracked_vehicle.track_id in self._counted_track_ids:
            return self.counts_by_road

        self._counted_track_ids.add(tracked_vehicle.track_id)
        road = tracked_vehicle.road or "unassigned"
        self._counted_roads[tracked_vehicle.track_id] = road
        self.counts_by_road[road] = self.counts_by_road.get(road, 0) + 1
        self.counts_by_class[tracked_vehicle.cls] = (
            self.counts_by_class.get(tracked_vehicle.cls, 0) + 1
        )
        return self.counts_by_road

    def reset(self) -> None:
        """Clear all counts and tracked-vehicle state (e.g. new video)."""
        self._counted_track_ids = set()
        self._counted_roads = {}
        self.counts_by_road = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.counts_by_class = {}
