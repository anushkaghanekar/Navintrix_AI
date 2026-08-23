"""Trajectory-based vehicle counting.

IMPORTANT: count on counting-line crossing, not on per-frame detections —
a vehicle present in 50 frames must add 1 to the count, not 50. Keep a set
of track_ids already counted per road to prevent duplicates.
"""


class VehicleCounter:
    def __init__(self):
        self._counted_track_ids: set[int] = set()
        self.counts_by_road = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.counts_by_class: dict[str, int] = {}

    def update(self, tracked_vehicle, crossed_counting_line: bool) -> None:
        """Call once per tracked vehicle per frame.

        TODO: if crossed_counting_line and track_id not already counted,
        increment counts_by_road[tracked_vehicle.road] and
        counts_by_class[tracked_vehicle.cls], and mark the track_id counted.
        `crossed_counting_line` should come from comparing previous/current
        position against configs/intersection.yaml's counting_line.
        """
        raise NotImplementedError
