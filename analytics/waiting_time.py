"""Per-vehicle and per-road waiting time.

A vehicle's waiting time starts accumulating once it enters the queue
region and stops when it crosses the counting line (departs). Aggregate
into average / max / per-road numbers for the controller and the report.
"""


class WaitingTimeTracker:
    def __init__(self):
        self._queue_entry_times: dict[int, float] = {}   # track_id -> timestamp

    def on_enter_queue(self, track_id: int, timestamp: float) -> None:
        """TODO: record queue entry time if not already recorded."""
        raise NotImplementedError

    def on_depart(self, track_id: int, timestamp: float) -> float:
        """TODO: compute and return this vehicle's total waiting time,
        then clear its entry from _queue_entry_times.
        """
        raise NotImplementedError

    def road_average(self, road: str) -> float:
        """TODO: average waiting time across vehicles currently/recently
        waiting on `road`."""
        raise NotImplementedError
