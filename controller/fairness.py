"""Anti-starvation tracking: a low-traffic road must not wait forever just
because its priority_score is consistently lower than the other roads.
"""


class FairnessTracker:
    def __init__(self, max_wait_seconds: float):
        self.max_wait_seconds = max_wait_seconds
        self._last_green_time: dict[str, float] = {
            "north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0,
        }

    def on_green_granted(self, road: str, timestamp: float) -> None:
        self._last_green_time[road] = timestamp

    def road_forcing_green(self, timestamp: float) -> str | None:
        """TODO: return the road whose wait since last green exceeds
        max_wait_seconds, if any (pick the worst offender if more than one).
        """
        raise NotImplementedError
