"""Traffic flow: vehicles crossing the counting line per unit time, per road."""


class FlowTracker:
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._crossing_timestamps: dict[str, list[float]] = {
            "north": [], "south": [], "east": [], "west": [],
        }

    def on_crossing(self, road: str, timestamp: float) -> None:
        """TODO: append timestamp, drop anything older than window_seconds."""
        raise NotImplementedError

    def current_flow(self, road: str) -> float:
        """TODO: return crossings-per-minute over the current window for `road`."""
        raise NotImplementedError
