"""ByteTrack integration.

Takes per-frame Detection lists from detection/detector.py and returns
detections annotated with a persistent track_id. A vehicle visible across
many frames must keep one ID — this is what counting/counter.py depends on
to avoid double-counting.
"""

from dataclasses import dataclass

from detection.detector import Detection


@dataclass
class TrackedVehicle:
    track_id: int
    cls: str
    confidence: float
    current_position: tuple[float, float]
    previous_position: tuple[float, float] | None
    first_seen_time: float
    last_seen_time: float
    road: str | None = None          # filled in by counting/roi.py
    movement: str | None = None      # filled in by counting/movement.py
    queue_entry_time: float | None = None
    waiting_time: float = 0.0


class VehicleTracker:
    def __init__(self, track_buffer: int, match_thresh: float):
        """TODO: initialize the ByteTrack tracker with these params."""
        raise NotImplementedError

    def update(self, detections: list[Detection], timestamp: float) -> list[TrackedVehicle]:
        """Feed one frame's detections in, get tracked vehicles with IDs out.

        TODO: run ByteTrack association, handle temporarily missed
        detections (occlusion) without dropping the track_id immediately.
        """
        raise NotImplementedError
