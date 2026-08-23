"""Queue length estimation.

Counts tracked vehicles currently inside a road's queue_roi (from
configs/intersection.yaml) with near-zero velocity. Physical (meter-based)
queue length via camera calibration is a stretch goal — start with a
vehicle count.
"""


def estimate_queue_length(tracked_vehicles: list, queue_roi, velocity_threshold: float = 1.0) -> int:
    """TODO: count vehicles inside queue_roi whose recent velocity is below
    velocity_threshold (i.e. stopped/near-stopped, not just passing through).
    """
    raise NotImplementedError
