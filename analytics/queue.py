"""Queue length estimation.

Counts tracked vehicles currently inside a road's queue_roi (from
configs/intersection.yaml) with near-zero velocity. Physical (meter-based)
queue length via camera calibration is a stretch goal — start with a
vehicle count.

Velocity is the per-frame displacement between a vehicle's previous and
current centers (pixels/frame). A vehicle is "stopped" when that
displacement is below ``velocity_threshold``: it is genuinely queueing, not
just passing through the region.
"""

from __future__ import annotations

from counting.roi import point_in_polygon


def estimate_queue_length(
    tracked_vehicles: list,
    queue_roi,
    velocity_threshold: float = 1.0,
) -> int:
    """Count vehicles inside ``queue_roi`` whose recent velocity is below
    ``velocity_threshold`` (i.e. stopped/near-stopped, not just passing
    through).

    ``queue_roi`` is the road's queue region as [[x,y], ...] from
    configs/intersection.yaml. ``velocity_threshold`` is in pixels-per-frame
    of displacement between a vehicle's previous and current centers. A
    vehicle with no previous position (just spawned) has no velocity
    evidence and is excluded (avoids counting a newly-entered vehicle as
    queued before its speed is known).

    Vehicle center is tracked_vehicle.current_position and must already be
    attributed to a road (counting/roi.py) for the queue to mean anything.
    """
    count = 0
    if not tracked_vehicles:
        return 0
    polygon = [(float(pt[0]), float(pt[1])) for pt in queue_roi]
    if len(polygon) < 3:
        return 0  # uncalibrated queue region cannot be measured

    for vehicle in tracked_vehicles:
        cur = vehicle.current_position
        if not point_in_polygon(cur[0], cur[1], polygon):
            continue
        prev = getattr(vehicle, "previous_position", None)
        if prev is None:
            continue  # no velocity evidence yet -> not counted as stopped
        displacement = ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2) ** 0.5
        if displacement < velocity_threshold:
            count += 1
    return count
