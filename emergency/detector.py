"""Emergency vehicle detection + approach verification.

Detecting the class (ambulance/fire_truck/police_vehicle) is not enough —
this module must confirm the vehicle is on an incoming road, moving toward
the intersection, and actually relevant (not moving away, not parked).
Class detection itself reuses detection/detector.py (the emergency classes
are already part of the model's label set); this module adds the
approach-relevance filter on top.
"""

from __future__ import annotations

from counting.roi import ROADS


def intersection_center(roi_config: dict) -> tuple[float, float]:
    """Approximation of the intersection center from the config's geometry.

    Computed as the centroid of all four roads' counting-line endpoints. This
    is purely config-driven (no hardcoded pixels): whichever approach the
    lines bound, the center lands between them. Falls back to the centroid of
    every ROAD roi vertex if no counting lines are configured.
    """
    pts: list[tuple[float, float]] = []
    for road in ROADS:
        line = roi_config["roads"][road]["counting_line"]
        if len(line) == 2:
            pts.append((float(line[0][0]), float(line[0][1])))
            pts.append((float(line[1][0]), float(line[1][1])))
    if pts:
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )
    # Fallback: centroid of all road ROI polygon vertices.
    pts = []
    for road in ROADS:
        for vertex in roi_config["roads"][road]["roi"]:
            pts.append((float(vertex[0]), float(vertex[1])))
    if not pts:
        raise ValueError("configs/intersection.yaml has no geometry to infer a center from")
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _is_moving_toward(
    prev_pos: tuple[float, float],
    current_pos: tuple[float, float],
    center_xy: tuple[float, float],
) -> bool:
    """True if the vehicle's displacement this frame points toward ``center``.

    Uses the dot product of the displacement vector with the vector from the
    vehicle's position to the center. A positive dot means the vehicle moved
    at least partly toward the intersection; a parked-but-jittering vehicle
    (tiny displacement) will have a near-zero dot and return False.
    """
    dx = current_pos[0] - prev_pos[0]
    dy = current_pos[1] - prev_pos[1]
    if dx == dy == 0.0:
        return False  # no movement at all
    vx = center_xy[0] - current_pos[0]
    vy = center_xy[1] - current_pos[1]
    return dx * vx + dy * vy > 0.0


def is_approaching_intersection(tracked_vehicle, roi_config) -> bool:
    """Return True if ``tracked_vehicle`` can be verified as actively
    approaching the intersection.

    Requirements (all must hold, conservative by design):
      * the vehicle is attributed to a road (counting/roi.py) and has a
        previous position (so we can measure its motion);
      * the movement this frame is toward the intersection center
        (``_distance_is_positive``), not away from it and not parked.
    A brand-new or parked vehicle fails these checks and is never treated as
    an active emergency approach — emergency priority must not fire on
    ambiguous evidence.
    """
    if tracked_vehicle.road not in ROADS:
        return False
    current = getattr(tracked_vehicle, "current_position", None)
    previous = getattr(tracked_vehicle, "previous_position", None)
    if current is None or previous is None:
        return False
    center = intersection_center(roi_config)
    return _is_moving_toward(previous, current, center)
