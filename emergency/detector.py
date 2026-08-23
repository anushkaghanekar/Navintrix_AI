"""Emergency vehicle detection + approach verification.

Detecting the class (ambulance/fire_truck/police_vehicle) is not enough —
this module must confirm the vehicle is on an incoming road, moving toward
the intersection, and actually relevant (not moving away, not parked).
Class detection itself reuses detection/detector.py; this module adds the
approach-relevance filter on top.
"""


def is_approaching_intersection(tracked_vehicle, roi_config) -> bool:
    """TODO: use tracked_vehicle.road, current/previous position, and the
    road's roi/counting_line to determine whether this vehicle is actually
    approaching the intersection rather than moving away from it or
    stationary off to the side.
    """
    raise NotImplementedError
