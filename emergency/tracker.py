"""Emergency-vehicle-specific tracking state.

Wraps tracking/bytetrack.py's TrackedVehicle with the extra fields needed
for emergency handling: distance_to_intersection, whether it's currently
"active" (approaching, relevant), and when it clears the intersection
(needed to trigger EMERGENCY_PASSED in the state machine).
"""

from dataclasses import dataclass


@dataclass
class EmergencyVehicleState:
    track_id: int
    cls: str
    road: str
    movement: str | None
    distance_to_intersection: float | None
    approaching_intersection: bool
    cleared: bool = False


def update_emergency_state(tracked_vehicle, roi_config) -> EmergencyVehicleState:
    """TODO: build/update an EmergencyVehicleState from a TrackedVehicle,
    including clearance detection (N consecutive frames past the
    intersection zone — see configs/signal.yaml's
    clearance_confirmation_frames).
    """
    raise NotImplementedError
