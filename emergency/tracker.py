"""Emergency-vehicle-specific tracking state.

Wraps tracking/bytetrack.py's TrackedVehicle with the extra fields needed
for emergency handling: distance_to_intersection, whether it's currently
"active" (approaching, relevant), and when it clears the intersection
(needed to trigger EMERGENCY_PASSED in the state machine).

Clearance is a persistence check: the vehicle must be seen as NOT actively
approaching for N consecutive frames (configs/signal.yaml's
``emergency.clearance_confirmation_frames``) before we declare it cleared.
A single dropped frame must not end emergency handling prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass

from emergency.detector import intersection_center, is_approaching_intersection


@dataclass
class EmergencyVehicleState:
    track_id: int
    cls: str
    road: str
    movement: str | None
    distance_to_intersection: float | None
    approaching_intersection: bool
    cleared: bool = False
    frames_not_approaching: int = 0  # internal counter, not part of ctor


def _distance(center, position) -> float:
    return ((position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2) ** 0.5


class EmergencyTracker:
    """Progresses EmergencyVehicleState for each track across frames."""

    def __init__(self, clearance_confirmation_frames: int):
        if clearance_confirmation_frames < 1:
            raise ValueError(
                f"clearance_confirmation_frames must be >= 1, got {clearance_confirmation_frames}"
            )
        self.clearance_confirmation_frames = int(clearance_confirmation_frames)
        self._states: dict[int, EmergencyVehicleState] = {}

    @classmethod
    def from_config(cls, signal_config: dict) -> "EmergencyTracker":
        """Build from configs/signal.yaml's 'emergency' block."""
        frames = signal_config["emergency"]["clearance_confirmation_frames"]
        return cls(frames)

    def update(self, tracked_vehicle, roi_config) -> EmergencyVehicleState:
        """Build/update the state for one tracked emergency vehicle.

        Recomputes approach + distance from the tracked positions and ROIs,
        and maintains the consecutive-frames "not approaching" counter used
        to decide clearance.
        """
        track_id = tracked_vehicle.track_id
        center = intersection_center(roi_config)
        approaching = is_approaching_intersection(tracked_vehicle, roi_config)

        cur = getattr(tracked_vehicle, "current_position", None)
        distance = _distance(center, cur) if cur is not None else None

        prev_state = self._states.get(track_id)
        if prev_state is None:
            frames_not_approaching = 0
        else:
            frames_not_approaching = 0 if approaching else (prev_state.frames_not_approaching + 1)

        cleared = frames_not_approaching >= self.clearance_confirmation_frames

        state = EmergencyVehicleState(
            track_id=track_id,
            cls=tracked_vehicle.cls,
            road=tracked_vehicle.road,
            movement=getattr(tracked_vehicle, "movement", None),
            distance_to_intersection=distance,
            approaching_intersection=approaching,
            cleared=cleared,
            frames_not_approaching=frames_not_approaching,
        )
        self._states[track_id] = state
        return state

    def clear(self, track_id: int) -> None:
        """Drop a track's state (e.g. when the vehicle fully leaves)."""
        self._states.pop(track_id, None)

    @property
    def active_states(self) -> list[EmergencyVehicleState]:
        return [s for s in self._states.values() if s.approaching_intersection]
