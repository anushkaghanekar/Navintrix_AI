"""Emergency priority policy — deterministic, not arbitrary.

When multiple emergency vehicles are present (stretch goal — core scope can
assume at most one active at a time), priority must be decided by an
explicit, documented policy (class, direction, distance, estimated arrival,
current phase) — never randomly.

Core policy (at most one active): return the single approaching vehicle.
Stretch policy (multiple): deterministic tie-break —
  * vehicles actively approaching the intersection first;
  * among those, the closest to the intersection wins;
  * ties broken by class priority: ambulance > fire_truck > police_vehicle.
Written down here and replicable from the state list alone.
"""

from __future__ import annotations

CLASS_PRIORITY = {"ambulance": 0, "fire_truck": 1, "police_vehicle": 2}


def select_priority_emergency(active_emergencies: list, current_phase) -> object | None:
    """Return the single highest-priority active emergency vehicle, or None.

    ``active_emergencies`` is a list of EmergencyVehicleState objects whose
    ``approaching_intersection`` flag says whether each is genuinely on an
    active approach. ``current_phase`` is accepted for interface stability
    (an explicit part of the documented policy signature) but this policy
    only needs approach + distance + class.

    With zero states -> None; with states all non-approaching -> None; with
    exactly one approaching -> that one; with several -> deterministic sort.
    """
    approaching = [e for e in active_emergencies if getattr(e, "approaching_intersection", False)]
    if not approaching:
        return None
    if len(approaching) == 1:
        return approaching[0]

    def sort_key(state):
        distance = state.distance_to_intersection
        if distance is None:
            distance = float("inf")
        return (
            # ascending, then class-priority ascending (lower = better)
            distance,
            CLASS_PRIORITY.get(state.cls, 99),
            # final deterministic tie-break so the sort is total/stable
            state.track_id,
        )

    return sorted(approaching, key=sort_key)[0]
