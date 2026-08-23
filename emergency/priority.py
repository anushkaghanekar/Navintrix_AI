"""Emergency priority policy — deterministic, not arbitrary.

When multiple emergency vehicles are present (stretch goal — core scope can
assume at most one active at a time), priority must be decided by an
explicit, documented policy (class, direction, distance, estimated arrival,
current phase) — never randomly.
"""


def select_priority_emergency(active_emergencies: list, current_phase) -> object | None:
    """TODO (core): with at most one active emergency vehicle, just return
    it if approaching_intersection is True, else None.

    TODO (stretch): with multiple simultaneous emergency vehicles, apply a
    documented deterministic tie-break (e.g. closest distance first, ties
    broken by class priority ambulance > fire_truck > police_vehicle) and
    write the policy down in the report.
    """
    raise NotImplementedError
