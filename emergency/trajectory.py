"""Determine the specific movement an emergency vehicle needs.

Don't just prioritize the whole road — if the trajectory indicates East ->
North, prioritize that specific movement, not the entire East approach.
Reuses counting/movement.py's entry/exit road classification logic.

If the intended direction can't be inferred yet (e.g. the vehicle just
entered the approach and hasn't established a heading), return "unknown"
and let the controller serve the whole road rather than guess a turn the
vehicle isn't making.
"""

from __future__ import annotations

from counting.movement import classify_movement


def required_movement(emergency_state, movement_history) -> str:
    """Return the movement the emergency vehicle needs, if it can be inferred.

    ``emergency_state`` provides the current ``road``; ``movement_history``
    is an iterable of the vehicle's past attributed road labels (in
    chronological/FIFO order). If the vehicle was seen on an entry road and
    then on a DIFFERENT road, the turn is classified via the movement table.
    With insufficient history/direction, return the sentinel ``"unknown"``.
    """
    entry_road = emergency_state.road
    # Walk the history for the *first* road this vehicle moved through that
    # differs from the current one (the exit it is heading toward).
    exit_road = None
    for past_road in movement_history:
        if past_road is not None and past_road != entry_road:
            exit_road = past_road
            break
    if exit_road is not None:
        return classify_movement(entry_road, exit_road)
    return "unknown"
