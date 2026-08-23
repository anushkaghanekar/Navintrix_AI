"""Determine the specific movement an emergency vehicle needs.

Don't just prioritize the whole road — if the trajectory indicates East ->
North, prioritize that specific movement, not the entire East approach.
Reuses counting/movement.py's entry/exit road classification logic.
"""


def required_movement(emergency_state, movement_history) -> str:
    """TODO: infer the emergency vehicle's intended movement (straight/
    left/right, and which road it needs green for) from its trajectory so
    far. Fall back to 'whole road' priority only if the movement genuinely
    can't be inferred yet (e.g. it just entered the approach).
    """
    raise NotImplementedError
