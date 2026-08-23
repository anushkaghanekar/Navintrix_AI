"""Movement detection: straight / left / right, from trajectory not one frame.

Compare a vehicle's entry road to its exit road (or exit heading once it's
past the intersection zone) to classify the movement. This feeds both
class-wise flow stats and emergency-vehicle movement-level priority.
"""


def classify_movement(entry_road: str, exit_road: str) -> str:
    """Return 'straight' | 'left' | 'right' given the entry and exit roads.

    TODO: define the entry->exit mapping (e.g. east->west = straight,
    east->north = left, east->south = right) based on your intersection's
    actual layout/config, not hardcoded assumptions.
    """
    raise NotImplementedError
