"""Movement detection: straight / left / right, from trajectory not one frame.

Compare a vehicle's entry road to its exit road (or exit the exit road once
it's past the intersection zone) to classify the movement. This feeds both
class-wise straight/left/right flow stats and emergency-vehicle
movement-level priority.

The turn geometry ("is east->north a left or a right?") depends on the
intersection's layout and traffic rules, so it is NOT hardcoded here. It
lives in configs/intersection.yaml under ``movement_directions``:

    movement_directions:        # entry_road: {exit_road: movement}
      north: {south: straight, east: left,   west: right}
      south: {north: straight, west: left,   east: right}
      east:  {west:  straight, south: left,  north: right}
      west:  {east:  straight, north: left,  south: right}

Right-hand-traffic convention, N-S vertical:
  * from the north, going south is straight; turning east is a left turn
    (counter-clockwise), west is a right turn (clockwise).
Changing intersections only means a new YAML table.
"""

from __future__ import annotations

import yaml

DEFAULT_CONFIG_PATH = "configs/intersection.yaml"
VALID_MOVEMENTS = {"straight", "left", "right"}


def load_movement_directions(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, dict[str, str]]:
    """Return {entry_road: {exit_road: 'straight'|'left'|'right'}} from
    configs/intersection.yaml's ``movement_directions`` section."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        directions = cfg["movement_directions"]
    except (KeyError, TypeError):
        raise ValueError(
            f"{config_path} must define 'movement_directions' (the "
            "entry-road -> exit-road -> movement table)"
        )
    if not isinstance(directions, dict) or not directions:
        raise ValueError(f"{config_path}: 'movement_directions' must be a non-empty mapping")
    for entry_road, exits in directions.items():
        if not isinstance(exits, dict):
            raise ValueError(
                f"{config_path}: movement_directions[{entry_road!r}] must be a mapping"
            )
        for exit_road, label in exits.items():
            if label not in VALID_MOVEMENTS:
                raise ValueError(
                    f"{config_path}: movement {entry_road!r}->{exit_road!r} "
                    f"must be one of {sorted(VALID_MOVEMENTS)}, got {label!r}"
                )
    return {
        entry: {exit_road: label for exit_road, label in exits.items()}
        for entry, exits in directions.items()
    }


def _default_movement_config():
    """Lazily loaded + cached default table (config-driven, one read)."""
    if getattr(_default_movement_config, "cached", None) is None:
        _default_movement_config.cached = load_movement_directions()
    return _default_movement_config.cached


def classify_movement(entry_road: str, exit_road: str, table: dict | None = None) -> str:
    """Return 'straight' | 'left' | 'right' for a vehicle that entered from
    ``entry_road`` and exited toward ``exit_road``.

    ``table`` is an optional {entry: {exit: movement}} mapping; it defaults
    to configs/intersection.yaml's movement_directions. The standalone kwarg
    keeps the function pure so tests can inject a mapping without hitting the
    filesystem, but a call does not require it.
    """
    if table is None:
        table = _default_movement_config()
    return table.get(entry_road, {}).get(exit_road, "unknown")
