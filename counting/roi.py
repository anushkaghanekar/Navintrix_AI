"""Road / ROI assignment.

Assigns each TrackedVehicle to one of north/south/east/west using the
polygons in configs/intersection.yaml. Nothing here should hardcode pixel
coordinates — load them from config so a second intersection only needs a
new YAML file.
"""


def load_roi_config(path: str) -> dict:
    """TODO: load and validate configs/intersection.yaml."""
    raise NotImplementedError


def assign_road(position: tuple[float, float], roi_config: dict) -> str | None:
    """Return 'north' | 'south' | 'east' | 'west' | None for a point,
    based on which road's `roi` polygon it falls inside.

    TODO: point-in-polygon test against each road's roi.
    """
    raise NotImplementedError
