"""Road / ROI assignment.

Assigns each TrackedVehicle to one of north/south/east/west using the
polygons in configs/intersection.yaml. Nothing here should hardcode pixel
coordinates — load them from config so a second intersection only needs a
new YAML file.

Also exposes the trajectory crossing test for the counting line: a vehicle
is counted not from being inside a road's ROI but from its tracked center
segment (previous -> current position) intersecting the road's counting
line. That is the "line crossing, not per-frame detection" rule the
counting module depends on.
"""

from __future__ import annotations

import yaml

DEFAULT_CONFIG_PATH = "configs/intersection.yaml"

# Canonical approach order — used anywhere we iterate roads.
ROADS = ("north", "south", "east", "west")


def load_roi_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load and validate configs/intersection.yaml.

    Required structure:
      roads:
        north/south/east/west:
          roi: [[x1,y1], ...]        # non-empty polygon
          queue_roi: [[x1,y1], ...]  # may be empty until a camera is calibrated
          counting_line: [[x1,y1],[x2,y2]]  # two endpoints
          allowed_movements: [...]   # e.g. [straight, left, right]

    Polygons/coords only need to be filled in once a camera angle is known,
    but the keys must exist so a second intersection is one YAML file away.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "roads" not in cfg:
        raise ValueError(f"{path}: missing 'roads' section")
    roads = cfg["roads"]
    missing = [name for name in ROADS if name not in roads]
    if missing:
        raise ValueError(f"{path}: missing road(s) {missing}")

    for name in ROADS:
        road = roads[name]
        for key in ("roi", "queue_roi", "counting_line", "allowed_movements"):
            if key not in road:
                raise ValueError(f"{path}: road {name!r} is missing key {key!r}")
        for key in ("roi", "queue_roi", "counting_line"):
            coords = road[key]
            if not isinstance(coords, list):
                raise ValueError(f"{path}: road {name!r} {key!r} must be a list")
            if key == "counting_line" and coords and len(coords) != 2:
                raise ValueError(f"{path}: road {name!r} counting_line must have exactly 2 points")
        if not isinstance(road["allowed_movements"], list) or not road["allowed_movements"]:
            raise ValueError(f"{path}: road {name!r} allowed_movements must be a non-empty list")
    return cfg


def _find_polygon(roi_config: dict, road: str) -> list[tuple[float, float]]:
    return [(float(pt[0]), float(pt[1])) for pt in roi_config["roads"][road]["roi"]]


def _counting_line(
    roi_config: dict, road: str
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    pts = roi_config["roads"][road]["counting_line"]
    if not pts:
        return None
    return ((float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1])))


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (handles concave polygons)."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps=1e-9) -> bool:
    """Collinear-point test with bounding box + epsilon horizon."""
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float],
    q1: tuple[float, float], q2: tuple[float, float],
) -> bool:
    """True if segments p1-p2 and q1-q2 properly or improperly intersect."""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if abs(d1) < 1e-9 and _on_segment(q1, q2, p1):
        return True
    if abs(d2) < 1e-9 and _on_segment(q1, q2, p2):
        return True
    if abs(d3) < 1e-9 and _on_segment(p1, p2, q1):
        return True
    if abs(d4) < 1e-9 and _on_segment(p1, p2, q2):
        return True
    return False


# ---- PUBLIC ATTRIBUTION API ----


def assign_road(position: tuple[float, float], roi_config: dict) -> str | None:
    """Return 'north' | 'south' | 'east' | 'west' | None for a point,
    based on which road's `roi` polygon it falls inside."""
    x, y = position
    for name in ROADS:
        polygon = _find_polygon(roi_config, name)
        if polygon and point_in_polygon(x, y, polygon):
            return name
    return None


def crosses_counting_line(
    prev_position: tuple[float, float],
    current_position: tuple[float, float],
    road: str,
    roi_config: dict,
) -> bool:
    """True if the prev->current trajectory segment crosses the road's
    counting line. Returns False if the road or line isn't configured."""
    if not road or road not in roi_config.get("roads", {}):
        return False
    line = _counting_line(roi_config, road)
    if line is None:
        return False
    return _segments_intersect(prev_position, current_position, line[0], line[1])
