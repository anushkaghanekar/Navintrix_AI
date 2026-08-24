"""Traffic density per road.

Starts as a raw vehicle count; the weighted version applies
configs/intersection.yaml's density_weights. Document in the report that
these weights are tunable project parameters, not physical constants.
"""

from __future__ import annotations

import yaml

DEFAULT_CONFIG_PATH = "configs/intersection.yaml"


def load_density_weights(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, float]:
    """Load configs/intersection.yaml ``density_weights``; cast to float."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        raw = cfg["density_weights"]
    except (KeyError, TypeError):
        raise ValueError(f"{config_path} must define 'density_weights'")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{config_path}: 'density_weights' must be a non-empty mapping")
    return {cls: float(weight) for cls, weight in raw.items()}


def compute_density(
    vehicle_counts_by_class: dict[str, int],
    weights: dict[str, float],
) -> float:
    """Weighted vehicle density: sum(weight[cls] * count[cls]).

    ``vehicle_counts_by_class`` is {class_name: count}; classes absent from
    the weights dict contribute 0 (no KeyError).
    """
    total = 0.0
    for cls, count in vehicle_counts_by_class.items():
        total += float(count) * float(weights.get(cls, 0.0))
    return total
