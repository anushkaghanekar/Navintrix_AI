"""Traffic density per road.

Starts as a raw vehicle count; the weighted version applies
configs/intersection.yaml's density_weights. Document in the report that
these weights are tunable project parameters, not physical constants.
"""


def compute_density(vehicle_counts_by_class: dict[str, int], weights: dict[str, float]) -> float:
    """TODO: sum(weight[cls] * count[cls] for cls in vehicle_counts_by_class)."""
    raise NotImplementedError
