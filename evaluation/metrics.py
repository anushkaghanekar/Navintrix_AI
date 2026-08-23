"""Metric calculations shared by all experiments — implement each once here
so every controller (fixed-time / density-only / proposed) is scored the
same way.
"""


def average_waiting_time(per_vehicle_waits: list[float]) -> float:
    raise NotImplementedError


def max_waiting_time(per_vehicle_waits: list[float]) -> float:
    raise NotImplementedError


def throughput(vehicles_crossed: int, duration_seconds: float) -> float:
    raise NotImplementedError


def emergency_response_time(emergency_detected_at: float, priority_granted_at: float) -> float:
    raise NotImplementedError


def emergency_clearance_time(priority_granted_at: float, intersection_cleared_at: float) -> float:
    raise NotImplementedError
