"""Metric calculations shared by all experiments — implement each once here
so every controller (fixed-time / density-only / proposed) is scored the
same way.
"""


def average_waiting_time(per_vehicle_waits: list[float]) -> float:
    """Mean waiting time across all departed vehicles (seconds).

    Returns 0.0 for an empty list (no vehicles departed yet, not "zero wait").
    """
    if not per_vehicle_waits:
        return 0.0
    return sum(per_vehicle_waits) / len(per_vehicle_waits)


def max_waiting_time(per_vehicle_waits: list[float]) -> float:
    """Worst-case (longest) individual vehicle wait (seconds).

    Returns 0.0 for an empty list.
    """
    if not per_vehicle_waits:
        return 0.0
    return max(per_vehicle_waits)


def throughput(vehicles_crossed: int, duration_seconds: float) -> float:
    """Vehicles per second over the measurement window.

    SUMO's simulation clock is in seconds, so this is the natural unit.
    The report can trivially scale to vehicles/minute or vehicles/hour.
    Returns 0.0 if duration is zero or negative (avoids division-by-zero).
    """
    if duration_seconds <= 0.0:
        return 0.0
    return float(vehicles_crossed) / float(duration_seconds)


def emergency_response_time(emergency_detected_at: float, priority_granted_at: float) -> float:
    """Seconds between first detection of the emergency vehicle and the
    moment its approach road received priority (green or a committed
    transition toward green).

    Clamped to >= 0.0 defensively — a negative value would indicate a
    timestamp ordering bug upstream, not a real measurement.
    """
    return max(0.0, float(priority_granted_at) - float(emergency_detected_at))


def emergency_clearance_time(priority_granted_at: float, intersection_cleared_at: float) -> float:
    """Seconds between priority being granted and the emergency vehicle
    clearing the intersection (i.e. no longer approaching, confirmed by
    N consecutive frames/steps of non-approach).

    Clamped to >= 0.0 for the same reason as emergency_response_time.
    """
    return max(0.0, float(intersection_cleared_at) - float(priority_granted_at))
