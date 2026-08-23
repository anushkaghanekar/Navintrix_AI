"""Experiment runner — drives simulation/traci_controller.run_simulation
across controllers x scenarios and writes raw results out for plots.py to
consume. Never hand-type a result into the report; everything should trace
back to a file this script produced.

Core scope: Experiment 1 (fixed-time vs density-only vs proposed) and
Experiment 4 (adaptive controller with/without emergency priority), each
across the 3 core scenarios (balanced / heavy / emergency).
"""


def run_experiment_1(scenarios: list[str]) -> dict:
    """TODO: for each scenario, run all three controllers, collect metrics
    from evaluation/metrics.py, return a results dict keyed by
    (controller, scenario).
    """
    raise NotImplementedError


def run_experiment_4(scenarios: list[str]) -> dict:
    """TODO: for each scenario, run the proposed controller with emergency
    priority on vs off, collect metrics, return results dict.
    """
    raise NotImplementedError
