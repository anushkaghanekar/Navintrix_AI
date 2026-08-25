"""Experiment runner — drives simulation/traci_controller.run_simulation
across controllers x scenarios and writes raw results out for plots.py to
consume. Never hand-type a result into the report; everything should trace
back to a file this script produced.

Core scope: Experiment 1 (fixed-time vs density-only vs proposed) and
Experiment 4 (adaptive controller with/without emergency priority), each
across the 3 core scenarios (balanced / heavy / emergency).
"""


from __future__ import annotations

from pathlib import Path
import yaml

from controller.adaptive_controller import AdaptiveController
from controller.emergency_controller import EmergencyController
from controller.fairness import FairnessTracker
from controller.state_machine import SafetyStateMachine
from emergency.priority import select_priority_emergency
from emergency.trajectory import required_movement
from simulation.sumo import load_scenario_config
from simulation.traci_controller import run_simulation


class FixedTimeController:
    """Baseline: doesn't submit dynamic phase change requests.
    The SafetyStateMachine automatically cycles phases when max_green_seconds expires.
    """
    def __init__(self, state_machine):
        self.state_machine = state_machine

    def set_time(self, timestamp: float) -> None:
        pass

    def choose_next_road(self, road_metrics: dict) -> str | None:
        return self.state_machine.current_green_road()


class DummyEmergencyController:
    """Dummy emergency controller that never requests priority (priority OFF)."""
    def __init__(self):
        self.mode = None

    def handle(self, active_emergencies: list, timestamp: float) -> bool:
        return False


def _build_emergency_controller(sm: SafetyStateMachine) -> EmergencyController:
    class PriorityShim:
        def select_priority_emergency(self, emergencies, phase):
            return select_priority_emergency(emergencies, phase)

    class TrajectoryShim:
        def required_movement(self, state, history):
            return required_movement(state, history)

    return EmergencyController(sm, PriorityShim(), TrajectoryShim())


def _resolve_scenario_path(scenario: str) -> str:
    """Accept scenario name or full path."""
    if scenario.endswith(".sumocfg") or "/" in scenario or "\\" in scenario:
        return scenario
    return load_scenario_config(scenario)


def run_experiment_1(
    scenarios: list[str],
    max_steps: int = 3600,
    config_path: str = "configs/signal.yaml",
    intersection_config_path: str = "configs/intersection.yaml",
) -> dict:
    """For each scenario, run all three controllers, collect metrics
    from evaluation/metrics.py, return a results dict keyed by
    (controller, scenario).
    """
    with open(config_path) as f:
        signal_cfg = yaml.safe_load(f)

    results = {}
    for scenario in scenarios:
        scenario_path = _resolve_scenario_path(scenario)
        scenario_name = Path(scenario_path).stem

        # 1. Fixed-time baseline
        sm_fixed = SafetyStateMachine(dict(signal_cfg["signal"]))
        ctrl_fixed = FixedTimeController(sm_fixed)
        em_dummy = DummyEmergencyController()
        metrics_fixed = run_simulation(
            scenario_path, sm_fixed, ctrl_fixed, em_dummy, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("fixed_time", scenario_name)] = metrics_fixed

        # 2. Density-only baseline (alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, no fairness override)
        sm_density = SafetyStateMachine(dict(signal_cfg["signal"]))
        density_cfg = {"controller": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}}
        fairness_dummy = FairnessTracker(max_wait_seconds=1e9)
        ctrl_density = AdaptiveController(sm_density, density_cfg, fairness_dummy)
        metrics_density = run_simulation(
            scenario_path, sm_density, ctrl_density, em_dummy, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("density_only", scenario_name)] = metrics_density

        # 3. Proposed Adaptive Controller
        sm_prop = SafetyStateMachine(dict(signal_cfg["signal"]))
        fairness_prop = FairnessTracker(
            max_wait_seconds=float(signal_cfg["controller"]["fairness"]["max_wait_before_forced_green_seconds"])
        )
        ctrl_prop = AdaptiveController(sm_prop, signal_cfg, fairness_prop)
        em_prop = _build_emergency_controller(sm_prop)
        metrics_prop = run_simulation(
            scenario_path, sm_prop, ctrl_prop, em_prop, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("proposed", scenario_name)] = metrics_prop

    return results


def run_experiment_4(
    scenarios: list[str],
    max_steps: int = 3600,
    config_path: str = "configs/signal.yaml",
    intersection_config_path: str = "configs/intersection.yaml",
) -> dict:
    """For each scenario, run the proposed controller with emergency
    priority on vs off, collect metrics, return results dict.
    """
    with open(config_path) as f:
        signal_cfg = yaml.safe_load(f)

    results = {}
    for scenario in scenarios:
        scenario_path = _resolve_scenario_path(scenario)
        scenario_name = Path(scenario_path).stem

        # Emergency priority ON
        sm_on = SafetyStateMachine(dict(signal_cfg["signal"]))
        fairness_on = FairnessTracker(
            max_wait_seconds=float(signal_cfg["controller"]["fairness"]["max_wait_before_forced_green_seconds"])
        )
        ctrl_on = AdaptiveController(sm_on, signal_cfg, fairness_on)
        em_on = _build_emergency_controller(sm_on)
        metrics_on = run_simulation(
            scenario_path, sm_on, ctrl_on, em_on, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("emergency_priority_on", scenario_name)] = metrics_on

        # Emergency priority OFF
        sm_off = SafetyStateMachine(dict(signal_cfg["signal"]))
        fairness_off = FairnessTracker(
            max_wait_seconds=float(signal_cfg["controller"]["fairness"]["max_wait_before_forced_green_seconds"])
        )
        ctrl_off = AdaptiveController(sm_off, signal_cfg, fairness_off)
        em_off = DummyEmergencyController()
        metrics_off = run_simulation(
            scenario_path, sm_off, ctrl_off, em_off, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("emergency_priority_off", scenario_name)] = metrics_off

    return results
