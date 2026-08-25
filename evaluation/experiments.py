"""Experiment runner — drives simulation/traci_controller.run_simulation
across controllers x scenarios and writes raw results out for plots.py to
consume. Never hand-type a result into the report; everything should trace
back to a file this script produced.

Core scope: Experiment 1 (fixed-time vs density-only vs proposed) and
Experiment 4 (adaptive controller with/without emergency priority), each
across the 3 core scenarios (balanced / heavy / emergency).
"""


from __future__ import annotations

import json
from pathlib import Path
import yaml

from controller.adaptive_controller import AdaptiveController
from controller.emergency_controller import EmergencyController
from controller.fairness import FairnessTracker
from controller.state_machine import SafetyStateMachine
from emergency.priority import select_priority_emergency
from emergency.trajectory import required_movement
from evaluation.metrics import throughput
from simulation.sumo import load_scenario_config
from simulation.traci_controller import run_simulation


class FixedTimeController:
    """Baseline: doesn't submit dynamic phase change requests.
    The SafetyStateMachine automatically cycles phases when max_green_seconds expires.
    """
    def __init__(self, state_machine):
        self.state_machine = state_machine
        self._now = 0.0

    def set_time(self, timestamp: float) -> None:
        self._now = float(timestamp)

    def choose_next_road(self, road_metrics: dict) -> str | None:
        return self.state_machine.current_green_road()


class NoFairnessTracker:
    """Fairness policy disabled explicitly for a baseline controller."""

    def __init__(self):
        self.last_green: tuple[str, float] | None = None

    def on_green_granted(self, road: str, timestamp: float) -> None:
        self.last_green = (road, float(timestamp))

    def road_forcing_green(self, timestamp: float) -> str | None:
        return None


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


def _load_signal_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if "signal" not in cfg or "controller" not in cfg:
        raise ValueError(f"{config_path} must define 'signal' and 'controller' blocks")
    return cfg


def _density_only_config(signal_cfg: dict) -> dict:
    try:
        coeffs = signal_cfg["controller"]["baselines"]["density_only"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "configs/signal.yaml must define controller.baselines.density_only "
            "coefficients for Experiment 1"
        ) from exc
    return {"controller": dict(coeffs)}


def _fairness_tracker(signal_cfg: dict) -> FairnessTracker:
    try:
        max_wait = signal_cfg["controller"]["fairness"][
            "max_wait_before_forced_green_seconds"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "configs/signal.yaml must define "
            "controller.fairness.max_wait_before_forced_green_seconds"
        ) from exc
    return FairnessTracker(max_wait_seconds=float(max_wait))


def _with_derived_metrics(raw_metrics: dict) -> dict:
    """Add metrics.py-derived fields without mutating run_simulation output."""
    metrics = dict(raw_metrics)
    duration = float(metrics.get("sim_seconds", 0.0))
    by_road = dict(metrics.get("throughput_by_road") or {})
    metrics["throughput_rate_total"] = throughput(
        int(metrics.get("throughput_total", 0)), duration
    )
    metrics["throughput_rate_by_road"] = {
        road: throughput(int(count), duration) for road, count in by_road.items()
    }
    return metrics


def _result_records(results: dict) -> list[dict]:
    records = []
    for controller, scenario in sorted(results):
        records.append(
            {
                "controller": controller,
                "scenario": scenario,
                "metrics": results[(controller, scenario)],
            }
        )
    return records


def _write_results(results: dict, output_dir: str | None, filename: str) -> None:
    if output_dir is None:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(
        json.dumps(_result_records(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_experiment_1(
    scenarios: list[str],
    max_steps: int = 3600,
    config_path: str = "configs/signal.yaml",
    intersection_config_path: str = "configs/intersection.yaml",
    output_dir: str | None = None,
) -> dict:
    """For each scenario, run all three controllers, collect metrics
    from evaluation/metrics.py, return a results dict keyed by
    (controller, scenario).

    If output_dir is provided, also writes experiment_1_results.json as
    raw, reproducible records for plots.py/report generation.
    """
    signal_cfg = _load_signal_config(config_path)

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
        results[("fixed_time", scenario_name)] = _with_derived_metrics(metrics_fixed)

        # 2. Density-only baseline (coefficients from config, no fairness override)
        sm_density = SafetyStateMachine(dict(signal_cfg["signal"]))
        density_cfg = _density_only_config(signal_cfg)
        fairness_dummy = NoFairnessTracker()
        ctrl_density = AdaptiveController(sm_density, density_cfg, fairness_dummy)
        metrics_density = run_simulation(
            scenario_path, sm_density, ctrl_density, em_dummy, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("density_only", scenario_name)] = _with_derived_metrics(metrics_density)

        # 3. Proposed Adaptive Controller
        sm_prop = SafetyStateMachine(dict(signal_cfg["signal"]))
        fairness_prop = _fairness_tracker(signal_cfg)
        ctrl_prop = AdaptiveController(sm_prop, signal_cfg, fairness_prop)
        em_prop = _build_emergency_controller(sm_prop)
        metrics_prop = run_simulation(
            scenario_path, sm_prop, ctrl_prop, em_prop, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("proposed", scenario_name)] = _with_derived_metrics(metrics_prop)

    _write_results(results, output_dir, "experiment_1_results.json")
    return results


def run_experiment_4(
    scenarios: list[str],
    max_steps: int = 3600,
    config_path: str = "configs/signal.yaml",
    intersection_config_path: str = "configs/intersection.yaml",
    output_dir: str | None = None,
) -> dict:
    """For each scenario, run the proposed controller with emergency
    priority on vs off, collect metrics, return results dict.

    If output_dir is provided, also writes experiment_4_results.json as
    raw, reproducible records for plots.py/report generation.
    """
    signal_cfg = _load_signal_config(config_path)

    results = {}
    for scenario in scenarios:
        scenario_path = _resolve_scenario_path(scenario)
        scenario_name = Path(scenario_path).stem

        # Emergency priority ON
        sm_on = SafetyStateMachine(dict(signal_cfg["signal"]))
        fairness_on = _fairness_tracker(signal_cfg)
        ctrl_on = AdaptiveController(sm_on, signal_cfg, fairness_on)
        em_on = _build_emergency_controller(sm_on)
        metrics_on = run_simulation(
            scenario_path, sm_on, ctrl_on, em_on, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("emergency_priority_on", scenario_name)] = _with_derived_metrics(metrics_on)

        # Emergency priority OFF
        sm_off = SafetyStateMachine(dict(signal_cfg["signal"]))
        fairness_off = _fairness_tracker(signal_cfg)
        ctrl_off = AdaptiveController(sm_off, signal_cfg, fairness_off)
        em_off = DummyEmergencyController()
        metrics_off = run_simulation(
            scenario_path, sm_off, ctrl_off, em_off, max_steps=max_steps,
            config_path=config_path, intersection_config_path=intersection_config_path,
        )
        results[("emergency_priority_off", scenario_name)] = _with_derived_metrics(metrics_off)

    _write_results(results, output_dir, "experiment_4_results.json")
    return results
