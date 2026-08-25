"""FastAPI backend entrypoint.

This API is intentionally a thin adapter over the project modules:
configuration comes from configs/*.yaml, signal state comes from
controller.state_machine.SafetyStateMachine, and simulation runs are
started through simulation.traci_controller.run_simulation. The backend
does not own signal authority and never sets phases directly.

Run: uvicorn backend.main:app --reload
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock
from time import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from controller.adaptive_controller import AdaptiveController
from controller.config import load_signal_config
from controller.emergency_controller import EmergencyController
from controller.fairness import FairnessTracker
from controller.state_machine import SafetyStateMachine
from counting.roi import load_roi_config
from emergency.priority import select_priority_emergency
from emergency.trajectory import required_movement
from evaluation.experiments import FixedTimeController, NoFairnessTracker
from simulation.sumo import load_scenario_config
from simulation.traci_controller import run_simulation


SIGNAL_CONFIG_PATH = "configs/signal.yaml"
INTERSECTION_CONFIG_PATH = "configs/intersection.yaml"
DEFAULT_CONTROLLER_MODE = "ADAPTIVE"
VALID_CONTROLLER_MODES = {"ADAPTIVE", "FIXED_TIME", "DENSITY_ONLY"}

app = FastAPI(title="Adaptive Traffic Signal Control API")


class _PriorityModule:
    def select_priority_emergency(self, emergencies, phase):
        return select_priority_emergency(emergencies, phase)


class _TrajectoryModule:
    def required_movement(self, state, history):
        return required_movement(state, history)


class BackendRuntime:
    """Holds the latest live/demo state exposed by the API.

    The first backend milestone is honest orchestration: routes return real
    config/FSM/controller state, and simulation starts call the real SUMO
    bridge. Per-step streaming can be added by teaching run_simulation to
    accept a callback that writes into this object.
    """

    def __init__(
        self,
        signal_config_path: str = SIGNAL_CONFIG_PATH,
        intersection_config_path: str = INTERSECTION_CONFIG_PATH,
    ):
        self.signal_config_path = signal_config_path
        self.intersection_config_path = intersection_config_path
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.signal_cfg = load_signal_config(self.signal_config_path)
            self.intersection_cfg = load_roi_config(self.intersection_config_path)
            self.mode = DEFAULT_CONTROLLER_MODE
            self.running = False
            self.stop_requested = False
            self.scenario: str | None = None
            self.last_error: str | None = None
            self.latest_results: dict | None = None
            self.controller_clock_seconds = 0.0
            self.last_updated = time()
            self._build_control_stack()
            self.latest_metrics = self._empty_metrics()
            self.latest_vehicles: list[dict] = []
            self.latest_emergency: dict | None = None

    @property
    def roads(self) -> list[str]:
        return list(self.intersection_cfg["roads"].keys())

    def _empty_metrics(self) -> dict:
        return {
            road: {
                "density": 0.0,
                "queue_length": 0,
                "waiting_time": 0.0,
                "flow": 0.0,
            }
            for road in self.roads
        }

    def _build_control_stack(self) -> None:
        self.state_machine = SafetyStateMachine(dict(self.signal_cfg["signal"]))
        if self.mode == "FIXED_TIME":
            self.adaptive_controller = FixedTimeController(self.state_machine)
        elif self.mode == "DENSITY_ONLY":
            coeffs = self.signal_cfg["controller"]["baselines"]["density_only"]
            self.adaptive_controller = AdaptiveController(
                self.state_machine,
                {"controller": dict(coeffs)},
                NoFairnessTracker(),
            )
        else:
            max_wait = self.signal_cfg["controller"]["fairness"][
                "max_wait_before_forced_green_seconds"
            ]
            self.adaptive_controller = AdaptiveController(
                self.state_machine,
                self.signal_cfg,
                FairnessTracker(max_wait_seconds=float(max_wait)),
            )
        self.emergency_controller = EmergencyController(
            self.state_machine, _PriorityModule(), _TrajectoryModule()
        )

    def set_mode(self, mode: str) -> dict:
        normalized = mode.strip().upper()
        if normalized not in VALID_CONTROLLER_MODES:
            raise ValueError(
                f"mode must be one of {', '.join(sorted(VALID_CONTROLLER_MODES))}"
            )
        with self._lock:
            if self.running:
                raise RuntimeError("controller mode cannot change while simulation is running")
            self.mode = normalized
            self._build_control_stack()
            self.last_updated = time()
            return self.status()

    def intersection(self) -> dict:
        with self._lock:
            return {
                "intersection": deepcopy(self.intersection_cfg.get("intersection", {})),
                "roads": deepcopy(self.intersection_cfg["roads"]),
                "movement_directions": deepcopy(
                    self.intersection_cfg.get("movement_directions", {})
                ),
                "density_weights": deepcopy(self.intersection_cfg.get("density_weights", {})),
            }

    def traffic(self) -> dict:
        with self._lock:
            signal_by_road = self._signal_by_road()
            return {
                road: {
                    "vehicles": self.latest_metrics.get(road, {}).get("density", 0.0),
                    "density": self.latest_metrics.get(road, {}).get("density", 0.0),
                    "queue": self.latest_metrics.get(road, {}).get("queue_length", 0),
                    "waiting_seconds": self.latest_metrics.get(road, {}).get(
                        "waiting_time", 0.0
                    ),
                    "flow": self.latest_metrics.get(road, {}).get("flow", 0.0),
                    "signal": signal_by_road.get(road, "RED"),
                }
                for road in self.roads
            }

    def vehicles(self) -> list[dict]:
        with self._lock:
            return deepcopy(self.latest_vehicles)

    def signals(self) -> dict:
        with self._lock:
            now = self.controller_clock_seconds
            phase = self.state_machine.phase.name
            elapsed = max(0.0, now - self.state_machine.phase_start_t)
            return {
                "phase": phase,
                "current_green_road": self.state_machine.current_green_road(),
                "pending_road": self.state_machine.pending_road(),
                "outgoing_road": self.state_machine.outgoing_road(),
                "green_roads": self.state_machine.green_roads(),
                "signals_by_road": self._signal_by_road(),
                "elapsed_phase_seconds": elapsed,
                "remaining_phase_seconds": self._remaining_phase_seconds(elapsed),
            }

    def emergency(self) -> dict:
        with self._lock:
            mode = getattr(
                self.emergency_controller.mode, "name", str(self.emergency_controller.mode)
            )
            return {
                "active": self.latest_emergency is not None,
                "mode": mode,
                "vehicle": deepcopy(self.latest_emergency),
            }

    def metrics(self) -> dict:
        with self._lock:
            return deepcopy(self.latest_metrics)

    def status(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "running": self.running,
                "stop_requested": self.stop_requested,
                "scenario": self.scenario,
                "last_error": self.last_error,
                "last_updated": self.last_updated,
                "latest_results": deepcopy(self.latest_results),
            }

    def start_simulation(
        self,
        background_tasks: BackgroundTasks,
        scenario: str,
        max_steps: int,
        use_gui: bool,
    ) -> dict:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        scenario_path = self._resolve_scenario_path(scenario)
        with self._lock:
            if self.running:
                raise RuntimeError("simulation is already running")
            self.running = True
            self.stop_requested = False
            self.scenario = Path(scenario_path).stem
            self.last_error = None
            self.latest_results = None
            self.controller_clock_seconds = 0.0
            self.latest_metrics = self._empty_metrics()
            self.latest_vehicles = []
            self.latest_emergency = None
            self._build_control_stack()
            self.last_updated = time()
        background_tasks.add_task(
            self._run_simulation_job, scenario_path, int(max_steps), bool(use_gui)
        )
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            self.stop_requested = True
            if not self.running:
                self.stop_requested = False
            self.last_updated = time()
            return self.status()

    def _run_simulation_job(self, scenario_path: str, max_steps: int, use_gui: bool) -> None:
        try:
            metrics = run_simulation(
                scenario_path,
                self.state_machine,
                self.adaptive_controller,
                self.emergency_controller,
                max_steps=max_steps,
                config_path=self.signal_config_path,
                intersection_config_path=self.intersection_config_path,
                use_gui=use_gui,
            )
        except Exception as exc:  # surfacing background failure through status endpoint
            with self._lock:
                self.last_error = str(exc)
                self.running = False
                self.stop_requested = False
                self.last_updated = time()
            return
        with self._lock:
            self.latest_results = metrics
            self.controller_clock_seconds = float(metrics.get("sim_seconds", 0.0))
            self.latest_metrics = self._metrics_from_results(metrics)
            self.running = False
            self.stop_requested = False
            self.last_updated = time()

    def _resolve_scenario_path(self, scenario: str) -> str:
        if scenario.endswith(".sumocfg") or "/" in scenario or "\\" in scenario:
            return scenario
        return load_scenario_config(scenario)

    def _signal_by_road(self) -> dict[str, str]:
        green = set(self.state_machine.green_roads())
        if self.state_machine.phase.name == "YELLOW":
            outgoing = self.state_machine.outgoing_road()
            return {
                road: ("YELLOW" if road == outgoing else "RED") for road in self.roads
            }
        if self.state_machine.phase.name == "ALL_RED":
            return {road: "RED" for road in self.roads}
        return {road: ("GREEN" if road in green else "RED") for road in self.roads}

    def _remaining_phase_seconds(self, elapsed: float) -> float:
        phase = self.state_machine.phase.name
        if phase == "GREEN":
            duration = self.state_machine.max_green_seconds
        elif phase == "YELLOW":
            duration = self.state_machine.yellow_seconds
        else:
            duration = self.state_machine.all_red_seconds
        return max(0.0, duration - elapsed)

    def _metrics_from_results(self, results: dict) -> dict:
        metrics = self._empty_metrics()
        for road in self.roads:
            if road in results.get("avg_queue_length_by_road", {}):
                metrics[road]["queue_length"] = results["avg_queue_length_by_road"][road]
            if road in results.get("waiting_avg_by_road", {}):
                metrics[road]["waiting_time"] = results["waiting_avg_by_road"][road]
            if road in results.get("final_flow_by_road", {}):
                metrics[road]["flow"] = results["final_flow_by_road"][road]
        return metrics

    def snapshot(self) -> dict:
        return {
            "traffic": self.traffic(),
            "vehicles": self.vehicles(),
            "signals": self.signals(),
            "emergency": self.emergency(),
            "metrics": self.metrics(),
            "controller": self.status(),
        }


_RUNTIME = BackendRuntime()


@app.get("/api/intersection")
def get_intersection():
    """Return static intersection/config metadata for the dashboard to render."""
    return _RUNTIME.intersection()


@app.get("/api/traffic")
def get_traffic():
    return _RUNTIME.traffic()


@app.get("/api/vehicles")
def get_vehicles():
    """Return currently tracked vehicles from the latest live/demo state."""
    return _RUNTIME.vehicles()


@app.get("/api/signals")
def get_signals():
    """Return current phase/road/timing from controller.state_machine.SafetyStateMachine."""
    return _RUNTIME.signals()


@app.get("/api/emergency")
def get_emergency():
    return _RUNTIME.emergency()


@app.get("/api/metrics")
def get_metrics():
    """Return live density/queue/waiting-time/flow per road."""
    return _RUNTIME.metrics()


@app.get("/api/controller/status")
def controller_status():
    return _RUNTIME.status()


@app.post("/api/controller/start")
def controller_start(
    background_tasks: BackgroundTasks,
    scenario: str = "balanced",
    max_steps: int = 3600,
    use_gui: bool = False,
):
    """Start a SUMO control loop in a background task."""
    try:
        return _RUNTIME.start_simulation(background_tasks, scenario, max_steps, use_gui)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/controller/stop")
def controller_stop():
    return _RUNTIME.stop()


@app.post("/api/controller/mode")
def controller_mode(mode: str):
    """Switch between ADAPTIVE / FIXED_TIME / DENSITY_ONLY for live demo comparisons."""
    try:
        return _RUNTIME.set_mode(mode)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Streams live vehicle counts, signal changes, emergency detections,
    and metric updates. Clients send a lightweight ping/message to request
    the latest snapshot.
    """
    await websocket.accept()
    try:
        await websocket.send_json(_RUNTIME.snapshot())
        while True:
            await websocket.receive_text()
            await websocket.send_json(_RUNTIME.snapshot())
    except WebSocketDisconnect:
        pass
