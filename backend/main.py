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
from time import monotonic, perf_counter, time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from analytics.density import compute_density, load_density_weights
from controller.adaptive_controller import AdaptiveController
from controller.config import load_signal_config
from controller.emergency_controller import EmergencyController
from controller.fairness import FairnessTracker
from controller.state_machine import SafetyStateMachine
from counting.roi import assign_road, load_roi_config
from detection.detector import build_detector
from emergency.priority import select_priority_emergency
from emergency.trajectory import required_movement
from emergency.tracker import EmergencyTracker
from evaluation.experiments import FixedTimeController, NoFairnessTracker
from simulation.sumo import load_scenario_config
from simulation.traci_controller import run_simulation
from tracking.bytetrack import VehicleTracker


SIGNAL_CONFIG_PATH = "configs/signal.yaml"
INTERSECTION_CONFIG_PATH = "configs/intersection.yaml"
MODEL_CONFIG_PATH = "configs/model.yaml"
DEFAULT_CONTROLLER_MODE = "ADAPTIVE"
VALID_CONTROLLER_MODES = {"ADAPTIVE", "FIXED_TIME", "DENSITY_ONLY"}
EMERGENCY_CLASSES = {"ambulance", "fire_truck", "police_vehicle"}

app = FastAPI(title="Adaptive Traffic Signal Control API")


class _PriorityModule:
    def select_priority_emergency(self, emergencies, phase):
        return select_priority_emergency(emergencies, phase)


class _TrajectoryModule:
    def required_movement(self, state, history):
        return required_movement(state, history)


class BackendRuntime:
    """Holds the latest live/demo state exposed by the API.

    The runtime has two independent producers of state:
      * the SUMO controller job, which fills aggregate simulation results;
      * the live vision path, which accepts camera frames and runs the trained
        YOLO model through tracking and emergency approach verification.

    The detector is loaded lazily so importing the API remains lightweight and
    the backend can still expose configuration/status routes on machines that
    do not have the model weights installed.
    """

    def __init__(
        self,
        signal_config_path: str = SIGNAL_CONFIG_PATH,
        intersection_config_path: str = INTERSECTION_CONFIG_PATH,
    ):
        self.signal_config_path = signal_config_path
        self.intersection_config_path = intersection_config_path
        self._lock = RLock()
        self._inference_lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.signal_cfg = load_signal_config(self.signal_config_path)
            self.intersection_cfg = load_roi_config(self.intersection_config_path)
            self.density_weights = load_density_weights(self.intersection_config_path)
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
            self.latest_detections: list[dict] = []
            self.detector = None
            self.live_tracker = VehicleTracker.from_config(MODEL_CONFIG_PATH)
            self.emergency_tracker = EmergencyTracker.from_config(self.signal_cfg)
            self.live_clock_origin: float | None = None
            self.inference_frame_count = 0
            self.last_inference_ms: float | None = None

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

    def reset_live_state(self) -> None:
        """Clear frame-tracking state without resetting the SUMO controller."""
        with self._inference_lock:
            with self._lock:
                self.live_tracker.reset()
                self.emergency_tracker = EmergencyTracker.from_config(self.signal_cfg)
                self.emergency_controller = EmergencyController(
                    self.state_machine, _PriorityModule(), _TrajectoryModule()
                )
                self.live_clock_origin = None
                self.latest_detections = []
                self.latest_vehicles = []
                self.latest_emergency = None
                self.latest_metrics = self._empty_metrics()
                self.controller_clock_seconds = 0.0
                self.last_inference_ms = None

    def inference_status(self) -> dict:
        """Return model/inference health without forcing model loading."""
        with self._lock:
            weights_path = "models/best.pt"
            if self.detector is not None:
                weights_path = self.detector.weights_path
            return {
                "model_loaded": self.detector is not None,
                "weights_path": weights_path,
                "weights_present": Path(weights_path).exists(),
                "frames_processed": self.inference_frame_count,
                "last_inference_ms": self.last_inference_ms,
                "confidence_threshold": (
                    self.detector.confidence_threshold if self.detector is not None else None
                ),
            }

    def _ensure_detector(self):
        if self.detector is None:
            try:
                self.detector = build_detector(MODEL_CONFIG_PATH)
            except Exception as exc:  # noqa: BLE001 - expose a useful API error
                raise RuntimeError(
                    "Unable to load the YOLO model from configs/model.yaml: "
                    f"{exc}"
                ) from exc
        return self.detector

    @staticmethod
    def _serialize_detection(detection) -> dict:
        return {
            "class": detection.cls,
            "cls": detection.cls,
            "confidence": float(detection.confidence),
            "bbox": {
                "x1": float(detection.x1),
                "y1": float(detection.y1),
                "x2": float(detection.x2),
                "y2": float(detection.y2),
            },
            "emergency": detection.cls in EMERGENCY_CLASSES,
        }

    @staticmethod
    def _serialize_tracked_vehicle(vehicle) -> dict:
        return {
            "track_id": int(vehicle.track_id),
            "class": vehicle.cls,
            "cls": vehicle.cls,
            "confidence": float(vehicle.confidence),
            "bbox": (
                {
                    "x1": float(vehicle.bbox[0]),
                    "y1": float(vehicle.bbox[1]),
                    "x2": float(vehicle.bbox[2]),
                    "y2": float(vehicle.bbox[3]),
                }
                if vehicle.bbox is not None
                else None
            ),
            "center": {
                "x": float(vehicle.current_position[0]),
                "y": float(vehicle.current_position[1]),
            },
            "previous_center": (
                {
                    "x": float(vehicle.previous_position[0]),
                    "y": float(vehicle.previous_position[1]),
                }
                if vehicle.previous_position is not None
                else None
            ),
            "road": vehicle.road,
            "movement": vehicle.movement,
            "emergency": vehicle.cls in EMERGENCY_CLASSES,
        }

    @staticmethod
    def _serialize_emergency_state(state) -> dict:
        return {
            "track_id": int(state.track_id),
            "class": state.cls,
            "road": state.road,
            "movement": state.movement,
            "distance_to_intersection": state.distance_to_intersection,
            "approaching_intersection": bool(state.approaching_intersection),
            "cleared": bool(state.cleared),
        }

    def _live_metrics(self, tracked_vehicles) -> dict:
        counts_by_road: dict[str, dict[str, int]] = {
            road: {} for road in self.roads
        }
        for vehicle in tracked_vehicles:
            if vehicle.road in counts_by_road:
                counts_by_road[vehicle.road][vehicle.cls] = (
                    counts_by_road[vehicle.road].get(vehicle.cls, 0) + 1
                )

        metrics = self._empty_metrics()
        for road, counts in counts_by_road.items():
            metrics[road]["density"] = compute_density(counts, self.density_weights)
        return metrics

    def infer_frame(self, frame) -> dict:
        """Run one camera frame through YOLO, tracking, and emergency logic.

        The returned value is a complete dashboard snapshot plus the raw
        frame detections. The model is protected by a lock because a single
        YOLO instance should not be invoked concurrently by multiple clients.
        """
        started = perf_counter()
        with self._inference_lock:
            detector = self._ensure_detector()
            detections = detector.detect(frame)

            now = monotonic()
            if self.live_clock_origin is None:
                self.live_clock_origin = now
            timestamp = max(0.0, now - self.live_clock_origin)

            tracked = self.live_tracker.update(detections, timestamp)
            for vehicle in tracked:
                vehicle.road = assign_road(vehicle.current_position, self.intersection_cfg)

            emergency_states = [
                self.emergency_tracker.update(vehicle, self.intersection_cfg)
                for vehicle in tracked
                if vehicle.cls in EMERGENCY_CLASSES
            ]
            selected = select_priority_emergency(
                emergency_states, self.state_machine.phase
            )

            # This preserves the safety state machine and emergency controller
            # semantics for a camera stream. Normal adaptive phase selection
            # remains owned by the controller/simulation loop.
            with self._lock:
                self.state_machine.tick(timestamp)
                self.emergency_controller.handle(emergency_states, timestamp)

            with self._lock:
                self.latest_detections = [
                    self._serialize_detection(detection) for detection in detections
                ]
                self.latest_vehicles = [
                    self._serialize_tracked_vehicle(vehicle) for vehicle in tracked
                ]
                self.latest_emergency = (
                    self._serialize_emergency_state(selected) if selected is not None else None
                )
                self.latest_metrics = self._live_metrics(tracked)
                self.controller_clock_seconds = timestamp
                self.inference_frame_count += 1
                self.last_inference_ms = (perf_counter() - started) * 1000.0

                payload = self.snapshot()
                payload.update(
                    {
                        "detections": deepcopy(self.latest_detections),
                        "inference_ms": self.last_inference_ms,
                        "frame_size": {
                            "width": int(frame.shape[1]),
                            "height": int(frame.shape[0]),
                        },
                    }
                )
                return payload

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
            "detections": deepcopy(self.latest_detections),
            "signals": self.signals(),
            "emergency": self.emergency(),
            "metrics": self.metrics(),
            "controller": self.status(),
            "inference": self.inference_status(),
        }


_RUNTIME = BackendRuntime()


def _decode_image_payload(payload: bytes):
    """Decode an HTTP/WebSocket image payload into a BGR OpenCV frame."""
    if not payload:
        raise ValueError("image payload is empty")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - requirements provide both
        raise RuntimeError("opencv-python and numpy are required for live inference") from exc

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("payload is not a supported image")
    return frame


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


@app.get("/api/inference/status")
def inference_status():
    """Return YOLO loading and live-frame processing status."""
    return _RUNTIME.inference_status()


@app.post("/api/inference/image")
async def inference_image(request: Request):
    """Run YOLO live inference on one encoded image.

    The request body must be JPEG/PNG/WebP bytes. Sending raw image bytes
    keeps this endpoint dependency-free for browser clients: use ``fetch``
    with ``body: blob`` and the image MIME type as ``Content-Type``.
    """
    try:
        frame = _decode_image_payload(await request.body())
        return await run_in_threadpool(_RUNTIME.infer_frame, frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/inference/reset")
def inference_reset():
    """Reset live tracking when the frontend switches camera/video source."""
    _RUNTIME.reset_live_state()
    return _RUNTIME.inference_status()


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
    """Stream dashboard state and accept encoded camera frames.

    Text messages request the latest snapshot. Binary messages must contain
    one JPEG/PNG/WebP frame; they are decoded, passed through YOLO/tracking,
    and returned as a snapshot containing ``detections`` and inference timing.
    """
    await websocket.accept()
    try:
        await websocket.send_json(_RUNTIME.snapshot())
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                try:
                    frame = _decode_image_payload(message["bytes"])
                    result = await run_in_threadpool(_RUNTIME.infer_frame, frame)
                    await websocket.send_json(result)
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc), "status_code": 400})
                except RuntimeError as exc:
                    await websocket.send_json({"error": str(exc), "status_code": 503})
            else:
                await websocket.send_json(_RUNTIME.snapshot())
    except WebSocketDisconnect:
        pass
