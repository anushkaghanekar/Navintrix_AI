"""Bridges controller/ (SafetyStateMachine + AdaptiveController +
EmergencyController) to a running SUMO simulation via TraCI.

This is the piece that turns "we have a controller" into "we have a
controller that actually drives traffic lights in a simulated
intersection".

Coordinate-domain note
----------------------
The video pipeline works in *pixel* space; SUMO vehicles live in
*network-meter* space. Our configs/intersection.yaml ROI/counting-line
polygons are calibrated for a camera image, so they must NOT be applied to
SUMO coordinates. Road membership here therefore comes from TraCI's edge
IDs via the config-driven ``simulation.edge_to_road`` mapping, and the
per-road metrics (density/queue/waiting) are computed from the SUMO vehicle
snapshot directly. The pure helpers in this module are unit-testable
without SUMO; ``run_simulation`` itself imports TraCI lazily and requires
a real SUMO installation plus a scenario built under simulation/scenarios/.

Phase authority
---------------
This bridge never invents a signal state: it reads the
SafetyStateMachine's current phase (GREEN/YELLOW/ALL_RED + road) and maps it
to an index in the SUMO signal program defined for the intersection
(netedit). The mapping lives in configs/signal.yaml's ``simulation`` block
and is validated here; a conflict-free green is guaranteed by the state
machine's single-green model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from analytics.density import load_density_weights
from analytics.traffic_flow import FlowTracker
from emergency.detector import _is_moving_toward
from emergency.tracker import EmergencyVehicleState

DEFAULT_CONFIG_PATH = "configs/signal.yaml"
DEFAULT_ROADS = ("north", "south", "east", "west")
DEFAULT_EMERGENCY_CLASSES = ("ambulance", "fire_truck", "police_vehicle")


def load_simulation_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load and validate the ``simulation`` block of configs/signal.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        sim = cfg["simulation"]
    except (KeyError, TypeError):
        raise ValueError(f"{config_path} must define a 'simulation' block")
    if not isinstance(sim, dict):
        raise ValueError(f"{config_path}: 'simulation' must be a mapping")
    validate_simulation_config(sim)
    return sim


def validate_simulation_config(sim: dict, roads: tuple[str, ...] = DEFAULT_ROADS) -> None:
    """Structural validation of the simulation block. Raises ValueError if
    required keys are missing or the roads are not all covered."""
    required = ("traffic_light_id", "phase_index_green", "phase_index_yellow",
                "phase_index_all_red", "edge_to_road")
    missing = [k for k in required if k not in sim]
    if missing:
        raise ValueError(f"simulation config missing required key(s): {', '.join(missing)}")
    for key in ("phase_index_green", "phase_index_yellow"):
        mapping = sim[key]
        if not isinstance(mapping, dict):
            raise ValueError(f"simulation.{key} must be a mapping road->phase index")
        uncovered = [r for r in roads if r not in mapping]
        if uncovered:
            raise ValueError(f"simulation.{key} missing road(s): {', '.join(uncovered)}")
    if not isinstance(sim["edge_to_road"], dict) or not sim["edge_to_road"]:
        raise ValueError("simulation.edge_to_road must be a non-empty edge->road mapping")
    center = sim.get("intersection_center_m")
    if center is not None:
        if (
            not isinstance(center, (list, tuple))
            or len(center) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in center)
        ):
            raise ValueError(
                "simulation.intersection_center_m must be null or an [x, y] pair of numbers"
            )
    vclass_map = sim.get("vehicle_class_mapping")
    if vclass_map is not None:
        if not isinstance(vclass_map, dict):
            raise ValueError(
                "simulation.vehicle_class_mapping must be a mapping SUMO vClass -> class name"
            )
        bad = [k for k, v in vclass_map.items() if not isinstance(k, str) or not isinstance(v, str)]
        if bad:
            raise ValueError(
                f"simulation.vehicle_class_mapping entries must be str->str, offending keys: {bad}"
            )


@dataclass
class SimVehicle:
    """A single snapshot of a SUMO vehicle, shaped for our pipeline.

    ``road`` is derived from ``edge_id`` via ``edge_to_road``.
    """
    vehicle_id: str
    cls: str
    edge_id: str
    position: tuple[float, float]   # network coordinates (meters)
    speed: float                    # m/s
    waiting_time: float             # seconds stopped (TraCI getWaitingTime)
    road: str | None = None


# ---- METRICS + PHASE MAPPING ----
def road_for_vehicle(vehicle: SimVehicle, edge_to_road: dict) -> str | None:
    """Map a vehicle's TraCI edge-id to our approach name (or None)."""
    return edge_to_road.get(vehicle.edge_id)


def assign_roads(vehicles: list[SimVehicle], edge_to_road: dict) -> None:
    """Fill each vehicle's ``road`` from the edge->road mapping (in place)."""
    for vehicle in vehicles:
        vehicle.road = road_for_vehicle(vehicle, edge_to_road)


def build_road_metrics(
    vehicles: list[SimVehicle],
    edge_to_road: dict,
    density_weights: dict,
    stopped_speed_threshold: float = 0.1,
) -> dict[str, dict]:
    """Per-road metrics for a single SUMO snapshot, in the shape the
    AdaptiveController consumes: {road: {density, queue_length,
    waiting_time, flow}}.

    * density       : class-weighted count (missing weights -> 1.0)
    * queue_length  : vehicles on the road with speed < stopped_speed_threshold
    * waiting_time  : the road's MAX TraCI waiting time (anti-starvation signal)
    * flow          : 0 here; the simulation loop fills it via a FlowTracker

    ``flow`` is deliberately left 0.0 for a single snapshot (flow needs a
    time window); run_simulation sets it per-step.
    """
    assign_roads(vehicles, edge_to_road)
    # Sorted so the result dict's key order (and therefore controller
    # tie-breaks) is reproducible across runs — ROADMAP requires fixed seeds.
    roads = sorted(set(edge_to_road.values()))
    result = {
        road: {"density": 0.0, "queue_length": 0, "waiting_time": 0.0, "flow": 0.0}
        for road in roads
    }
    for vehicle in vehicles:
        road = vehicle.road
        if road is None:
            continue
        row = result[road]
        row["density"] += float(density_weights.get(vehicle.cls, 1.0))
        if vehicle.speed < stopped_speed_threshold:
            row["queue_length"] += 1
        row["waiting_time"] = max(row["waiting_time"], float(getattr(vehicle, "waiting_time", 0.0)))
    return result


def signal_state_for_fsm(state_machine, sim_config: dict) -> int:
    """Map the SafetyStateMachine's current phase to a SUMO signal-program
    phase index (config-driven). Returns:
      ALL_RED   -> phase_index_all_red
      YELLOW    -> phase_index_yellow[the road exiting green / current]
      GREEN     -> phase_index_green[current green road]
    """
    phase = state_machine.phase
    if phase.name == "ALL_RED":
        return sim_config["phase_index_all_red"]
    road = state_machine.current_green_road() or state_machine.outgoing_road()
    if road is None:
        return sim_config["phase_index_all_red"]
    name = "YELLOW" if phase.name == "YELLOW" else "GREEN"
    return sim_config[f"phase_index_{name.lower()}"][road]


# ---- VEHICLE SNAPSHOT ----
def snapshot_vehicles(traci_module, vclass_to_cls: dict) -> list[SimVehicle]:
    """Pull the current per-vehicle snapshot out of a running TraCI session.

    Each vehicle's SUMO vClass is translated through ``vclass_to_cls``
    (configs/signal.yaml ``simulation.vehicle_class_mapping``); vehicles
    whose vClass is not mapped are skipped entirely — the same policy as the
    vision pipeline, which only ever sees its trained class set.
    """
    vehicles: list[SimVehicle] = []
    api = traci_module.vehicle
    for vid in api.getIDList():
        cls = vclass_to_cls.get(api.getVehicleClass(vid))
        if cls is None:
            continue
        x, y = api.getPosition(vid)
        vehicles.append(
            SimVehicle(
                vehicle_id=str(vid),
                cls=str(cls),
                edge_id=str(api.getRoadID(vid)),
                position=(float(x), float(y)),
                speed=float(api.getSpeed(vid)),
                waiting_time=float(api.getWaitingTime(vid)),
            )
        )
    return vehicles


def build_emergency_states(
    vehicles: list[SimVehicle],
    previous_positions: dict[str, tuple[float, float]],
    center_m: tuple[float, float],
    emergency_classes: set[str],
    clearance_confirmation_steps: int,
    id_registry: dict[str, int],
    clearance_counters: dict[str, int],
) -> list[EmergencyVehicleState]:
    """Emergency-relevant states for one SUMO snapshot.

    This is the simulation-domain equivalent of EmergencyTracker.update():
    approach verification reuses the SAME geometric rule as the video
    pipeline (emergency/detector._is_moving_toward) but in meter space
    against the config-supplied intersection center — pixel ROIs must never
    be applied to network coordinates. Clearance keeps the consecutive-
    steps semantics of emergency.tracker (a vehicle must fail to approach
    for ``clearance_confirmation_steps`` consecutive snapshots before it is
    marked cleared).

    Mutates ``id_registry`` (stable int track ids for string SUMO ids) and
    ``clearance_counters`` ({vehicle_id -> consecutive non-approaching
    count}); both are owned by the caller so state survives across steps.
    """
    states: list[EmergencyVehicleState] = []
    for vehicle in vehicles:
        if vehicle.cls not in emergency_classes or vehicle.road is None:
            continue
        track_id = id_registry.setdefault(vehicle.vehicle_id, len(id_registry) + 1)
        previous = previous_positions.get(vehicle.vehicle_id)
        approaching = previous is not None and _is_moving_toward(
            previous, vehicle.position, center_m
        )
        frames_not_approaching = (
            0 if approaching else clearance_counters.get(vehicle.vehicle_id, 0) + 1
        )
        clearance_counters[vehicle.vehicle_id] = frames_not_approaching
        distance = (
            (vehicle.position[0] - center_m[0]) ** 2
            + (vehicle.position[1] - center_m[1]) ** 2
        ) ** 0.5
        states.append(
            EmergencyVehicleState(
                track_id=track_id,
                cls=vehicle.cls,
                road=vehicle.road,
                movement=None,
                distance_to_intersection=distance,
                approaching_intersection=approaching,
                cleared=frames_not_approaching >= clearance_confirmation_steps,
                frames_not_approaching=frames_not_approaching,
            )
        )
    return states


# ---- SIMULATION LOOP ----
def run_simulation(
    sumocfg_path: str,
    state_machine,
    adaptive_controller,
    emergency_controller,
    max_steps: int,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    intersection_config_path: str = "configs/intersection.yaml",
    use_gui: bool = False,
) -> dict:
    """Run a SUMO scenario end-to-end under control of our controllers.

    Per step (keyword-only args are paths to configs; defaults match the
    repo layout):
      1. advance SUMO one step and read the simulation clock;
      2. tick() the SafetyStateMachine FIRST, recording any green grant to
         the adaptive controller's fairness tracker (bookkeeping of observed
         transitions — never phase-setting);
      3. snapshot vehicles from TraCI and build per-road metrics
         (density/queue/waiting snapshot + sliding-window flow);
      4. let the EmergencyController handle any verified emergency approach;
         while it returns True the adaptive controller stays paused;
      5. otherwise ask the AdaptiveController for its choice — both request
         phases exclusively through SafetyStateMachine.request_phase_change;
      6. DISPLAY the machine's current phase in SUMO by applying the
         config-mapped signal-program index (this bridge has no authority
         over *which* phase shows — only over translating it to TraCI).

    TraCI is imported lazily so the pure helpers stay importable without a
    SUMO installation; a missing install raises RuntimeError with setup
    instructions instead of an opaque ImportError at import time.

    Returns aggregate metrics shaped for evaluation/:
      scenario, steps, sim_seconds,
      throughput_by_road / throughput_total   (edge-transition crossings),
      waiting_avg_by_road / waiting_max_by_road (completed waits, seconds),
      avg_queue_length_by_road                 (mean stopped count per step),
      final_flow_by_road                       (vehicles/min at the end),
      signal_phase_applications + applied_phase_sequence (display audit),
      emergency_active_steps.
    """
    try:
        import traci
    except ImportError as exc:
        raise RuntimeError(
            "TraCI is required to run a simulation. Install SUMO (set SUMO_HOME) "
            "and 'pip install traci sumolib'."
        ) from exc

    sim_config = load_simulation_config(config_path)
    density_weights = load_density_weights(intersection_config_path)

    with open(config_path) as f:
        full_config = yaml.safe_load(f) or {}
    # Same persistence rule as the video pipeline: N consecutive misses clear.
    clearance_steps = int(
        full_config.get("emergency", {}).get("clearance_confirmation_frames", 15)
    )

    tl_id = sim_config["traffic_light_id"]
    edge_to_road = sim_config["edge_to_road"]
    vclass_to_cls = dict(sim_config.get("vehicle_class_mapping") or {})
    stopped_speed = float(sim_config.get("stopped_speed_threshold_mps", 0.1))
    flow_window = float(sim_config.get("flow_window_seconds", 60.0))
    center_raw = sim_config.get("intersection_center_m")
    center_m = tuple(float(v) for v in center_raw) if center_raw is not None else None
    emergency_classes = set(DEFAULT_EMERGENCY_CLASSES)

    roads = sorted(set(edge_to_road.values()))
    flow_tracker = FlowTracker(window_seconds=flow_window)
    throughput = {road: 0 for road in roads}
    completed_waits: dict[str, list[float]] = {road: [] for road in roads}
    queue_sum = {road: 0.0 for road in roads}

    last_edge: dict[str, str] = {}
    last_position: dict[str, tuple[float, float]] = {}
    stop_accum: dict[str, float] = {}
    emergency_ids: dict[str, int] = {}
    emergency_counters: dict[str, int] = {}

    fairness = getattr(adaptive_controller, "fairness", None)
    applied_phases: list[int] = []
    last_phase_index: int | None = None
    emergency_active_steps = 0
    steps_done = 0
    now = 0.0

    command = ["sumo-gui" if use_gui else "sumo", "-c", str(sumocfg_path)]
    traci.start(command)
    try:
        for _ in range(int(max_steps)):
            prev_now = now
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            step_dt = max(0.0, now - prev_now)

            prev_phase_name = state_machine.phase.name
            prev_green_road = state_machine.current_green_road()

            # The FSM advances before anything consults it this step.
            state_machine.tick(now)
            granted = (
                state_machine.phase.name == "GREEN"
                and (
                    prev_phase_name != "GREEN"
                    or state_machine.current_green_road() != prev_green_road
                )
            )
            if granted and fairness is not None:
                fairness.on_green_granted(state_machine.current_green_road(), now)

            vehicles = snapshot_vehicles(traci, vclass_to_cls)
            assign_roads(vehicles, edge_to_road)
            road_metrics = build_road_metrics(
                vehicles, edge_to_road, density_weights, stopped_speed
            )
            for road in roads:
                road_metrics[road]["flow"] = flow_tracker.current_flow(road, now=now)
                queue_sum[road] += road_metrics[road]["queue_length"]

            # Departure detection: a vehicle LEAVING its approach edge is the
            # simulation-native counting-line crossing — fires exactly once
            # per vehicle per approach, so counts cannot double up.
            current_edges = {v.vehicle_id: v.edge_id for v in vehicles}
            current_positions = {v.vehicle_id: v.position for v in vehicles}
            for vid, prev_edge in last_edge.items():
                if current_edges.get(vid) == prev_edge:
                    continue
                road = edge_to_road.get(prev_edge)
                if road is None:
                    continue
                flow_tracker.on_crossing(road, now)
                throughput[road] += 1
                completed_waits[road].append(stop_accum.pop(vid, 0.0))
            last_edge = current_edges

            # Accumulate stopped time on approach edges (delay proxy).
            for v in vehicles:
                if v.road is not None and v.speed < stopped_speed:
                    stop_accum[v.vehicle_id] = stop_accum.get(v.vehicle_id, 0.0) + step_dt

            active_emergencies: list[EmergencyVehicleState] = []
            if center_m is not None:
                active_emergencies = build_emergency_states(
                    vehicles,
                    last_position,
                    center_m,
                    emergency_classes,
                    clearance_steps,
                    emergency_ids,
                    emergency_counters,
                )
            last_position = current_positions

            # Drop bookkeeping for vehicles that left the simulation.
            gone = set(stop_accum) - set(current_edges)
            for vid in gone:
                stop_accum.pop(vid, None)
                emergency_counters.pop(vid, None)
                emergency_ids.pop(vid, None)

            # Always consult the emergency controller (even with no active
            # emergencies) so its mode correctly returns to EMERGENCY_PASSED
            # once the vehicle has gone; it returns False in that case and
            # the adaptive controller resumes.
            emergency_active = bool(emergency_controller.handle(active_emergencies, now))
            if emergency_active:
                emergency_active_steps += 1
            else:
                adaptive_controller.set_time(now)
                adaptive_controller.choose_next_road(road_metrics)

            phase_index = signal_state_for_fsm(state_machine, sim_config)
            if phase_index != last_phase_index:
                traci.trafficlight.setPhase(tl_id, phase_index)
                applied_phases.append(phase_index)
                last_phase_index = phase_index

            steps_done += 1

        final_metrics = {
            "scenario": Path(str(sumocfg_path)).stem,
            "steps": steps_done,
            "sim_seconds": now,
            "throughput_by_road": dict(throughput),
            "throughput_total": sum(throughput.values()),
            "waiting_avg_by_road": {
                road: (sum(w) / len(w) if w else 0.0)
                for road, w in completed_waits.items()
            },
            "waiting_max_by_road": {
                road: (max(w) if w else 0.0) for road, w in completed_waits.items()
            },
            "avg_queue_length_by_road": {
                road: (queue_sum[road] / steps_done if steps_done else 0.0)
                for road in roads
            },
            "final_flow_by_road": {
                road: flow_tracker.current_flow(road, now=now) for road in roads
            },
            "signal_phase_applications": len(applied_phases),
            "applied_phase_sequence": applied_phases,
            "emergency_active_steps": emergency_active_steps,
        }
        return final_metrics
    finally:
        traci.close()