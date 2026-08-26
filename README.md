# AI-Based Adaptive Traffic Signal Control System (with Emergency Vehicle Priority)

Deep-learning traffic-camera pipeline (YOLO11 + ByteTrack) feeding an adaptive,
safety-constrained signal controller, evaluated in SUMO against fixed-time and
density-only baselines. Emergency vehicles get movement-level priority through
an explicit safety state machine.

**Full plan, timeline, and scope rationale: see [ROADMAP.md](./ROADMAP.md).**
**This is a scaffold, not a finished system** — folders, configs, and module
stubs are set up so the team can start immediately; the detection, tracking,
controller, and simulation logic in each stub is intentionally left as
`# TODO` for the team to design and implement.

## Scope

This scaffold is organized around the **core** build (see ROADMAP.md for the
full core/stretch split and reasoning):

- YOLO11 detector fine-tuned on UA-DETRAC + a small custom emergency-vehicle set
- ByteTrack for persistent IDs
- ROI/road assignment, trajectory-based counting, straight/left/right movement
- Density, queue length, waiting time, flow metrics per approach
- Adaptive controller (density + queue + wait + flow, with fairness) behind a
  safety state machine (yellow → all-red → green)
- Emergency detection with approach verification and movement-level priority
- SUMO/TraCI simulation, 3 core scenarios (balanced / heavy / emergency)
- Fixed-time and density-only baselines, compared against the proposed controller
- FastAPI + WebSocket backend, React dashboard (video panel, 4-way overview,
  metrics, emergency panel)

RL controller, the full 7-way ablation, the full 10-scenario matrix, and
multi-emergency-vehicle conflict policy are stretch goals — see ROADMAP.md.

## Project structure

```text
traffic-ai/
├── configs/          # YAML: intersection ROIs, signal timing, model thresholds
├── data/              # raw/ and processed/ dataset storage (gitignored)
├── datasets/          # dataset-specific working dirs
├── scripts/           # dataset prep / conversion / split / validation scripts
├── detection/         # YOLO detector wrapper + inference
├── tracking/          # ByteTrack integration
├── counting/          # ROI logic, trajectory-based counting, movement detection
├── analytics/         # density, queue, waiting-time, flow calculations
├── emergency/         # emergency detection, tracking, trajectory, priority
├── controller/        # adaptive controller, safety state machine, fairness
├── simulation/         # SUMO network + TraCI integration + scenarios
├── evaluation/        # metrics, experiment runner, plotting
├── backend/            # FastAPI app + WebSocket endpoints
├── frontend/           # React dashboard (see frontend/README.md to init)
└── tests/              # unit / integration / simulation tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

SUMO must be installed separately (`sumo`, `sumo-gui`, and the `SUMO_HOME`
environment variable set) — see https://sumo.dlr.de/docs/Installing/index.html.
It's free and works fine for a zero-budget project.

## Running the SUMO simulation + experiments

The three core scenarios (balanced / heavy / emergency) are committed as
**source** files under `simulation/scenarios/` (network nodes/edges, the
tl_1 traffic-light program, and deterministic route files). The generated
SUMO network (`intersection.net.xml`) is a build artifact and is gitignored.

Requires a SUMO installation (`sumo`, `netconvert` on PATH, SUMO_HOME set),
plus `pip install traci sumolib`.

```bash
# 1. Build the shared network + verify the tl program against config
bash scripts/build_sumo_scenarios.sh

# 2. Run Experiment 1 (fixed-time vs density-only vs proposed) and
#    Experiment 4 (emergency priority on/off) across the 3 scenarios,
#    writing raw result JSONs to evaluation/results/
python scripts/run_experiments.py --scenarios balanced heavy emergency
```

The experiment runners do not fabricate numbers — they only write what the
real TraCI run produced, so the report/plots can trace back to those JSONs.

## Team roles (suggested 4-way split)

| Role | Owns | Primary folders |
|---|---|---|
| Detection & data | Dataset prep, YOLO fine-tuning, evaluation | `scripts/`, `detection/`, `data/` |
| Tracking & analytics | ByteTrack, counting, movement, density/queue/wait/flow | `tracking/`, `counting/`, `analytics/` |
| Controller & simulation | Safety FSM, adaptive controller, emergency priority, SUMO/TraCI | `controller/`, `emergency/`, `simulation/` |
| Backend & dashboard | FastAPI, WebSockets, React UI, experiment plots | `backend/`, `frontend/`, `evaluation/` |

Roles overlap by design at the integration points (emergency detection touches
both detection and controller people; evaluation touches everyone) — that's
intentional, not a scoping error.

## Do not

(Carried over from the original spec — these are the traps that turn a real
project into an unconvincing one.)

- Don't count detections per frame — count on trajectory/line-crossing.
- Don't hardcode ROI/intersection coordinates in application logic — config only.
- Don't let anything bypass the safety state machine, including "just for the demo."
- Don't report a number you didn't actually measure.
