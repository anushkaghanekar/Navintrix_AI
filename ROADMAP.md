# Roadmap — AI-Based Adaptive Traffic Signal Control System

## Decision

Build it — but the **core** scope below, not the full 40-section spec. Every
piece of the original spec is technically sound; the risk isn't feasibility,
it's building all of it shallowly instead of a tight subset deeply. A working
safety state machine + a real emergency-priority demo with measured
before/after numbers is a stronger deliverable — for grading and for
interviews — than a system that nominally has an RL agent and ablation study
but isn't fully evaluated anywhere.

Assumes a team of 3-4 and a 5-6 month window (~20-24 weeks). The plan below
targets 24 weeks with 2 weeks of slack built in; compress to ~20 by cutting
straight to the "if ahead of schedule" stretch items.

## Core scope (build this, evaluate it for real)

1. YOLO11 (n or s — fine-tune, don't train from scratch) on UA-DETRAC for the
   4 normal classes + a curated emergency set for ambulance/fire truck/police
2. ByteTrack for persistent track IDs
3. ROI/road assignment (config-driven) → trajectory-based counting (line
   crossing, not per-frame) → straight/left/right movement detection
4. Per-road metrics: density, queue length, waiting time, traffic flow
5. Adaptive controller: weighted priority score (density + queue + wait +
   flow) with fairness / anti-starvation
6. Safety state machine: green → yellow → all-red → next-green, authoritative
   over both the adaptive controller and emergency logic
7. Emergency detection with approach verification (on an incoming road,
   moving toward the intersection — not just "class present somewhere in
   frame") + movement-level priority + the three conflict cases (arrives
   during green / during red / already inside intersection)
8. SUMO + TraCI, 3 scenarios: balanced, heavy, emergency
9. Two baselines — fixed-time, density-only — compared against the proposed
   controller (Experiment 1) and against emergency priority on/off
   (Experiment 4)
10. Dashboard: video panel with boxes/IDs/counting lines, four-way overview,
    per-road metric cards, signal state, emergency panel
11. README, fixed seeds, dependency versions, run commands — every reported
    number must be reproducible from a fresh clone

## Stretch (only once core is built AND evaluated, in this order)

1. Full 7-configuration ablation study (A: fixed-time → G: +RL)
2. Full 10-scenario matrix (currently trimmed to 3)
3. Multiple-simultaneous-emergency-vehicle policy
4. Reinforcement-learning controller (state/action/reward already sketched
   in the original spec — architecture supports adding it without touching
   the safety FSM)
5. Exhaustive automated test suite (core ships with the load-bearing tests
   only — safety transitions, duplicate-count prevention, emergency priority)
6. Meter-calibrated (not just vehicle-count) queue length

Do not start on stretch items until every core item has been run end-to-end
with real measured numbers. A half-built stretch feature is worse for the
report than not attempting it.

## Timeline (24 weeks)

**Week 1 — Setup**
Repo, environments, config skeleton agreed, dataset accounts created
(AI City Challenge — institutional email for the request form; UA-DETRAC is
directly downloadable). Roles finalized. Backend/dashboard person starts
scaffolding FastAPI + React against mocked data in parallel with everything
below — don't let that person sit idle waiting for real data.

**Weeks 2-6 — Data + Detection**
Prepare UA-DETRAC (YOLO format, splits, leakage checks). Pull a slice of AI
City Challenge Track 1 for counting/movement validation. Curate the emergency
vehicle set (Roboflow sources + manual labeling — budget this as real time,
it's the one dataset step without a one-line download). Fine-tune YOLO11,
evaluate (precision/recall/mAP50/mAP50:95, per-class, confusion matrix).
*Checkpoint: detector hits a reasonable mAP on held-out test data, emergency
classes included, numbers are real.*

**Weeks 6-9 — Tracking + Analytics** (overlaps end of detection phase)
ByteTrack integration, ROI/road config, trajectory-based counting with
duplicate prevention, movement detection, density/queue/wait/flow.
*Checkpoint: a recorded video in, a per-road vehicle count and metrics out,
counts don't double up across frames.*

**Weeks 9-13 — Controller + Safety**
Safety state machine first (this is the one piece that must be right before
anything else touches it). Then the adaptive controller with fairness, then
emergency detection with approach verification and movement-level priority,
then the three conflict cases.
*Checkpoint: feed the controller a synthetic traffic state, watch it produce
a safe phase sequence; feed it an emergency event mid-green, watch it
transition safely instead of snapping red-to-green.*

**Weeks 12-15 — SUMO integration** (overlaps controller phase)
Four-way SUMO network, TraCI wiring so the controller drives real SUMO signal
phases, the 3 core scenarios.
*Checkpoint: controller runs a full SUMO simulation end-to-end without a
human in the loop.*

**Weeks 15-17 — Baselines + core experiments**
Fixed-time and density-only baselines. Run Experiment 1 and Experiment 4
across the 3 scenarios, collect the metrics list from the spec (waiting time,
queue, throughput, emergency response/clearance time).
*Checkpoint: a table of real numbers comparing the three controllers exists.*

**Weeks 15-20 — Backend + dashboard** (mostly parallel, wired to real data
once weeks 9-17 land)
FastAPI endpoints + WebSocket streaming of live metrics/signals/emergency
events, React dashboard panels.

**Weeks 20-22 — Integration, testing, polish**
Run the full end-to-end demo scenario (ambulance arrives mid-cycle, priority
granted, clears, normal control resumes). Load-bearing tests: safety
transitions, duplicate-count prevention, emergency priority, conflicting-green
prevention. Fix what breaks.

**Weeks 22-24 — Documentation + report**
README, architecture diagrams, results generated straight from experiment
output (not hand-typed into the report), limitations and future work section
(this is where RL/full ablation/full scenario matrix get honestly listed as
future work if not attempted).

If the team is ahead at week 20, spend the remaining time on stretch items in
the order listed above, not on padding the core.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Emergency dataset is thin/inconsistent | Combine multiple small Roboflow sources + manual labeling; budget it as its own 1-2 week task, not a side task |
| UA-DETRAC official host is occasionally slow/flaky | Download early, keep a local/mirrored copy, don't depend on re-downloading mid-project |
| AI City Challenge needs an institutional email + approval wait (2-3 business days) for the evaluation server | Not required for the core scope — only needed if you want to submit to their public leaderboard, which isn't part of this plan; the direct dataset download doesn't need it |
| Free-tier GPU quota (Colab/Kaggle) runs out mid-training | Fine-tune, don't train from scratch; use YOLO11n/s, not the largest variant; checkpoint frequently |
| SUMO/TraCI integration eats more time than planned | It's scheduled with 3 weeks of overlap with the controller phase specifically because this tends to run long |
| Team tries to build everything at once | Don't touch the stretch list until core is fully evaluated — see above |
