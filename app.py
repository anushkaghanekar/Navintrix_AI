from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import time

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# The nano checkpoint is small enough for free-tier demo instances. Ultralytics
# downloads it on first startup when it is not present in the working tree.
MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.pt")
SAMPLE_PATH = Path(__file__).with_name("inter.jpeg")
DIRECTIONS = ("North", "East", "South", "West")
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
VEHICLE_POINT_VALUES = {
    "bicycle": 1,
    "motorcycle": 2,
    "car": 3,
    "truck": 5,
    "bus": 6,
}
QUEUE_POINTS_PER_METERS = 5
EMERGENCY_POINTS = 30
QUEUE_ZONE_START = 0.58
VEHICLE_LENGTH_METERS = {
    "bicycle": 2,
    "motorcycle": 3,
    "car": 6,
    "truck": 10,
    "bus": 12,
}
EMERGENCY_TERMS = ("ambulance", "fire truck", "firetruck", "police", "emergency")


@dataclass
class DirectionResult:
    direction: str
    image: Image.Image
    annotated: np.ndarray
    counts: dict[str, int]
    vehicle_count: int
    queue_vehicle_count: int
    detected_emergency_count: int
    manual_emergency_count: int
    queue_meters: float
    confidence: float

    @property
    def emergency_count(self) -> int:
        return self.detected_emergency_count + self.manual_emergency_count

    @property
    def vehicle_points(self) -> int:
        return sum(self.counts.get(label, 0) * points for label, points in VEHICLE_POINT_VALUES.items())

    @property
    def queue_points(self) -> int:
        return round(self.queue_meters / QUEUE_POINTS_PER_METERS)

    @property
    def emergency_points(self) -> int:
        return self.emergency_count * EMERGENCY_POINTS

    @property
    def demand_points(self) -> int:
        return max(1, self.vehicle_points + self.queue_points + self.emergency_points)


@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    return YOLO(MODEL_PATH)


def read_image(uploaded_file) -> Image.Image:
    return Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")


def estimate_queue(prediction) -> tuple[int, float]:
    """Estimate a stopped queue from vehicles in the lower road-facing image zone."""
    image_height = prediction.orig_shape[0]
    queue_vehicle_count = 0
    queue_meters = 0.0
    for class_id, box in zip(prediction.boxes.cls.tolist(), prediction.boxes.xyxy.tolist()):
        label = str(prediction.names[int(class_id)]).lower()
        if label not in VEHICLE_LENGTH_METERS:
            continue
        _, top, _, bottom = box
        center_y = ((top + bottom) / 2) / image_height
        if center_y >= QUEUE_ZONE_START:
            queue_vehicle_count += 1
            queue_meters += VEHICLE_LENGTH_METERS[label] + 2
    return queue_vehicle_count, round(queue_meters, 1)


def analyze_direction(direction: str, image: Image.Image, manual_emergency: int, confidence: float) -> DirectionResult:
    model = load_model()
    prediction = model.predict(np.asarray(image), conf=confidence, imgsz=640, verbose=False)[0]
    names = prediction.names
    counts: dict[str, int] = {}
    detected_emergency_count = 0

    for class_id in prediction.boxes.cls.tolist():
        label = str(names[int(class_id)]).lower()
        counts[label] = counts.get(label, 0) + 1
        if any(term in label for term in EMERGENCY_TERMS):
            detected_emergency_count += 1

    vehicle_count = sum(counts.get(label, 0) for label in VEHICLE_CLASSES)
    queue_vehicle_count, queue_meters = estimate_queue(prediction)
    return DirectionResult(
        direction=direction,
        image=image,
        annotated=prediction.plot(),
        counts=counts,
        vehicle_count=vehicle_count,
        queue_vehicle_count=queue_vehicle_count,
        detected_emergency_count=detected_emergency_count,
        manual_emergency_count=manual_emergency,
        queue_meters=queue_meters,
        confidence=confidence,
    )


def adaptive_green_times(
    results: list[DirectionResult],
    cycle_length: int = 120,
    minimum_green: int = 15,
) -> dict[str, int]:
    """Allocate green time so higher demand ALWAYS means more green.

    Waiting-time logic (why this reduces queues):
    1. Each approach earns transparent demand points from vehicles + queue + emergency.
    2. The fixed green budget (cycle minus yellow/all-red changeover) is shared
       proportionally to those points, on top of a fair minimum for every road.
    3. Largest-remainder rounding keeps the cycle exact without breaking shares.
    4. A final rank-enforcement pass guarantees monotonicity: if approach A has
       strictly more points than B, A gets strictly more green (or at worst the
       same when both are pinned at the minimum). Any rounding inversion is fixed
       by swapping the two allocations, which preserves the cycle total.
    5. Emergency approaches are boosted to the maximum green so responders are
       never stuck behind a rounding decision.
    """
    if not results:
        return {}
    changeover_time = 6
    green_budget = cycle_length - (changeover_time * len(results))
    minimum_green = max(1, min(minimum_green, green_budget // len(results)))
    demand_weights = {result.direction: max(1, result.demand_points) for result in results}
    total_demand = sum(demand_weights.values())
    if total_demand <= 0:
        equal = green_budget // len(results)
        greens = {direction: equal for direction in demand_weights}
        greens[list(demand_weights)[0]] += green_budget - sum(greens.values())
        return greens

    remaining_green = max(0, green_budget - (minimum_green * len(results)))
    raw_extra = {
        direction: remaining_green * weight / total_demand
        for direction, weight in demand_weights.items()
    }
    extra = {direction: int(value) for direction, value in raw_extra.items()}
    leftover = remaining_green - sum(extra.values())
    # Largest remainder: hand leftover seconds to the biggest fractional parts.
    for direction, _ in sorted(raw_extra.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[:leftover]:
        extra[direction] += 1
    greens = {direction: minimum_green + extra[direction] for direction in demand_weights}

    # Rank enforcement: sort by demand desc, sort greens desc, re-assign so the
    # busiest approach always holds the largest green. Preserves cycle total.
    ranked_directions = sorted(demand_weights, key=lambda d: (-demand_weights[d], d))
    sorted_greens = sorted(greens.values(), reverse=True)
    greens = {direction: green for direction, green in zip(ranked_directions, sorted_greens)}

    # Emergency pre-emption: the approach with responders gets the max green.
    emergency_rank = sorted(
        (r for r in results if r.emergency_count > 0),
        key=lambda r: (-r.emergency_count, -r.demand_points),
    )
    if emergency_rank:
        leader = emergency_rank[0].direction
        current_max = max(greens.values())
        if greens[leader] < current_max:
            holder = next(d for d in ranked_directions if greens[d] == current_max and d != leader)
            greens[holder], greens[leader] = greens[leader], greens[holder]
    return greens


def signal_phase_order(results: list[DirectionResult]) -> list[str]:
    """Serve approaches in an order that minimizes total vehicle waiting time.

    Greedy highest-demand-first: the approach holding the most waiting vehicles
    goes first, because every second it waits costs `vehicles` vehicle-seconds,
    while delaying a quiet approach only costs a few vehicle-seconds. Emergency
    approaches always jump the queue.
    """
    return [
        result.direction
        for result in sorted(
            results,
            key=lambda r: (-r.emergency_count, -r.demand_points, -r.vehicle_count, r.direction),
        )
    ]


def phase_for_direction(direction: str, phase: str) -> str:
    phase_direction, phase_color = phase.rsplit(" ", 1)
    if direction != phase_direction:
        return "red"
    return phase_color.lower()


def build_signal_schedule(green_times: dict[str, int], phase_order: list[str] | None = None) -> list[tuple[str | None, str, int]]:
    order = list(phase_order) if phase_order else list(DIRECTIONS)
    schedule = []
    for direction in order:
        schedule.append((direction, "green", green_times[direction]))
        schedule.append((direction, "yellow", 4))
        schedule.append((None, "all-red", 2))
    return schedule


@st.fragment(run_every="1s")
def render_live_simulation(green_times: dict[str, int], phase_order: list[str] | None = None) -> None:
    order = list(phase_order) if phase_order else list(DIRECTIONS)
    schedule = build_signal_schedule(green_times, order)
    total_cycle = sum(duration for _, _, duration in schedule)
    elapsed = int(time.time() - st.session_state["simulation_started_at"]) % total_cycle
    elapsed_stage = 0
    active_direction = None
    active_color = "all-red"
    remaining = 0
    for direction, color, duration in schedule:
        if elapsed < elapsed_stage + duration:
            active_direction = direction
            active_color = color
            remaining = elapsed_stage + duration - elapsed
            break
        elapsed_stage += duration

    if active_direction is None:
        headline = "All approaches stopped"
        phase_note = "Changeover clearance"
    else:
        headline = f"{active_direction} {active_color}"
        phase_note = f"{remaining}s remaining"
    st.markdown(f'<div class="live-banner"><div><span class="panel-title">LIVE CYCLE</span><strong>{headline}</strong></div><span>{phase_note}</span></div>', unsafe_allow_html=True)
    st.progress((elapsed + 1) / total_cycle, text=f"Cycle {elapsed + 1}s / {total_cycle}s")

    light_columns = st.columns(4, gap="small")
    for column, direction in zip(light_columns, order):
        with column:
            light_state = active_color if active_direction == direction else "red"
            if active_direction is None:
                light_state = "red"
            st.markdown(
                f'<div class="light-stage"><h3>{direction}</h3>'
                f'<div class="light-row"><span class="light {"red" if light_state == "red" else ""}"></span> RED</div>'
                f'<div class="light-row"><span class="light {"yellow" if light_state == "yellow" else ""}"></span> YELLOW</div>'
                f'<div class="light-row"><span class="light {"green" if light_state == "green" else ""}"></span> GREEN</div></div>',
                unsafe_allow_html=True,
            )


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        :root {
          --bg: #F8FAFC; --surface: #FFFFFF; --ink: #0F172A; --muted: #64748B;
          --line: #E2E8F0; --line-soft: #F1F5F9; --accent: #2563EB; --accent-hover: #1D4ED8;
          --accent-soft: #EFF6FF; --success: #059669; --success-soft: #ECFDF5;
          --warn: #D97706; --danger: #E11D48; --danger-soft: #FFF1F2;
          --radius: 14px; --radius-sm: 10px;
          --shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 16px -4px rgba(15,23,42,.06);
        }
        .stApp { background: var(--bg); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(248,250,252,.8); backdrop-filter: blur(8px); }
        .block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); font-size: .9rem; }
        [data-testid="stSidebar"] h3 { color: var(--ink); font-size: 1rem; font-weight: 600; }
        html, body, .stApp, h1, h2, h3, p, label, span, div { font-family: 'Inter', -apple-system, sans-serif !important; }
        h1 { letter-spacing: -0.03em; font-size: clamp(2rem, 4vw, 3.1rem) !important; line-height: 1.05 !important; max-width: 640px; font-weight: 700 !important; margin-bottom: .6rem !important; }
        h2 { font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
        h3 { font-weight: 600 !important; }
        .eyebrow { color: var(--accent); font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
        .lede { color: var(--muted); font-size: 1rem; max-width: 620px; line-height: 1.6; }
        .hero { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 28px; box-shadow: var(--shadow); margin-bottom: 8px; }
        .topline { display:flex; align-items:center; justify-content:space-between; gap: 12px; margin-bottom: 18px; }
        .brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:1rem; }
        .brand-dot { width:10px; height:10px; border-radius:50%; background:var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
        .model-chip { border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; color: var(--muted); background: var(--bg); font-size: 11.5px; font-weight: 500; white-space: nowrap; }
        .model-chip b { color: var(--success); font-weight: 600; }
        .section-rule { display:flex; align-items:center; gap:12px; margin: 30px 0 6px; }
        .section-rule h2 { margin:0; font-size:1.05rem !important; }
        .section-rule::after { content:''; height:1px; background:var(--line); flex:1; }
        .section-num { font-size:11px; font-weight:700; letter-spacing:.06em; color:var(--accent); background:var(--accent-soft); border-radius:999px; padding:4px 10px; }
        .panel-title { color: var(--muted); font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
        .direction-title { display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; }
        .direction-title strong { font-size:.95rem; }
        .card { background: var(--surface); border:1px solid var(--line); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); }
        .signal { background: var(--surface); border:1px solid var(--line); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); position:relative; overflow:hidden; }
        .signal::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: var(--accent); }
        .signal strong { color: var(--ink); font-size: 2.1rem; font-weight:700; letter-spacing:-0.02em; }
        .signal .sub { color: var(--muted); font-size:.85rem; margin-top:2px; }
        .signal-meta { color: var(--muted); font-size: 11.5px; }
        .point-line { display:inline-block; margin-top:10px; background:var(--line-soft); border-radius:999px; padding:4px 10px; color: var(--ink); font-size: 11.5px; font-weight:600; }
        .live-banner { display:flex; justify-content:space-between; align-items:center; gap: 16px; background: var(--ink); color: white; border-radius: var(--radius); padding: 16px 18px; margin-bottom: 12px; }
        .live-banner strong { display:block; color: white; font-size: 1.3rem; margin-top: 4px; }
        .live-pill { background:#22C55E; color:#052E16; font-size:11px; font-weight:700; border-radius:999px; padding:5px 11px; }
        .light-stage { background: var(--surface); border:1px solid var(--line); border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow); min-height: 178px; }
        .light-stage h3 { color: var(--ink); margin: 0 0 10px; font-size: .95rem; }
        .light-row { display:flex; align-items:center; gap: 10px; color: var(--muted); font-size: 11.5px; font-weight:600; margin: 9px 0; }
        .light { width: 14px; height: 14px; border-radius: 50%; background: #E2E8F0; border:1px solid #CBD5E1; }
        .light.green { background: #22C55E; border-color:#16A34A; box-shadow: 0 0 10px rgba(34,197,94,.5); }
        .light.yellow { background: #F59E0B; border-color:#D97706; box-shadow: 0 0 10px rgba(245,158,11,.45); }
        .light.red { background: #F43F5E; border-color:#E11D48; box-shadow: 0 0 10px rgba(244,63,94,.45); }
        .emergency-note { border:1px solid #FECDD3; border-left:3px solid var(--danger); padding: 12px 14px; background: var(--danger-soft); color: #881337; border-radius: 0 10px 10px 0; font-size:.9rem; }
        .ok-note { border:1px solid #A7F3D0; border-left:3px solid var(--success); padding: 12px 14px; background: var(--success-soft); color: #065F46; border-radius: 0 10px 10px 0; font-size:.9rem; }
        div[data-testid="stFileUploader"] { background: var(--bg); border: 1.5px dashed #CBD5E1; border-radius: 10px; min-height: 92px; }
        div[data-testid="stFileUploader"] section { padding: 11px; }
        div[data-testid="stFileUploader"] small { color: var(--muted); }
        div[data-testid="stFileUploader"]:hover { border-color: var(--accent); background: var(--accent-soft); }
        button[kind="primary"] { background: var(--accent) !important; border: 0 !important; min-height: 46px; font-weight: 600 !important; border-radius: 10px !important; }
        button[kind="primary"]:hover { background: var(--accent-hover) !important; }
        [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--shadow); }
        [data-testid="stMetricValue"] { color: var(--ink); font-weight:700; }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; }
        [data-testid="stAlert"] { border-radius: 10px; }
        [data-testid="stImage"] img { border-radius: 10px; }
        .evidence-card { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:12px; box-shadow:var(--shadow); }
        .ev-title { font-weight:600; font-size:.9rem; margin:2px 2px 8px; display:flex; justify-content:space-between; align-items:center; }
        .ev-badge { font-size:11px; font-weight:600; background:var(--line-soft); border-radius:999px; padding:3px 9px; color:var(--muted); }
        @media (max-width: 800px) { .topline { align-items: flex-start; flex-direction: column; } .model-chip { white-space: normal; } .hero{padding:20px;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Signal Sense", page_icon="🚦", layout="wide")
    render_styles()

    st.markdown(
        '<div class="hero"><div class="topline"><div class="brand"><span class="brand-dot"></span>Signal Sense</div>'
        '<div class="model-chip"><b>●</b> YOLO26m · local inference</div></div>'
        '<div class="eyebrow">Traffic intelligence</div>'
        '<h1>One junction. Four views. A clearer signal plan.</h1>'
        '<p class="lede">Upload the latest frame from each approach. Signal Sense counts road users, measures queues, and recommends fair green times.</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Detection settings")
        confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
        use_sample = st.checkbox("Use bundled sample for every approach", disabled=not SAMPLE_PATH.exists())
        st.caption("The bundled YOLO26m checkpoint is COCO-trained. It detects common road vehicles, but it cannot identify emergency vehicles unless the checkpoint has custom emergency classes.")
        if not SAMPLE_PATH.exists():
            st.warning("No bundled sample image found. Upload four approach images to begin.")

    st.markdown('<div class="section-rule"><span class="section-num">01</span><h2>Add approach images</h2></div>', unsafe_allow_html=True)
    st.caption("One recent frame per approach. Good light and a clear view of the stop line work best.")
    uploaded: dict[str, object | None] = {}
    manual_emergency: dict[str, int] = {}
    image_columns = st.columns(4, gap="small")
    for column, direction in zip(image_columns, DIRECTIONS):
        with column:
            st.markdown(f'<div class="direction-title"><strong>{direction}</strong><span class="panel-title">approach</span></div>', unsafe_allow_html=True)
            uploaded[direction] = st.file_uploader("", type=["jpg", "jpeg", "png"], key=f"upload_{direction}", label_visibility="collapsed")
            manual_emergency[direction] = st.number_input("Emergency override", min_value=0, max_value=10, value=0, step=1, key=f"emergency_{direction}", help="Use this only when the current model cannot identify emergency classes.")
            st.caption("Queue length is estimated automatically from detections in the near-road zone.")

    ready = use_sample or all(uploaded.values())
    if not ready:
        st.info("Add one image for each approach, or enable the sample image in the sidebar.")

    analyze = st.button("Analyze junction", type="primary", use_container_width=True, disabled=not ready)
    if analyze:
        results: list[DirectionResult] = []
        progress = st.progress(0, text="Loading YOLO26m...")
        for index, direction in enumerate(DIRECTIONS):
            if use_sample:
                image = Image.open(SAMPLE_PATH).convert("RGB")
            else:
                image = read_image(uploaded[direction])
            progress.progress(index / len(DIRECTIONS), text=f"Analyzing {direction.lower()} approach...")
            results.append(analyze_direction(direction, image, manual_emergency[direction], confidence))
        progress.progress(1.0, text="Junction analysis complete")
        st.session_state["results"] = results

    results = st.session_state.get("results", [])
    if not results:
        return

    total_vehicles = sum(result.vehicle_count for result in results)
    total_emergency = sum(result.emergency_count for result in results)
    # Busiest = most demand points (vehicles + queue + emergency). Green time
    # is then allocated monotonically from those points, so this approach
    # always receives the longest green.
    priority = max(results, key=lambda result: (result.emergency_count, result.demand_points))
    green_times = adaptive_green_times(results)
    phase_order = signal_phase_order(results)
    ranked_results = sorted(results, key=lambda r: (-r.demand_points, -r.vehicle_count))
    total_points = sum(result.demand_points for result in results)
    green_budget = sum(green_times.values())
    demand_share = {r.direction: (r.demand_points / total_points * 100 if total_points else 0) for r in results}

    st.markdown('<div class="section-rule"><span class="section-num">02</span><h2>Junction overview</h2></div>', unsafe_allow_html=True)
    metrics = st.columns(5)
    metrics[0].metric("Vehicles detected", total_vehicles)
    metrics[1].metric("Demand points", total_points)
    metrics[2].metric("Emergency vehicles", total_emergency)
    metrics[3].metric("Busiest approach", priority.direction)
    metrics[4].metric("Suggested priority", f"{green_times[priority.direction]} sec")

    if total_emergency == 0:
        st.markdown('<div class="emergency-note">No emergency-class detections were returned. This YOLO26m checkpoint is COCO-trained, so use the per-approach override when a responder is visible but not classifiable by the model.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-note">Emergency priority is active. The affected approach gets maximum green first.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rule"><span class="section-num">03</span><h2>Recommended cycle</h2></div>', unsafe_allow_html=True)
    st.caption("120-second cycle · 15-second minimum · car 3 · bus 6 · truck 5 · motorcycle 2 · bicycle 1 · queue 1 pt / 5 m · emergency 30 pts.")
    st.caption(f"Highest demand goes first and gets the most green: {' → '.join(phase_order)}. Serving the longest queue first clears the most vehicle-seconds of waiting per second of green.")
    rank_table = [
        {
            "Rank": i + 1,
            "Approach": r.direction,
            "Points": r.demand_points,
            "Demand share": f"{demand_share[r.direction]:.1f}%",
            "Green": f"{green_times[r.direction]}s",
            "Why": "emergency first" if r.emergency_count > 0 else f"{r.vehicle_count} vehicles · {r.queue_meters:g}m queue",
        }
        for i, r in enumerate(ranked_results)
    ]
    st.dataframe(rank_table, use_container_width=True, hide_index=True)
    signal_columns = st.columns(4, gap="small")
    for column, result in zip(signal_columns, results):
        with column:
            green_time = green_times[result.direction]
            priority_class = " is-priority" if result.direction == priority.direction else ""
            st.markdown(
                f'<div class="signal{priority_class}"><div class="panel-title">{result.direction}'
                f'</div><strong>{green_time}s</strong><div class="sub">{result.vehicle_count} vehicles · {result.queue_meters:g}m queue</div>'
                f'<div class="point-line">{result.demand_points} pts</div><br><span class="signal-meta">vehicles {result.vehicle_points} · queue {result.queue_points} · emergency {result.emergency_points}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-rule"><span class="section-num">04</span><h2>Simulation</h2></div>', unsafe_allow_html=True)
    st.caption("Replays the recommended timings on your still images. Not a live camera feed.")
    live_simulation = st.toggle("Run live cycle", value=False)
    if live_simulation:
        plan_key = tuple(green_times.items()) + tuple(phase_order)
        if st.session_state.get("simulation_plan") != plan_key:
            st.session_state["simulation_plan"] = plan_key
            st.session_state["simulation_started_at"] = time.time()
        render_live_simulation(green_times, phase_order)
    else:
        st.session_state.pop("simulation_started_at", None)
        phases = [f"{direction} {color}" for direction in phase_order for color in ("green", "yellow")]
        selected_phase = st.select_slider("Manual phase preview", options=phases, value=phases[0])
        phase_direction, phase_color = selected_phase.rsplit(" ", 1)
        st.caption(f"Previewing {phase_direction} {phase_color.lower()}. The active green phase receives {green_times[phase_direction]} seconds.")

    st.markdown('<div class="section-rule"><span class="section-num">05</span><h2>Evidence</h2></div>', unsafe_allow_html=True)
    evidence_columns = st.columns(4, gap="small")
    for column, result in zip(evidence_columns, results):
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="ev-title">{result.direction}<span class="ev-badge">{result.demand_points} pts</span></div>', unsafe_allow_html=True)
                st.image(result.annotated, use_container_width=True)
                breakdown = " · ".join(f"{label}: {count}" for label, count in sorted(result.counts.items())) or "No detections"
                st.caption(f"{breakdown}")


if __name__ == "__main__":
    main()
