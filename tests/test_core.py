"""Load-bearing tests for the core scope. These are the ones that matter
most — if only a few tests get written, write these.

Fill in each with pytest.skip("TODO") removed once the corresponding
module exists.
"""

import pytest


def test_counting_does_not_double_count_same_track_id():
    """The core counting guarantee: one track_id = one count, even if the
    vehicle crosses the counting line across many frames."""
    from counting.counter import VehicleCounter
    from tracking.bytetrack import TrackedVehicle

    counter = VehicleCounter()
    tv = TrackedVehicle(
        track_id=1, cls="car", confidence=0.9,
        current_position=(400, 150), previous_position=(400, 250),
        first_seen_time=0, last_seen_time=0, road="north",
    )
    # Vehicle visible (and crossing) across N frames.
    for _ in range(50):
        counter.update(tv, crossed_counting_line=True)
    assert counter.counts_by_road == {"north": 1, "south": 0, "east": 0, "west": 0}
    assert counter.counts_by_class == {"car": 1}


_SIGNAL_CFG = {
    "min_green_seconds": 10,
    "max_green_seconds": 60,
    "yellow_seconds": 3,
    "all_red_seconds": 2,
}


def _build(**overrides):
    from controller.state_machine import SafetyStateMachine

    cfg = dict(_SIGNAL_CFG)
    cfg.update(overrides)
    return SafetyStateMachine(cfg)


def test_state_machine_never_skips_yellow_and_all_red():
    from controller.state_machine import SignalPhase

    m = _build()
    assert m.current_green_road() == "north"

    # Request a hand-off after min green is satisfied.
    assert m.request_phase_change("east", 11.0) is True
    assert m.phase is SignalPhase.YELLOW

    # Still yellow before the yellow window elapses.
    m.tick(12.5)
    assert m.phase is SignalPhase.YELLOW

    # Yellow_seconds=3 from t=11 -> all-red no earlier than t=14.
    m.tick(14.0)
    assert m.phase is SignalPhase.ALL_RED
    m.tick(15.0)
    assert m.phase is SignalPhase.ALL_RED

    # all_red_seconds=2 from t=14 -> green on east at t=16.
    m.tick(16.0)
    assert m.phase is SignalPhase.GREEN
    assert m.current_green_road() == "east"
    assert m.green_roads() == ["east"]


def test_state_machine_refuses_conflicting_green():
    """The FSM never lets two approaches hold green simultaneously.

    A request to switch from the current green to a conflicting approach
    is only honored through the full yellow -> all-red -> green sequence,
    so at no instant does green_roads() ever exceed one entry. Requests to
    re-target mid-transition, or for the already-green road, are refused.
    """
    import random

    from controller.state_machine import SignalPhase

    m = _build()
    rng = random.Random(0)
    roads = ("north", "south", "east", "west")

    # Already-green request is refused.
    assert m.green_roads() == ["north"]
    assert m.request_phase_change("north", 5.0) is False

    # Min-green not satisfied -> a conflicting request is refused.
    assert m.request_phase_change("west", 5.0) is False

    # A hand-off request after min: accepted, but green leaves immediately
    # (yellow), so north and west are never green at the same time.
    assert m.request_phase_change("west", 11.0) is True
    assert m.phase is SignalPhase.YELLOW
    # Mid-transition re-target is refused.
    assert m.request_phase_change("east", 12.0) is False
    assert m.request_phase_change("south", 12.0) is False
    # Walk the sequence and assert the one-green invariant at every tick.
    for t in (11.5, 14.0, 15.0, 16.0, 17.0):
        m.tick(t)
        assert len(m.green_roads()) <= 1, f"conflicting green at t={t}"

    # Long chaotic stress: random requests + emergency flips never yield
    # two concurrent greens, and each observed green change passes through
    # BOTH yellow and all-red.
    m = _build()
    t = 0.0
    last_green_road = None
    seen_yellow = seen_all_red = False
    for _ in range(400):
        t += 0.5
        if rng.random() < 0.3:
            road = rng.choice(roads)
            m.request_phase_change(road, t, emergency=(rng.random() < 0.1))
        phase = m.tick(t)
        if phase is SignalPhase.YELLOW:
            seen_yellow = True
        elif phase is SignalPhase.ALL_RED:
            seen_all_red = True
        elif phase is SignalPhase.GREEN:
            road = m.current_green_road()
            if last_green_road is not None and road != last_green_road:
                assert seen_yellow and seen_all_red, "green hand-off skipped a safety stage"
            last_green_road = road
            seen_yellow = seen_all_red = False
        assert len(m.green_roads()) <= 1


def test_state_machine_enforces_min_green():
    m = _build()
    # Before min_green (10s) is met: refused.
    assert m.request_phase_change("south", 5.0) is False
    assert m.request_phase_change("south", 9.5) is False
    # Immediately after the min window, accepted.
    assert m.request_phase_change("south", 10.0001) is True

    # Emergency cuts the min wait but still starts the full sequence.
    m2 = _build()
    assert m2.request_phase_change("west", 1.0, emergency=True) is True
    from controller.state_machine import SignalPhase

    assert m2.phase is SignalPhase.YELLOW


def test_state_machine_enforces_max_green():
    from controller.state_machine import SignalPhase

    # max_green=5 with no incoming requests must still force a rotation.
    m = _build(min_green_seconds=2, max_green_seconds=5, yellow_seconds=1, all_red_seconds=1)
    for t in (1.0, 3.0):
        assert m.tick(t) is SignalPhase.GREEN  # still within max
    m.tick(5.0)  # max reached -> forced off green even with no request
    assert m.phase is SignalPhase.YELLOW
    assert m.tick(6.0) is SignalPhase.ALL_RED
    assert m.tick(7.0) is SignalPhase.GREEN
    assert m.current_green_road() == "east"  # rotation north -> east


def test_state_machine_invalid_timers_raise():
    import pytest

    from controller.state_machine import SafetyStateMachine

    missing = dict(_SIGNAL_CFG)
    del missing["yellow_seconds"]
    with pytest.raises(ValueError):
        SafetyStateMachine(missing)
    with pytest.raises(ValueError):
        _build(yellow_seconds=0)


# ---------------------------------------------------------------------------
# controller/adaptive_controller.py + controller/fairness.py
# ---------------------------------------------------------------------------

def test_priority_score_is_weighted_sum():
    from controller.adaptive_controller import priority_score

    coeffs = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0, "delta": 4.0}
    assert priority_score(1, 2, 3, 4, coeffs) == pytest.approx(1 + 4 + 9 + 16)


def test_fairness_forces_green_after_max_wait():
    from controller.fairness import FairnessTracker

    tracker = FairnessTracker(max_wait_seconds=90.0)
    assert tracker.road_forcing_green(50.0) is None      # nobody starved yet

    tracker = FairnessTracker(max_wait_seconds=90.0)
    assert tracker.road_forcing_green(50.0) is None      # nobody starved yet

    # Everyone granted at 0: by t=120 all exceed the 90s ceiling, so a
    # road IS forced (deterministic tie-break picks the first in iteration).
    for road in ("north", "south", "east", "west"):
        tracker.on_green_granted(road, 0.0)
    assert tracker.road_forcing_green(120.0) in ("north", "south", "east", "west")

    # Make west the clear worst offender (all others got green at 100).
    tracker.on_green_granted("north", 100.0)
    tracker.on_green_granted("south", 100.0)
    tracker.on_green_granted("east", 100.0)
    # west last granted at 0 -> waited 120s, overshoot 30s; others only 20s
    assert tracker.road_forcing_green(120.0) == "west"


def test_adaptive_requests_highest_priority_road():
    from controller.adaptive_controller import AdaptiveController
    from controller.state_machine import SafetyStateMachine, SignalPhase

    cfg = dict(_SIGNAL_CFG)
    sm = SafetyStateMachine(cfg)
    controller_cfg = {"controller": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}}

    class FakeFairness:
        def road_forcing_green(self, t):
            return None

    controller = AdaptiveController(sm, controller_cfg, FakeFairness())
    controller.set_time(11.0)  # past min green so a request is accepted

    metrics = {
        "north": {"density": 10, "queue_length": 2, "waiting_time": 5, "flow": 1},
        "east": {"density": 40, "queue_length": 5, "waiting_time": 20, "flow": 3},
    }
    # east has clearly higher weighted score -> chosen and requested.
    assert controller.choose_next_road(metrics) == "east"
    # Request actually reached the machine.
    assert sm.phase is SignalPhase.YELLOW
    assert sm.pending_road() == "east"


def test_adaptive_fairness_override_wins_over_raw_score():
    from controller.adaptive_controller import AdaptiveController
    from controller.state_machine import SafetyStateMachine, SignalPhase
    from controller.fairness import FairnessTracker

    sm = SafetyStateMachine(dict(_SIGNAL_CFG))
    fairness = FairnessTracker(max_wait_seconds=90.0)
    controller = AdaptiveController(
        sm, {"controller": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}}, fairness
    )
    controller.set_time(120.0)

    # Make west the starved road (last granted at 0); all others recent.
    fairness.on_green_granted("west", 0.0)
    for road in ("north", "south", "east"):
        fairness.on_green_granted(road, 100.0)

    metrics = {
        "north": {"density": 100, "queue_length": 20, "waiting_time": 50, "flow": 5},
        "west": {"density": 0, "queue_length": 0, "waiting_time": 0, "flow": 0},
    }
    # Even though north has the higher raw priority, the starved west is forced.
    assert controller.choose_next_road(metrics) == "west"
    assert sm.pending_road() == "west"


def test_adaptive_empty_metrics_is_noop():
    from controller.adaptive_controller import AdaptiveController

    class FakeSM:
        def __init__(self):
            self.calls = []

        def current_green_road(self):
            return "north"

        def request_phase_change(self, road, now):
            self.calls.append(road)

    class FakeFairness:
        def road_forcing_green(self, t):
            return None

    controller = AdaptiveController(FakeSM(), {}, FakeFairness())
    assert controller.choose_next_road({}) == "north"
    assert controller.state_machine.calls == []


# ---------------------------------------------------------------------------
# emergency/  (detector, tracker, trajectory, priority)
# ---------------------------------------------------------------------------

def _mkv(track_id, road, current, previous, cls="ambulance"):
    """Build a TrackedVehicle on a named road at current/previous positions."""
    from tracking.bytetrack import TrackedVehicle

    return TrackedVehicle(
        track_id=track_id, cls=cls, confidence=0.9,
        current_position=current, previous_position=previous,
        first_seen_time=0, last_seen_time=1, road=road,
    )


def test_emergency_is_approaching_intersection():
    from emergency.detector import is_approaching_intersection

    cfg = _load_intersection()  # center is (480, 480); north approach is the top edge
    # A vehicle on the north approach moving south (down, toward center) -> True.
    approaching = _mkv(1, "north", (480, 150), (480, 100))
    assert is_approaching_intersection(approaching, cfg) is True
    # Moving away (north, up/skyward, away from center) -> False.
    moving_away = _mkv(2, "north", (480, 100), (480, 150))
    assert is_approaching_intersection(moving_away, cfg) is False
    # Parked / no movement -> False.
    parked = _mkv(3, "north", (400, 150), (400, 150))
    assert is_approaching_intersection(parked, cfg) is False
    # No previous position (brand-new track) -> False (conservative).
    fresh = _mkv(4, "north", (400, 150), None)
    assert is_approaching_intersection(fresh, cfg) is False
    # Not on a road -> False.
    off_road = _mkv(5, None, (400, 150), (400, 100))
    assert is_approaching_intersection(off_road, cfg) is False


def test_emergency_tracker_clearance_after_consecutive_frames():
    from emergency.tracker import EmergencyTracker

    cfg = _load_intersection()
    tracker = EmergencyTracker(clearance_confirmation_frames=3)

    # Frame 1: approaching (moving toward center, north approach -> down).
    st = tracker.update(_mkv(1, "north", (480, 150), (480, 100)), cfg)
    assert st.approaching_intersection is True
    assert st.cleared is False
    assert st.distance_to_intersection is not None

    # Next 3 frames: NOT approaching (moving up, away) -> cleared after 3.
    st = None
    for _ in range(3):
        st = tracker.update(_mkv(1, "north", (480, 100), (480, 150)), cfg)
    assert st.cleared is True
    assert st.frames_not_approaching >= 3


def test_emergency_tracker_requires_consecutive_misses_to_clear():
    from emergency.tracker import EmergencyTracker

    cfg = _load_intersection()
    tracker = EmergencyTracker(clearance_confirmation_frames=3)
    # A "not approaching" frame, then a reappearance resets the counter.
    tracker.update(_mkv(1, "north", (480, 150), (480, 100)), cfg)   # approaching
    tracker.update(_mkv(1, "north", (480, 100), (480, 150)), cfg)   # away (miss 1)
    tracker.update(_mkv(1, "north", (480, 150), (480, 100)), cfg)   # approaching -> reset
    st = tracker.update(_mkv(1, "north", (480, 150), (480, 100)), cfg)
    assert st.cleared is False  # the reset prevented premature clearance


def test_emergency_required_movement_infers_and_falls_back():
    from emergency.trajectory import required_movement
    from emergency.tracker import EmergencyVehicleState

    state = EmergencyVehicleState(
        track_id=1, cls="ambulance", road="east",
        movement=None, distance_to_intersection=50.0,
        approaching_intersection=True,
    )
    # Entered east, heading toward north -> east->north = right (config table:
    # a vehicle coming from the east traveling west turns right to head north).
    assert required_movement(state, ["east", "east", "north"]) == "right"
    # Entry east, exit west -> straight.
    assert required_movement(state, ["east", "west"]) == "straight"
    # Entry east, exit south -> left.
    assert required_movement(state, ["east", "south"]) == "left"
    # No meaningful exit yet -> "unknown" (serve whole road).
    assert required_movement(state, ["east", "east"]) == "unknown"


def test_emergency_priority_selects_deterministically():
    from emergency.priority import select_priority_emergency
    from emergency.tracker import EmergencyVehicleState

    def state(track_id, cls, dist, approaching=True):
        return EmergencyVehicleState(
            track_id=track_id, cls=cls, road="north", movement=None,
            distance_to_intersection=dist, approaching_intersection=approaching,
        )

    # No approaching -> None.
    assert select_priority_emergency([state(1, "ambulance", 10, approaching=False)], None) is None
    # Single approaching -> returns it.
    single = state(2, "ambulance", 5)
    assert select_priority_emergency([single], None) is single
    # Multiple: closest first.
    close = state(3, "fire_truck", 2)
    far = state(4, "ambulance", 50)
    assert select_priority_emergency([far, close], None) is close
    # Same distance -> class priority (ambulance wins).
    amb = state(5, "ambulance", 2)
    fire = state(6, "fire_truck", 2)
    assert select_priority_emergency([fire, amb], None) is amb


def test_emergency_priority_only_triggers_when_actually_approaching():
    from emergency.detector import is_approaching_intersection
    from emergency.priority import select_priority_emergency
    from emergency.tracker import EmergencyVehicleState

    cfg = _load_intersection()
    # A tracked vehicle moving AWAY is not approaching -> priority returns None
    # even though it is an emergency class.
    moving_away = _mkv(1, "north", (480, 100), (480, 150))
    assert is_approaching_intersection(moving_away, cfg) is False
    state = EmergencyVehicleState(
        track_id=1, cls="ambulance", road="north", movement=None,
        distance_to_intersection=50.0, approaching_intersection=False,
    )
    assert select_priority_emergency([state], None) is None


def test_emergency_priority_still_goes_through_safety_transition():
    pytest.skip("TODO: controller/emergency_controller.py + state_machine.py")


def test_dataset_split_has_no_video_leakage_across_train_test():
    pytest.skip("TODO: scripts/split_dataset.py")


# ---------------------------------------------------------------------------
# scripts/convert_annotations.py + scripts/prepare_detrac.py
# ---------------------------------------------------------------------------

def test_convert_annotations_to_yolo_format_normalizes_boxes():
    from scripts.convert_annotations import to_yolo_format

    boxes = [{"class_name": "car", "x1": 240, "y1": 0, "x2": 480, "y2": 540}]
    out = to_yolo_format(boxes, {"car": 0}, image_width=960, image_height=540)
    assert out == "0 0.375000 0.500000 0.250000 1.000000"


def test_convert_annotations_clips_bounds_and_drops_unknown():
    from scripts.convert_annotations import to_yolo_format

    # Box sticking out of the frame: clipped to image bounds before normalizing.
    boxes = [{"class_name": "car", "x1": -10, "y1": -20, "x2": 100, "y2": 200}]
    out = to_yolo_format(boxes, {"car": 0}, image_width=960, image_height=540)
    assert out == "0 0.052083 0.185185 0.104167 0.370370"

    # Unknown class contributes no line.
    unknown = [{"class_name": "car", "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    assert to_yolo_format(unknown, {}, 960, 540) == ""

    # Zero-area box contributes no line.
    degenerate = [{"class_name": "car", "x1": 10, "y1": 10, "x2": 10, "y2": 50}]
    assert to_yolo_format(degenerate, {"car": 0}, 960, 540) == ""


def test_prepare_detrac_parse_handles_both_xml_dialects(tmp_path):
    from scripts.prepare_detrac import parse_detrac_xml

    official = tmp_path / "official.xml"
    official.write_text(
        """<sequence name="MVI_00001">
        <frame_num>1</frame_num>
        <target id="1"><attribute vehicle="car"/><box left="0" top="0" width="100" height="60"/></target>
        <target id="2"><attribute vehicle="van"/><box left="10" top="20" width="30" height="40"/></target>
        <frame_num>2</frame_num>
        <target id="3"><attribute vehicle="others"/><box left="5" top="5" width="20" height="20"/></target>
        </sequence>"""
    )
    frames = parse_detrac_xml(official)
    assert frames[1][0]["class_name"] == "car"
    assert frames[1][0]["x2"] == 100.0
    assert frames[1][1] == {"class_name": "van", "x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0}
    assert frames[2][0]["class_name"] == "others"

    wrapped = tmp_path / "wrapped.xml"
    wrapped.write_text(
        """<sequence name="MVI_00002">
        <frame num="7">
          <target id="1"><attribute vehicle="bus"/><box left="5" top="5" width="50" height="70"/></target>
        </frame>
        <frame num="8"></frame>
        </sequence>"""
    )
    frames2 = parse_detrac_xml(wrapped)
    assert frames2[7][0]["class_name"] == "bus"
    assert frames2[7][0]["x2"] == 55.0
    assert frames2[8] == []  # wrapper may declare frames with no targets


def test_prepare_detrac_end_to_end_yolo_layout(tmp_path):
    """Run the full converter on a tiny synthetic UA-DETRAC tree."""
    import json
    from pathlib import Path

    from PIL import Image

    from scripts.prepare_detrac import main

    repo_root = Path(__file__).resolve().parent.parent
    raw = tmp_path / "raw"
    seq_dir = raw / "images" / "train" / "MVI_00001"
    seq_dir.mkdir(parents=True, exist_ok=True)
    xml_dir = raw / "annotations" / "train"
    xml_dir.mkdir(parents=True)
    for frame in (1, 2, 3):
        Image.new("RGB", (960, 540), color=(128, 128, 128)).save(
            seq_dir / f"img{frame:05d}.jpg"
        )
    (xml_dir / "MVI_00001.xml").write_text(
        """<sequence name="MVI_00001">
        <frame_num>1</frame_num>
        <target id="1"><attribute vehicle="car"/><box left="0" top="0" width="100" height="60"/></target>
        <target id="2"><attribute vehicle="van"/><box left="10" top="20" width="30" height="40"/></target>
        <frame_num>2</frame_num>
        <target id="3"><attribute vehicle="bus"/><box left="100" top="50" width="200" height="100"/></target>
        <target id="4"><attribute vehicle="others"/><box left="300" top="60" width="40" height="40"/></target>
        <frame_num>3</frame_num>
        </sequence>"""
    )

    out = tmp_path / "out"
    config = repo_root / "configs" / "model.yaml"
    main(["--raw-dir", str(raw), "--out-dir", str(out), "--config", str(config)])

    assert (out / "images" / "train" / "MVI_00001_img00001.jpg").is_file()
    label_1 = (out / "labels" / "train" / "MVI_00001_img00001.txt")
    assert label_1.exists()
    assert label_1.read_text().splitlines() == [
        "0 0.052083 0.055556 0.104167 0.111111",  # car   (id 0)
        "3 0.026042 0.074074 0.031250 0.074074",  # van -> truck (id 3)
    ]
    label_2 = (out / "labels" / "train" / "MVI_00001_img00002.txt").read_text()
    assert label_2 == "2 0.208333 0.185185 0.208333 0.185185"  # bus (id 2); others dropped
    # Frame with no objects keeps a legitimately empty label file.
    assert (out / "labels" / "train" / "MVI_00001_img00003.txt").read_text() == ""

    summary = json.loads((out / "prep_summary.json").read_text())
    seq = summary["sequences"]["MVI_00001"]
    assert seq["split"] == "train"
    assert seq["images"] == 3
    assert seq["boxes"] == 3
    assert seq["dropped_boxes"] == 1
    assert summary["classes_kept_counts"] == {"car": 1, "motorcycle": 0, "bus": 1, "truck": 1}
    assert summary["classes_dropped_counts"] == {"others": 1}


# ---------------------------------------------------------------------------
# detection/detector.py + detection/inference.py
# ---------------------------------------------------------------------------

def test_detector_class_names_from_config_in_training_order():
    from pathlib import Path

    from detection.detector import load_class_names

    repo_root = Path(__file__).resolve().parent.parent
    names = load_class_names(str(repo_root / "configs" / "model.yaml"))
    assert names == ["car", "motorcycle", "bus", "truck",
                     "ambulance", "fire_truck", "police_vehicle"]


def test_detector_result_to_detections_maps_indexes_and_filters_confidence():
    from detection.detector import _result_to_detections

    class FakeBoxes:
        def __init__(self):
            self.xyxy = [[0, 0, 100, 60], [5, 5, 50, 50], [10, 10, 20, 20]]
            self.conf = [0.9, 0.7, 0.8]
            self.cls = [0, 4, 3]

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class_names = ["car", "motorcycle", "bus", "truck", "ambulance"]
    detections = _result_to_detections(FakeResult(FakeBoxes()), class_names, 0.75)
    # cls 0 -> car (conf 0.9); cls 4 -> ambulance (0.7 below 0.75, dropped);
    # cls 3 -> truck (conf 0.8). Emitted in model order; sorting is detect()'s job.
    assert len(detections) == 2
    assert detections[0].cls == "car"
    assert detections[0].confidence == 0.9
    assert (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2) == (
        0.0, 0.0, 100.0, 60.0,
    )
    assert detections[1].cls == "truck"


def test_detector_detect_runs_predict_and_sorts_by_confidence():
    from detection.detector import VehicleDetector

    det = VehicleDetector.__new__(VehicleDetector)
    det.confidence_threshold = 0.5
    det.iou_threshold = 0.45
    det.class_names = ["car", "motorcycle", "bus", "truck", "ambulance"]

    class FakeBoxes:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class StubModel:
        def predict(self, source, conf, iou, verbose):
            return [
                FakeResult(FakeBoxes([[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.6], [0, 4])),
                FakeResult(FakeBoxes([[50, 50, 60, 60], [0, 0, 1, 1]], [0.95, 0.4], [3, 2])),
            ]

    det.model = StubModel()
    dets = det.detect("whatever-the-frame-is")
    # Merged across both predict results, 0.4 dropped below threshold, sorted desc.
    assert [(d.cls, d.confidence) for d in dets] == [
        ("truck", 0.95),
        ("car", 0.9),
        ("ambulance", 0.6),
    ]


def test_run_on_video_calls_callback_for_every_frame(tmp_path):
    import cv2
    import numpy as np

    from detection.detector import Detection
    from detection.inference import run_on_video

    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    try:
        for i in range(6):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    finally:
        writer.release()

    class FakeDetector:
        def detect(self, frame):
            return [Detection("car", 0.9, 0, 0, 10, 10)]

    seen = []
    count = run_on_video(str(video), FakeDetector(), lambda idx, dets: seen.append((idx, len(dets))))
    assert count == 6
    assert [idx for idx, _ in seen] == list(range(6))
    assert all(n == 1 for _, n in seen)


def test_run_on_video_skips_corrupt_frames_and_honors_max_frames(tmp_path):
    import cv2
    import numpy as np

    from detection.inference import run_on_video

    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    try:
        for i in range(6):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    finally:
        writer.release()

    class FlakyDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            if self.calls % 2 == 1:  # frames 0, 2, 4 raise
                raise RuntimeError("corrupt frame")
            return []

    seen = []
    count = run_on_video(str(video), FlakyDetector(), lambda idx, dets: seen.append(idx))
    # 3 good frames pass through the callback (indices 1, 3, 5); corrupt ones are skipped.
    assert count == 3
    assert seen == [1, 3, 5]

    class CountingDetector:
        def detect(self, frame):
            return []

    assert run_on_video(str(video), CountingDetector(), lambda idx, dets: None, max_frames=2) == 2


def test_run_on_video_raises_for_unopenable_source(tmp_path):
    import pytest

    from detection.inference import run_on_video

    with pytest.raises(IOError):
        run_on_video(str(tmp_path / "does_not_exist.mp4"), object(), lambda *a: None)


# ---------------------------------------------------------------------------
# tracking/bytetrack.py
# ---------------------------------------------------------------------------

def _mkdet(cls, conf, x1, y1, x2, y2):
    from detection.detector import Detection

    return Detection(cls, conf, x1, y1, x2, y2)


def test_tracker_assigns_stable_id_across_frames():
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=30, match_thresh=0.8)
    box = (100, 100, 200, 200)
    ids = []
    for t in range(3):
        out = tracker.update([_mkdet("car", 0.9, *box)], timestamp=t)
        ids.append([tv.track_id for tv in out])
    assert ids == [[1], [1], [1]]


def test_tracker_survives_temporary_occlusion():
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=30, match_thresh=0.8)
    box = (100, 100, 200, 200)

    out0 = tracker.update([_mkdet("car", 0.9, *box)], timestamp=0)
    out1 = tracker.update([_mkdet("car", 0.9, *box)], timestamp=1)
    out2 = tracker.update([], timestamp=2)                      # occluded
    out3 = tracker.update([_mkdet("car", 0.9, *box)], timestamp=3)  # reappears

    assert [tv.track_id for tv in out0] == [1]
    assert [tv.track_id for tv in out1] == [1]
    assert out2 == []  # lost track is withheld while not visible
    assert [tv.track_id for tv in out3] == [1]  # same ID, no double-count
    reappeared = out3[0]
    assert reappeared.previous_position is not None  # last known position preserved
    assert reappeared.first_seen_time == 0.0
    assert reappeared.last_seen_time == 3.0


def test_tracker_distinct_vehicles_get_distinct_ids():
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=10, match_thresh=0.8)
    out = tracker.update(
        [_mkdet("car", 0.9, 0, 0, 50, 50), _mkdet("bus", 0.9, 300, 300, 400, 400)],
        timestamp=0,
    )
    ids = [tv.track_id for tv in out]
    assert ids == [1, 2]
    # Swap detection order next frame; IDs stay glued to physical position.
    out1 = tracker.update(
        [_mkdet("bus", 0.9, 300, 300, 400, 400), _mkdet("car", 0.9, 0, 0, 50, 50)],
        timestamp=1,
    )
    by_pos = {round(tv.current_position[0]): tv.track_id for tv in out1}
    assert by_pos[25.0] == 1     # left vehicle kept ID 1
    assert by_pos[350.0] == 2    # right vehicle kept ID 2


def test_tracker_expires_track_after_buffer_and_issues_new_id():
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=2, match_thresh=0.8)
    box = (100, 100, 200, 200)

    assert [tv.track_id for tv in tracker.update([_mkdet("car", 0.9, *box)], 0)] == [1]
    assert [tv.track_id for tv in tracker.update([_mkdet("car", 0.9, *box)], 1)] == [1]
    assert tracker.update([], 2) == []
    assert tracker.update([], 3) == []
    assert tracker.update([], 4) == []  # 3rd miss past buffer=2 -> removed
    refound = tracker.update([_mkdet("car", 0.9, *box)], 5)
    assert [tv.track_id for tv in refound] == [2]  # brand-new ID, never reused


def test_tracker_low_score_detection_keeps_track_alive_byte_stage():
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=10, match_thresh=0.8)
    box = (100, 100, 200, 200)
    out0 = tracker.update([_mkdet("car", 0.9, *box)], 0)
    # A low-confidence box (below track_thresh=0.5) reappearing: BYTE stage 2
    # should recover the SAME track, not spawn a second one.
    out1 = tracker.update([_mkdet("car", 0.3, *box)], 1)
    assert [tv.track_id for tv in out0] == [1]
    assert [tv.track_id for tv in out1] == [1]
    assert out1[0].confidence == 0.3


def test_hungarian_assignment_matches_bruteforce():
    import itertools

    import numpy as np

    from tracking.bytetrack import _hungarian_assignment

    rng = np.random.default_rng(0)
    for shape in [(2, 2), (3, 3), (2, 4), (4, 2)]:
        cost = rng.random(shape)
        rows, cols = _hungarian_assignment(cost)
        assert len(rows) == min(shape)
        sol = sum(cost[r, c] for r, c in zip(rows, cols))
        # Brute force over all max-cardinality matchings:
        #  n <= m: inject each row to a distinct column.
        #  n >  m: inject each column to a distinct row.
        best = float("inf")
        if shape[0] <= shape[1]:
            for perm in itertools.permutations(range(shape[1]), shape[0]):
                val = sum(cost[i, perm[i]] for i in range(shape[0]))
                best = min(best, val)
        else:
            for perm in itertools.permutations(range(shape[0]), shape[1]):
                val = sum(cost[perm[j], j] for j in range(shape[1]))
                best = min(best, val)
        assert abs(sol - best) < 1e-9


def test_tracker_from_config_reads_tracker_values():
    from pathlib import Path

    from tracking.bytetrack import VehicleTracker

    repo_root = Path(__file__).resolve().parent.parent
    tracker = VehicleTracker.from_config(str(repo_root / "configs" / "model.yaml"))
    assert tracker.track_buffer == 30
    assert tracker.match_thresh == 0.8
    assert tracker.track_thresh == 0.5
    assert tracker.low_match_thresh == 0.5


def test_tracker_moving_vehicle_keeps_single_id():
    """A vehicle translating between frames (IoU < 0.8) must not fragment.

    BYTE's cost gate is 1 - IoU <= match_thresh (default 0.8 -> accepts
    IoU >= 0.2), not a minimum-IoU gate; otherwise a vehicle that moves
    more than ~a quarter of its box width per frame gets re-identified
    every frame and counting would over-count.
    """
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=30, match_thresh=0.8)
    ids = []
    for t, x in enumerate([0, 10, 20, 30, 40, 50]):
        out = tracker.update([_mkdet("car", 0.9, x, 100, x + 60, 200)], timestamp=t)
        ids.append([tv.track_id for tv in out])
    assert ids == [[1], [1], [1], [1], [1], [1]]


def test_tracker_reset_restarts_ids():
    from tracking.bytetrack import VehicleTracker

    tracker = VehicleTracker(track_buffer=10, match_thresh=0.8)
    box = (100, 100, 200, 200)
    assert [tv.track_id for tv in tracker.update([_mkdet("car", 0.9, *box)], 0)] == [1]
    tracker.reset()
    assert [tv.track_id for tv in tracker.update([_mkdet("car", 0.9, *box)], 0)] == [1]


# ---------------------------------------------------------------------------
# counting/roi.py + counting/counter.py + counting/movement.py
# ---------------------------------------------------------------------------

def _load_intersection():
    from pathlib import Path

    from counting.roi import load_roi_config

    repo_root = Path(__file__).resolve().parent.parent
    return load_roi_config(str(repo_root / "configs" / "intersection.yaml"))


def test_roi_assigns_road_by_polygon():
    from counting.roi import assign_road

    cfg = _load_intersection()
    assert assign_road((400, 150), cfg) == "north"    # top approach
    assert assign_road((400, 850), cfg) == "south"    # bottom approach
    assert assign_road((150, 500), cfg) == "west"     # left approach
    assert assign_road((850, 400), cfg) == "east"     # right approach
    assert assign_road((480, 480), cfg) is None       # intersection center gap


def test_roi_config_validation_requires_all_roads():
    from counting.roi import load_roi_config

    # Missing a required road should raise.
    import tempfile
    import textwrap

    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    temp.write(
        textwrap.dedent(
            """\
            roads:
              north:
                roi: [[0,0],[1,0],[1,1],[0,1]]
                queue_roi: []
                counting_line: []
                allowed_movements: [straight]
            """
        )
    )
    temp.close()
    try:
        import pytest

        with pytest.raises(ValueError):
            load_roi_config(temp.name)
    finally:
        import os

        os.unlink(temp.name)


def test_crosses_counting_line_segment_intersection():
    from counting.roi import crosses_counting_line

    cfg = _load_intersection()
    # North counting_line is y=200 between x=360..600.
    # Moving from (480, 300) to (480, 400) never touches y=200 -> no crossing.
    assert not crosses_counting_line((480, 300), (480, 400), "north", cfg)
    # Moving from (480, 300) to (480, 150) crosses y=200.
    assert crosses_counting_line((480, 300), (480, 150), "north", cfg)
    # A lane shift that stays below the line does not cross it.
    assert not crosses_counting_line((360, 120), (480, 150), "north", cfg)
    # An unknown / unconfigured road must not raise — it just never crosses.
    assert not crosses_counting_line((480, 300), (480, 150), "", cfg)
    assert not crosses_counting_line((480, 300), (480, 150), "bogus", cfg)


def test_counter_counts_track_once_not_per_frame():
    from counting.counter import VehicleCounter
    from tracking.bytetrack import TrackedVehicle

    counter = VehicleCounter()
    # 50 frames of the same track on the same road, all crossing the line:
    # must result in exactly 1 count.
    for t in range(50):
        counter.update(
            TrackedVehicle(
                track_id=7, cls="car", confidence=0.9,
                current_position=(400, 150),
                previous_position=(400, 250),
                first_seen_time=0, last_seen_time=t,
                road="north",
            ),
            crossed_counting_line=True,
        )
    assert counter.counts_by_road == {"north": 1, "south": 0, "east": 0, "west": 0}
    assert counter.counts_by_class == {"car": 1}


def test_counter_separates_vehicles_by_track_id():
    from counting.counter import VehicleCounter
    from tracking.bytetrack import TrackedVehicle

    counter = VehicleCounter()
    base = dict(
        confidence=0.9,
        current_position=(400, 150),
        previous_position=(400, 200),
        first_seen_time=0, last_seen_time=0,
        road="north",
    )
    counter.update(TrackedVehicle(track_id=1, cls="car", **base), True)
    counter.update(TrackedVehicle(track_id=2, cls="truck", **base), True)
    # Same track_id crosses again -> no increment.
    counter.update(TrackedVehicle(track_id=1, cls="car", **base), True)
    assert counter.counts_by_road == {"north": 2, "south": 0, "east": 0, "west": 0}
    assert counter.counts_by_class == {"car": 1, "truck": 1}


def test_counter_ignores_non_crossing_and_unassigned_road():
    from counting.counter import VehicleCounter
    from tracking.bytetrack import TrackedVehicle

    counter = VehicleCounter()
    base = dict(
        track_id=1, cls="car", confidence=0.9,
        current_position=(0, 150), previous_position=(0, 200),
        first_seen_time=0, last_seen_time=0,
    )
    counter.update(TrackedVehicle(road="north", **base), False)  # not crossing -> no count
    assert counter.counts_by_road["north"] == 0
    counter.update(TrackedVehicle(road=None, **base), True)  # road unknown -> "unassigned"
    assert counter.counts_by_road.get("unassigned") == 1


def test_movement_classify_uses_config_table():
    from counting.movement import classify_movement

    table = {
        "east": {"west": "straight", "north": "left", "south": "right"},
        "north": {"south": "straight", "east": "left", "west": "right"},
    }
    assert classify_movement("east", "west", table) == "straight"
    assert classify_movement("east", "north", table) == "left"
    assert classify_movement("north", "west", table) == "right"
    assert classify_movement("east", "east", table) == "unknown"  # no U-turn

    # Default table comes from configs/intersection.yaml.
    assert classify_movement("west", "east") == "straight"


# ---------------------------------------------------------------------------
# analytics/  (density, queue, waiting_time, traffic_flow)
# ---------------------------------------------------------------------------

def test_density_weighted_sum():
    from analytics.density import compute_density

    weights = {"car": 1.0, "motorcycle": 0.5, "bus": 2.5, "truck": 2.5}
    assert compute_density({"car": 4, "bus": 2}, weights) == pytest.approx(9.0)
    # A class absent from weights contributes 0 (no error).
    assert compute_density({"car": 4, "police": 9}, weights) == pytest.approx(4.0)


def test_queue_length_counts_stopped_inside_region():
    from analytics.queue import estimate_queue_length
    from tracking.bytetrack import TrackedVehicle

    queue_roi = [[0, 0], [100, 0], [100, 100], [0, 100]]
    stopped_prev = (50 + 0.2, 50 + 0.2)  # tiny drift (< 1px) -> "stopped"
    stopped = TrackedVehicle(
        track_id=1, cls="car", confidence=0.9,
        current_position=(50, 50), previous_position=stopped_prev,
        first_seen_time=0, last_seen_time=1, road="north",
    )
    moving = TrackedVehicle(
        track_id=2, cls="car", confidence=0.9,
        current_position=(50, 50), previous_position=(30, 30),
        first_seen_time=0, last_seen_time=1, road="north",
    )
    outside = TrackedVehicle(
        track_id=3, cls="car", confidence=0.9,
        current_position=(500, 500), previous_position=(501, 501),
        first_seen_time=0, last_seen_time=1, road="north",
    )
    no_prev = TrackedVehicle(
        track_id=4, cls="car", confidence=0.9,
        current_position=(50, 50), previous_position=None,
        first_seen_time=0, last_seen_time=1, road="north",
    )
    assert estimate_queue_length([stopped, moving, outside, no_prev], queue_roi) == 1
    assert estimate_queue_length([], queue_roi) == 0


def test_waiting_time_accumulates_and_aggregates():
    from analytics.waiting_time import WaitingTimeTracker

    tracker = WaitingTimeTracker()
    tracker.on_enter_queue(1, 10.0, road="north")   # enters queue at t=10
    assert tracker.current_wait(1, 15.0) == pytest.approx(5.0)
    assert tracker.on_depart(1, 25.0) == pytest.approx(15.0)
    assert tracker.current_wait(1, 30.0) == 0.0        # no longer queued
    assert tracker.road_average("north") == pytest.approx(15.0)
    assert tracker.road_max("north") == pytest.approx(15.0)

    # Repeat entry survives; depart is idempotent.
    tracker.on_enter_queue(2, 40.0, road="south")
    tracker.on_enter_queue(2, 41.0, road="south")       # already queued, ignored
    assert tracker.on_depart(2, 50.0) == pytest.approx(10.0)
    assert tracker.on_depart(2, 60.0) == 0.0            # already departed

    # Never queued -> 0.
    assert tracker.on_depart(999, 60.0) == 0.0


def test_traffic_flow_window_and_rate():
    from analytics.traffic_flow import FlowTracker

    tracker = FlowTracker(window_seconds=60.0)
    # Crossings at t=5, 55, 58 (all within 60s of each other as of t=58).
    for t in (5.0, 55.0, 58.0):
        tracker.on_crossing("north", t)
    assert tracker.current_flow("north", now=58.0) == pytest.approx(3.0)

    # Querying later slides the window: the t=5 crossing drops out by t=110.
    assert tracker.current_flow("north", now=110.0) == pytest.approx(2.0)

    # By t=130 the earliest survivor (t=55) has also aged out.
    assert tracker.current_flow("north", now=130.0) == pytest.approx(0.0)

    # Unknown road just has zero flow.
    assert tracker.current_flow("bogus") == 0.0
