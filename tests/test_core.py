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
    """Emergency priority cuts the min-green wait but NEVER skips yellow/all-red.

    Drive the real SafetyStateMachine + EmergencyController + priority with a
    single approaching ambulance, then walk the timers and assert the exact
    GREEN -> YELLOW -> ALL_RED -> GREEN(road) sequence — the three safety
    stages are mandatory even for an emergency.
    """
    from controller.emergency_controller import EmergencyController
    from controller.state_machine import SafetyStateMachine, SignalPhase
    from emergency.priority import select_priority_emergency
    from emergency.tracker import EmergencyVehicleState
    from emergency.trajectory import required_movement

    signal = dict(_SIGNAL_CFG)  # min=10, yellow=3, all_red=2, maxgreen=60
    sm = SafetyStateMachine(signal)

    class FakePriority:
        def select_priority_emergency(self, emergencies, phase):
            return select_priority_emergency(emergencies, phase)

    class FakeTrajectory:
        def required_movement(self, state, history):
            return required_movement(state, history)

    controller = EmergencyController(sm, FakePriority(), FakeTrajectory())

    # A single approaching ambulance on the south approach.
    amb = EmergencyVehicleState(
        track_id=1, cls="ambulance", road="south", movement=None,
        distance_to_intersection=100.0, approaching_intersection=True,
    )

    # t=3: well before min_green (10s) — an emergency request still accepted,
    # but the FSM immediately leaves GREEN for YELLOW.
    assert controller.handle([amb], 3.0) is True
    assert sm.phase is SignalPhase.YELLOW
    assert sm.green_roads() == []  # no conflicting green remains

    # 3s of yellow (t=3..6), then all-red for 2s (t=6..8), then green.
    sm.tick(6.0)
    assert sm.phase is SignalPhase.ALL_RED
    sm.tick(8.0)
    assert sm.phase is SignalPhase.GREEN
    assert sm.current_green_road() == "south"
    # The safety sequence was never skipped: we saw yellow then all-red.
    assert sm.pending_road() is None


def test_dataset_split_has_no_video_leakage_across_train_test(tmp_path):
    """Verify that split_dataset groups frames by sequence prefix and never
    leaks any sequence across train/val/test splits.
    """
    from PIL import Image
    from scripts.split_dataset import split_dataset
    from scripts.validate_dataset import validate_dataset

    raw_dir = tmp_path / "raw_dataset"
    raw_dir.mkdir(parents=True)

    # Create 5 sequences, each with 4 frames
    seq_names = [f"SEQ_{i:02d}" for i in range(1, 6)]
    for seq in seq_names:
        for f in range(1, 5):
            img_file = raw_dir / f"{seq}_frame_{f:04d}.jpg"
            Image.new("RGB", (64, 64), color=(100, 100, 100)).save(img_file)
            lbl_file = raw_dir / f"{seq}_frame_{f:04d}.txt"
            lbl_file.write_text("0 0.5 0.5 0.2 0.2\n")

    out_dir = tmp_path / "split_dataset"
    splits = split_dataset(
        data_dir=raw_dir,
        out_dir=out_dir,
        train_frac=0.6,
        val_frac=0.2,
        seed=123,
        class_names=["car"],
    )

    train_seqs = set(splits["train"])
    val_seqs = set(splits["val"])
    test_seqs = set(splits["test"])

    # All sequences accounted for
    assert train_seqs | val_seqs | test_seqs == set(seq_names)

    # Strict zero-leakage guarantee
    assert train_seqs.isdisjoint(val_seqs)
    assert train_seqs.isdisjoint(test_seqs)
    assert val_seqs.isdisjoint(test_seqs)

    # Check generated files
    assert (out_dir / "data.yaml").is_file()
    assert (out_dir / "images" / "train").is_dir()
    assert len(list((out_dir / "images" / "train").glob("*.jpg"))) == len(train_seqs) * 4

    # Validate the split dataset
    val_summary = validate_dataset(out_dir, sample_visual_checks=2)
    assert val_summary["valid_images"] == 20
    assert val_summary["corrupt_images"] == 0
    assert val_summary["missing_labels"] == 0
    assert val_summary["total_boxes"] == 20
    assert val_summary["invalid_boxes"] == 0



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


# ---------------------------------------------------------------------------
# simulation/sumo.py + simulation/traci_controller.py
# ---------------------------------------------------------------------------

def _sim_cfg_dict():
    """A valid ``simulation`` block (mirrors configs/signal.yaml's indices)."""
    return {
        "traffic_light_id": "tl_1",
        "phase_index_green": {"north": 0, "south": 0, "east": 2, "west": 2},
        "phase_index_yellow": {"north": 1, "south": 1, "east": 3, "west": 3},
        "phase_index_all_red": 4,
        "edge_to_road": {"N0": "north", "S0": "south", "E0": "east", "W0": "west"},
    }


def test_sim_config_validation_requires_core_keys():
    from simulation.traci_controller import validate_simulation_config

    with pytest.raises(ValueError, match="required key"):
        validate_simulation_config({})
    cfg = _sim_cfg_dict()
    del cfg["phase_index_green"]["west"]
    with pytest.raises(ValueError, match="west"):
        validate_simulation_config(cfg)
    cfg = _sim_cfg_dict()
    cfg["edge_to_road"] = {}
    with pytest.raises(ValueError, match="edge_to_road"):
        validate_simulation_config(cfg)


def test_sim_config_validation_checks_optional_keys():
    from simulation.traci_controller import validate_simulation_config

    cfg = _sim_cfg_dict()
    cfg["intersection_center_m"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="intersection_center_m"):
        validate_simulation_config(cfg)
    cfg = _sim_cfg_dict()
    cfg["vehicle_class_mapping"] = {"passenger": 3}
    with pytest.raises(ValueError, match="vehicle_class_mapping"):
        validate_simulation_config(cfg)
    cfg = _sim_cfg_dict()
    cfg["intersection_center_m"] = None
    cfg["vehicle_class_mapping"] = {"passenger": "car"}
    validate_simulation_config(cfg)  # a fully valid block passes silently


def test_load_simulation_config_reads_repo_yaml():
    from pathlib import Path

    from simulation.traci_controller import load_simulation_config

    repo_root = Path(__file__).resolve().parent.parent
    sim = load_simulation_config(str(repo_root / "configs" / "signal.yaml"))
    assert sim["traffic_light_id"] == "tl_1"
    assert set(sim["edge_to_road"].values()) == {"north", "south", "east", "west"}
    assert set(sim["vehicle_class_mapping"].values()) <= {
        "car", "motorcycle", "bus", "truck",
        "ambulance", "fire_truck", "police_vehicle",
    }


def test_sumo_scenario_loader_validates_names(tmp_path, monkeypatch):
    import simulation.sumo as sumo_mod

    monkeypatch.setattr(sumo_mod, "SCENARIO_DIR", tmp_path)
    assert sumo_mod.available_scenarios() == []
    with pytest.raises(FileNotFoundError):
        sumo_mod.load_scenario_config("balanced")
    (tmp_path / "balanced.sumocfg").write_text("<configuration/>")
    assert sumo_mod.available_scenarios() == ["balanced"]
    assert sumo_mod.load_scenario_config("balanced").endswith("balanced.sumocfg")
    with pytest.raises(ValueError):
        sumo_mod.load_scenario_config("../outside")


def test_road_assignment_from_edge_ids():
    from simulation.traci_controller import SimVehicle, assign_roads, road_for_vehicle

    def veh(edge):
        return SimVehicle(
            vehicle_id="v", cls="car", edge_id=edge,
            position=(0.0, 0.0), speed=0.0, waiting_time=0.0,
        )

    mapping = {"N0": "north", "E0": "east"}
    assert road_for_vehicle(veh("N0"), mapping) == "north"
    assert road_for_vehicle(veh(":tl_1_0"), mapping) is None  # internal edge
    vehicles = [veh("N0"), veh("E0"), veh(":tl_1_0")]
    assign_roads(vehicles, mapping)
    assert [v.road for v in vehicles] == ["north", "east", None]


def test_build_road_metrics_aggregates_one_snapshot():
    from simulation.traci_controller import SimVehicle, build_road_metrics

    def veh(cls, edge, speed, wait, road=None):
        return SimVehicle(
            vehicle_id=cls + edge, cls=cls, edge_id=edge,
            position=(0.0, 0.0), speed=speed, waiting_time=wait, road=road,
        )

    vehicles = [
        veh("car", "N0", 0.0, 7.0),        # stopped on north
        veh("bus", "S0", 5.0, 0.0),        # moving on south
        veh("truck", "E0", 0.05, 3.0),     # below threshold -> queued
        veh("motorcycle", "X1", 0.0, 9.0), # unmapped edge -> ignored
        veh("ambulance", "W0", 8.0, 0.0),  # class without a weight -> 1.0
    ]
    weights = {"car": 1.0, "bus": 2.5, "truck": 2.5}
    metrics = build_road_metrics(vehicles, _sim_cfg_dict()["edge_to_road"], weights)
    assert set(metrics) == {"north", "south", "east", "west"}
    assert metrics["north"] == {"density": 1.0, "queue_length": 1, "waiting_time": 7.0, "flow": 0.0}
    assert metrics["south"]["density"] == pytest.approx(2.5)
    assert metrics["south"]["queue_length"] == 0
    assert metrics["east"] == {"density": 2.5, "queue_length": 1, "waiting_time": 3.0, "flow": 0.0}
    assert metrics["west"]["density"] == pytest.approx(1.0)


def test_signal_state_mapping_uses_outgoing_road_for_yellow():
    from simulation.traci_controller import signal_state_for_fsm

    sim_cfg = _sim_cfg_dict()
    m = _build()
    assert m.phase.name == "GREEN"
    assert signal_state_for_fsm(m, sim_cfg) == 0            # green[north]

    assert m.request_phase_change("west", 11.0, emergency=True) is True
    assert m.phase.name == "YELLOW"
    # Yellow displays the EXITING road's transition phase (north -> 1);
    # the incoming road's yellow index (west -> 3) must not be used yet.
    assert signal_state_for_fsm(m, sim_cfg) == 1

    m.tick(14.0)
    assert m.phase.name == "ALL_RED"
    assert signal_state_for_fsm(m, sim_cfg) == 4

    m.tick(16.0)
    assert m.phase.name == "GREEN"
    assert signal_state_for_fsm(m, sim_cfg) == 2            # green[west]


def test_snapshot_vehicles_maps_vclasses_and_drops_unknown():
    from types import SimpleNamespace

    from simulation.traci_controller import snapshot_vehicles

    rows = {
        "v1": ("passenger", "N0", (1.0, 2.0), 0.0, 4.0),
        "v2": ("bicycle", "N0", (3.0, 4.0), 1.0, 0.0),
        "v3": ("emergency", "W0", (-5.0, 5.0), 6.0, 0.0),
    }
    api = SimpleNamespace(
        getIDList=lambda: ["v1", "v2", "v3"],
        getVehicleClass=lambda vid: rows[vid][0],
        getRoadID=lambda vid: rows[vid][1],
        getPosition=lambda vid: rows[vid][2],
        getSpeed=lambda vid: rows[vid][3],
        getWaitingTime=lambda vid: rows[vid][4],
    )
    vehicles = snapshot_vehicles(
        SimpleNamespace(vehicle=api),
        {"passenger": "car", "emergency": "ambulance"},
    )
    assert [v.vehicle_id for v in vehicles] == ["v1", "v3"]  # bicycle dropped
    assert vehicles[0].cls == "car"
    assert vehicles[0].position == (1.0, 2.0)
    assert vehicles[1].cls == "ambulance"
    assert vehicles[1].speed == 6.0
    assert vehicles[1].waiting_time == 0.0


def test_sim_emergency_states_verify_approach_in_meter_space():
    from simulation.traci_controller import SimVehicle, build_emergency_states

    center = (0.0, 0.0)
    id_registry, counters = {}, {}

    def veh(vid, pos):
        return SimVehicle(
            vehicle_id=vid, cls="ambulance", edge_id="W0",
            position=pos, speed=5.0, waiting_time=0.0, road="west",
        )

    # First sight: no previous position -> conservatively "not approaching".
    states = build_emergency_states(
        [veh("a", (-40.0, 5.0))], {}, center, {"ambulance"}, 3, id_registry, counters,
    )
    assert states[0].approaching_intersection is False
    assert states[0].cleared is False

    # Moving toward the center -> approaching, stable int id assigned.
    states = build_emergency_states(
        [veh("a", (-35.0, 5.0))], {"a": (-40.0, 5.0)}, center,
        {"ambulance"}, 3, id_registry, counters,
    )
    assert states[0].approaching_intersection is True
    assert states[0].track_id == 1
    assert states[0].distance_to_intersection == pytest.approx((35.0 ** 2 + 5.0 ** 2) ** 0.5)

    # Moving away three consecutive snapshots -> cleared.
    for _ in range(3):
        states = build_emergency_states(
            [veh("a", (-50.0, 5.0))], {"a": (-45.0, 5.0)}, center,
            {"ambulance"}, 3, id_registry, counters,
        )
    assert counters["a"] == 3
    assert states[0].cleared is True

    # Non-emergency classes never surface as emergencies.
    car = SimVehicle(
        vehicle_id="c", cls="car", edge_id="N0",
        position=(0.0, -10.0), speed=1.0, waiting_time=0.0, road="north",
    )
    assert build_emergency_states([car], {}, center, {"ambulance"}, 3, {}, {}) == []


class _FakeTraCI:
    """Scripted stand-in for the traci module (lets us test the full loop
    without SUMO installed).

    Each step is a list of vehicle tuples:
        (id, vclass, edge, (x, y), speed_mps, waiting_seconds)
    Clock advances 1 simulated second per step; every applied signal phase
    is recorded as (time, tls_id, program_index).
    """

    def __init__(self, steps, invariant_check=None):
        from types import SimpleNamespace

        self.steps = steps
        self._invariant = invariant_check
        self._t = 0.0
        self._i = -1
        self.current = []
        self.started = None
        self.closed = False
        self.applied = []

        self.simulation = SimpleNamespace(
            getTime=lambda: self._t,
        )
        # Real traci exposes simulationStep() at module level.
        self.simulationStep = self._advance
        self.vehicle = SimpleNamespace(
            getIDList=lambda: [v[0] for v in self.current],
            getVehicleClass=lambda vid: self._find(vid)[1],
            getRoadID=lambda vid: self._find(vid)[2],
            getPosition=lambda vid: self._find(vid)[3],
            getSpeed=lambda vid: self._find(vid)[4],
            getWaitingTime=lambda vid: self._find(vid)[5],
        )
        self.trafficlight = SimpleNamespace(setPhase=self._apply_phase)

    def _find(self, vid):
        for v in self.current:
            if v[0] == vid:
                return v
        raise KeyError(vid)

    def _advance(self):
        self._i += 1
        self._t += 1.0
        self.current = list(self.steps[self._i])

    def _apply_phase(self, tls_id, index):
        if self._invariant is not None:
            self._invariant(index)
        self.applied.append((self._t, tls_id, index))

    def start(self, command):
        self.started = command
        self._t = 0.0
        self._i = -1
        self.current = []
        self.closed = False
        self.applied = []

    def close(self):
        self.closed = True


def _write_sim_config_with_center(tmp_path):
    """Repo signal config with the intersection center measured (meters)."""
    import yaml

    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((repo_root / "configs" / "signal.yaml").read_text())
    cfg["simulation"]["intersection_center_m"] = [0.0, 0.0]
    path = tmp_path / "signal_test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _scripted_steps():
    """22 one-second steps:
    - car_n queues on the north approach (t=1..12), departs onto E1 at t=13;
    - car_s sits queued on south the whole time;
    - an ambulance drives in on west (t=5..14), straight at the center.
    """
    steps = []
    for t in range(1, 23):
        vehs = []
        if t <= 12:
            vehs.append(("car_n", "passenger", "N0", (2.0, -30.0), 0.0, float(t)))
        else:
            vehs.append(("car_n", "passenger", "E1", (60.0, -30.0), 8.0, 0.0))
        vehs.append(("car_s", "passenger", "S0", (0.0, 30.0), 0.0, float(t)))
        if 5 <= t <= 14:
            vehs.append(("amb1", "emergency", "W0",
                         (-40.0 + 5.0 * (t - 5), 5.0), 5.0, 0.0))
        steps.append(vehs)
    return steps


def test_run_simulation_drives_safety_fsm_through_sumo(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    from controller.adaptive_controller import AdaptiveController
    from controller.emergency_controller import EmergencyController
    from controller.fairness import FairnessTracker
    from controller.state_machine import SafetyStateMachine
    from emergency.priority import select_priority_emergency
    from emergency.trajectory import required_movement
    from simulation.traci_controller import run_simulation

    repo_root = Path(__file__).resolve().parent.parent
    config_path = _write_sim_config_with_center(tmp_path)
    intersection_cfg = str(repo_root / "configs" / "intersection.yaml")

    sm = SafetyStateMachine(dict(_SIGNAL_CFG))
    adaptive = AdaptiveController(
        sm,
        {"controller": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}},
        FairnessTracker(max_wait_seconds=90.0),
    )

    class PriorityShim:
        def select_priority_emergency(self, emergencies, phase):
            return select_priority_emergency(emergencies, phase)

    class TrajectoryShim:
        def required_movement(self, state, history):
            return required_movement(state, history)

    emergency = EmergencyController(sm, PriorityShim(), TrajectoryShim())

    def no_conflicting_green(index):
        # Invoked on every signal application: the FSM holds at most one
        # green, and the index applied to SUMO is exactly the one mapped
        # from the FSM's current state (directional authority check — the
        # SUMO program itself may pair north+south into one phase).
        assert len(sm.green_roads()) <= 1

    fake = _FakeTraCI(_scripted_steps(), invariant_check=no_conflicting_green)
    monkeypatch.setitem(sys.modules, "traci", fake)

    metrics = run_simulation(
        "scenarios/balanced.sumocfg", sm, adaptive, emergency, max_steps=22,
        config_path=config_path, intersection_config_path=intersection_cfg,
    )

    assert fake.started[:2] == ["sumo", "-c"]
    assert fake.closed is True
    assert metrics["steps"] == 22
    assert metrics["sim_seconds"] == pytest.approx(22.0)

    # Display audit: green(north) -> yellow(north) -> all-red -> green(west)
    # -> ... -> yellow(west). The ambulance's emergency request lands at t=6,
    # BEFORE min_green (10s) elapses — proof that emergency cuts the min-green
    # floor — but the yellow/all-red stages are still fully observed.
    seq = []
    for idx in metrics["applied_phase_sequence"]:
        if not seq or seq[-1] != idx:
            seq.append(idx)
    assert seq == [0, 1, 4, 2, 3]
    first_yellow_t = next(t for t, _, idx in fake.applied if idx == 1)
    assert first_yellow_t == pytest.approx(6.0)
    assert first_yellow_t < _SIGNAL_CFG["min_green_seconds"]

    # Active from first verified approach (t=6) until the ambulance reaches
    # the center (t=12); once past it, displacement is no longer toward the
    # intersection, so approach verification conservatively drops it.
    assert metrics["emergency_active_steps"] == 7
    # Both departures counted once each (the emergency vehicle counts too,
    # same as the video pipeline's class-agnostic line-crossing count).
    assert metrics["throughput_by_road"] == {
        "north": 1, "south": 0, "east": 0, "west": 1,
    }
    assert metrics["throughput_total"] == 2
    assert metrics["waiting_avg_by_road"]["north"] == pytest.approx(12.0)
    assert metrics["waiting_max_by_road"]["north"] == pytest.approx(12.0)
    assert metrics["waiting_avg_by_road"]["south"] == 0.0  # never departed
    assert metrics["avg_queue_length_by_road"]["south"] == pytest.approx(1.0)
    assert metrics["final_flow_by_road"]["north"] == pytest.approx(1.0)
    assert metrics["final_flow_by_road"]["west"] == pytest.approx(1.0)


def test_run_simulation_reports_missing_traci_clearly(monkeypatch):
    import sys

    from simulation.traci_controller import run_simulation

    monkeypatch.setitem(sys.modules, "traci", None)
    with pytest.raises(RuntimeError, match="SUMO"):
        run_simulation("x.sumocfg", None, None, None, 1)


# ---------------------------------------------------------------------------
# evaluation/ (metrics.py, experiments.py, plots.py)
# ---------------------------------------------------------------------------

def test_evaluation_metrics_calculations():
    from evaluation.metrics import (
        average_waiting_time,
        max_waiting_time,
        throughput,
        emergency_response_time,
        emergency_clearance_time,
    )

    # Waiting time
    assert average_waiting_time([]) == 0.0
    assert average_waiting_time([10.0, 20.0, 30.0]) == pytest.approx(20.0)
    assert max_waiting_time([]) == 0.0
    assert max_waiting_time([10.0, 45.0, 30.0]) == pytest.approx(45.0)

    # Throughput
    assert throughput(10, 0) == 0.0
    assert throughput(10, -5) == 0.0
    assert throughput(30, 60.0) == pytest.approx(0.5)

    # Emergency times
    assert emergency_response_time(10.0, 15.0) == pytest.approx(5.0)
    assert emergency_response_time(20.0, 15.0) == 0.0  # clamped to >= 0
    assert emergency_clearance_time(15.0, 27.0) == pytest.approx(12.0)
    assert emergency_clearance_time(30.0, 25.0) == 0.0


def test_evaluation_plots_generates_image_files(tmp_path):
    from evaluation.plots import plot_waiting_time_comparison, plot_emergency_response

    sample_results_exp1 = {
        ("fixed_time", "balanced"): {"waiting_avg_by_road": {"north": 15.0, "south": 10.0}},
        ("density_only", "balanced"): {"waiting_avg_by_road": {"north": 12.0, "south": 8.0}},
        ("proposed", "balanced"): {"waiting_avg_by_road": {"north": 9.0, "south": 6.0}},
    }
    out_exp1 = tmp_path / "exp1.png"
    plot_waiting_time_comparison(sample_results_exp1, str(out_exp1))
    assert out_exp1.is_file()
    assert out_exp1.stat().st_size > 0

    sample_results_exp4 = {
        ("emergency_priority_off", "emergency"): {"emergency_active_steps": 25},
        ("emergency_priority_on", "emergency"): {"emergency_active_steps": 8},
    }
    out_exp4 = tmp_path / "exp4.png"
    plot_emergency_response(sample_results_exp4, str(out_exp4))
    assert out_exp4.is_file()
    assert out_exp4.stat().st_size > 0


def test_experiments_runners_with_fake_traci(tmp_path, monkeypatch):
    import json
    import sys
    from evaluation.experiments import run_experiment_1, run_experiment_4

    config_path = _write_sim_config_with_center(tmp_path)
    fake = _FakeTraCI(_scripted_steps())
    monkeypatch.setitem(sys.modules, "traci", fake)

    # Mock scenario path
    scen_file = tmp_path / "balanced.sumocfg"
    scen_file.write_text("<configuration/>")

    res1 = run_experiment_1(
        [str(scen_file)], max_steps=10, config_path=config_path,
        output_dir=str(tmp_path / "results"),
    )
    assert ("fixed_time", "balanced") in res1
    assert ("density_only", "balanced") in res1
    assert ("proposed", "balanced") in res1
    assert res1[("proposed", "balanced")]["throughput_rate_total"] >= 0.0
    exp1_records = json.loads((tmp_path / "results" / "experiment_1_results.json").read_text())
    assert {row["controller"] for row in exp1_records} == {
        "fixed_time", "density_only", "proposed",
    }
    assert all(row["scenario"] == "balanced" for row in exp1_records)

    res4 = run_experiment_4(
        [str(scen_file)], max_steps=10, config_path=config_path,
        output_dir=str(tmp_path / "results"),
    )
    assert ("emergency_priority_on", "balanced") in res4
    assert ("emergency_priority_off", "balanced") in res4
    exp4_records = json.loads((tmp_path / "results" / "experiment_4_results.json").read_text())
    assert {row["controller"] for row in exp4_records} == {
        "emergency_priority_on", "emergency_priority_off",
    }


# ---------------------------------------------------------------------------
# backend/main.py
# ---------------------------------------------------------------------------

def test_backend_endpoints_expose_config_and_fsm_state():
    from fastapi.testclient import TestClient

    import backend.main as backend_mod

    backend_mod._RUNTIME.reset()
    client = TestClient(backend_mod.app)

    intersection = client.get("/api/intersection")
    assert intersection.status_code == 200
    body = intersection.json()
    assert body["intersection"]["name"] == "main_intersection"
    assert set(body["roads"]) == {"north", "south", "east", "west"}

    signals = client.get("/api/signals")
    assert signals.status_code == 200
    assert signals.json()["phase"] == "GREEN"
    assert signals.json()["current_green_road"] == "north"
    assert signals.json()["signals_by_road"]["north"] == "GREEN"
    assert signals.json()["remaining_phase_seconds"] == pytest.approx(60.0)

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert set(metrics.json()) == {"north", "south", "east", "west"}
    assert metrics.json()["north"]["density"] == 0.0

    mode = client.post("/api/controller/mode", params={"mode": "density_only"})
    assert mode.status_code == 200
    assert mode.json()["mode"] == "DENSITY_ONLY"
    assert client.post("/api/controller/mode", params={"mode": "bogus"}).status_code == 400


def test_backend_start_runs_simulation_background_task(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.main as backend_mod

    backend_mod._RUNTIME.reset()
    scen_file = tmp_path / "balanced.sumocfg"
    scen_file.write_text("<configuration/>")

    def fake_run_simulation(
        sumocfg_path,
        state_machine,
        adaptive_controller,
        emergency_controller,
        max_steps,
        **kwargs,
    ):
        assert sumocfg_path == str(scen_file)
        assert max_steps == 3
        assert state_machine.request_phase_change("east", 11.0, emergency=True) is True
        return {
            "scenario": "balanced",
            "steps": 3,
            "sim_seconds": 3.0,
            "throughput_by_road": {"north": 1, "south": 0, "east": 0, "west": 0},
            "throughput_total": 1,
            "waiting_avg_by_road": {"north": 4.0, "south": 0.0, "east": 0.0, "west": 0.0},
            "waiting_max_by_road": {"north": 4.0, "south": 0.0, "east": 0.0, "west": 0.0},
            "avg_queue_length_by_road": {"north": 1.0, "south": 0.0, "east": 0.0, "west": 0.0},
            "final_flow_by_road": {"north": 1.0, "south": 0.0, "east": 0.0, "west": 0.0},
            "signal_phase_applications": 1,
            "applied_phase_sequence": [0],
            "emergency_active_steps": 0,
        }

    monkeypatch.setattr(backend_mod, "run_simulation", fake_run_simulation)
    client = TestClient(backend_mod.app)

    started = client.post(
        "/api/controller/start",
        params={"scenario": str(scen_file), "max_steps": 3},
    )
    assert started.status_code == 200
    assert started.json()["scenario"] == "balanced"

    status = client.get("/api/controller/status").json()
    assert status["running"] is False
    assert status["last_error"] is None
    assert status["latest_results"]["steps"] == 3

    traffic = client.get("/api/traffic").json()
    assert traffic["north"]["queue"] == pytest.approx(1.0)
    assert traffic["north"]["waiting_seconds"] == pytest.approx(4.0)
