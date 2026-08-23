"""Load-bearing tests for the core scope. These are the ones that matter
most — if only a few tests get written, write these.

Fill in each with pytest.skip("TODO") removed once the corresponding
module exists.
"""

import pytest


def test_counting_does_not_double_count_same_track_id():
    pytest.skip("TODO: implement once counting/counter.py exists")


def test_state_machine_never_skips_yellow_and_all_red():
    pytest.skip("TODO: implement once controller/state_machine.py exists")


def test_state_machine_refuses_conflicting_green():
    pytest.skip("TODO: implement once controller/state_machine.py exists")


def test_state_machine_enforces_min_green():
    pytest.skip("TODO")


def test_state_machine_enforces_max_green():
    pytest.skip("TODO")


def test_emergency_priority_only_triggers_when_actually_approaching():
    pytest.skip("TODO: emergency/detector.py's is_approaching_intersection")


def test_emergency_priority_still_goes_through_safety_transition():
    pytest.skip("TODO: controller/emergency_controller.py + state_machine.py")


def test_fairness_forces_green_after_max_wait():
    pytest.skip("TODO: controller/fairness.py")


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
