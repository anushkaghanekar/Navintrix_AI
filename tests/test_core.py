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
