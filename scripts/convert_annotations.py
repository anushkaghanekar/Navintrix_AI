"""Shared annotation-format conversion helpers (XML/JSON/CSV -> YOLO txt),
used by prepare_detrac.py / prepare_aicity.py / prepare_emergency.py so
conversion logic lives in one place instead of being copy-pasted per script.
"""


def to_yolo_format(annotations, class_name_to_id: dict, image_width: int, image_height: int):
    """TODO: convert a list of {class_name, x1, y1, x2, y2} boxes into YOLO's
    normalized `class_id x_center y_center width height` per-line format.
    """
    raise NotImplementedError
