"""Shared annotation-format conversion helpers (XML/JSON/CSV -> YOLO txt),
used by prepare_detrac.py / prepare_aicity.py / prepare_emergency.py so
conversion logic lives in one place instead of being copy-pasted per script.

YOLO txt convention, one line per object:
    `<class_id> <x_center> <y_center> <width> <height>`
with all four box values normalized to [0, 1] by the image dimensions and
the origin at the top-left corner.
"""


def to_yolo_format(annotations, class_name_to_id: dict, image_width: int, image_height: int) -> str:
    """Convert a list of {class_name, x1, y1, x2, y2} boxes into YOLO's
    normalized `class_id x_center y_center width height` text format.

    Boxes are absolute pixel coordinates with (x1, y1) the top-left corner
    and (x2, y2) the bottom-right. Boxes are first clipped to the image
    bounds (UA-DETRAC occasionally annotates a box that extends a pixel or
    two past the frame), then normalized by the image dimensions. A box
    whose class is absent from `class_name_to_id`, or that clips to zero
    width/height, contributes no line.

    Returns the file content for one YOLO label file ("" for no boxes).
    """
    out_lines = []
    for annotation in annotations:
        class_id = class_name_to_id.get(annotation["class_name"])
        if class_id is None:
            continue
        x1 = min(max(annotation["x1"], 0.0), image_width)
        y1 = min(max(annotation["y1"], 0.0), image_height)
        x2 = min(max(annotation["x2"], 0.0), image_width)
        y2 = min(max(annotation["y2"], 0.0), image_height)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        center_x = (x1 + x2) / 2.0 / image_width
        center_y = (y1 + y2) / 2.0 / image_height
        yolo_width = box_width / image_width
        yolo_height = box_height / image_height
        out_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} {yolo_width:.6f} {yolo_height:.6f}"
        )
    return "\n".join(out_lines)
