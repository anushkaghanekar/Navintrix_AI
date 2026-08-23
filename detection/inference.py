"""Frame-source -> VehicleDetector inference loop.

Handles pulling frames from a video file / stream and running the detector
on each one. Kept separate from detector.py so batch evaluation and the
live backend can both drive it.
"""

from __future__ import annotations


def run_on_video(video_path, detector, callback, max_frames: int | None = None) -> int:
    """Iterate frames of video_path, call detector.detect(frame), and
    invoke callback(frame_index, detections) for each frame.

    ``frame_index`` is the 0-based position of the frame in the source video
    and is preserved even when a frame is skipped, so downstream code can
    always align detections back to the source timeline / ground truth.
    Returns the number of frames actually processed (i.e. number of callback
    invocations). ``max_frames`` (optional) bounds how many source frames
    are read — useful for quick smoke runs during evaluation.

    Robustness: a frame that raises inside the detector is skipped with a
    warning and does not stop the loop, so a single corrupt frame cannot
    kill a long evaluation run. End-of-stream and unopenable sources are
    handled cleanly (IOError for the latter).
    """
    import cv2  # lazy: heavy lib, and detector.py already carries its own

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video source: {video_path}")

    source_index = 0
    processed = 0
    skipped = 0
    try:
        while True:
            if max_frames is not None and source_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break  # clean end-of-stream
            try:
                detections = detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
                skipped += 1
                print(f"[inference] skipping corrupt source frame {source_index}: {exc}")
                source_index += 1
                continue
            callback(source_index, detections)
            processed += 1
            source_index += 1
    finally:
        cap.release()

    if skipped:
        print(f"[inference] processed {processed} frames, skipped {skipped} corrupt frames")
    return processed
