"""Frame-source -> VehicleDetector inference loop.

Handles pulling frames from a video file / stream and running the detector
on each one. Kept separate from detector.py so batch evaluation and the
live backend can both drive it.
"""


def run_on_video(video_path: str, detector, callback):
    """Iterate frames of video_path, call detector.detect(frame), and
    invoke callback(frame_index, detections) for each frame.

    TODO: open video with cv2.VideoCapture, loop, handle end-of-stream,
    handle dropped/corrupt frames without crashing the pipeline.
    """
    raise NotImplementedError
