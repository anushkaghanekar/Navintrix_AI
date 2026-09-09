from types import SimpleNamespace

from detection.detector import Detection


class _FakeDetector:
    confidence_threshold = 0.5
    weights_path = "models/best.pt"

    def detect(self, _frame):
        return [Detection("ambulance", 0.91, 430.0, 100.0, 500.0, 170.0)]


def test_backend_live_frame_runs_detection_tracking_and_serialization():
    from backend.main import BackendRuntime

    runtime = BackendRuntime()
    runtime.detector = _FakeDetector()

    payload = runtime.infer_frame(SimpleNamespace(shape=(960, 960, 3)))

    assert payload["frame_size"] == {"width": 960, "height": 960}
    assert payload["detections"][0]["class"] == "ambulance"
    assert payload["detections"][0]["emergency"] is True
    assert payload["vehicles"][0]["track_id"] == 1
    assert payload["vehicles"][0]["road"] == "north"
    assert payload["inference"]["frames_processed"] == 1


def test_backend_live_state_can_be_reset_without_reloading_detector():
    from backend.main import BackendRuntime

    runtime = BackendRuntime()
    runtime.detector = _FakeDetector()
    runtime.infer_frame(SimpleNamespace(shape=(960, 960, 3)))

    runtime.reset_live_state()

    assert runtime.latest_detections == []
    assert runtime.latest_vehicles == []
    assert runtime.latest_emergency is None
    assert runtime.inference_frame_count == 1
    assert runtime.detector is not None
