"""ByteTrack integration.

Takes per-frame Detection lists from detection/detector.py and returns
detections annotated with a persistent track_id. A vehicle visible across
many frames must keep one ID — this is what counting/counter.py depends on
to avoid double-counting.

Implementation choice
---------------------
This module is a self-contained implementation of the BYTE association
algorithm from "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box" (Zhang et al., ECCV 2022) rather than a wrapper over
ultralytics' BYTETracker. Reasons:

  * The counting module needs per-track metadata (current/previous
    positions, first/last seen times) that our TrackedVehicle dataclass
    already shapes — a wrapper would force us to translate ByteTrack's
    internal track objects back into this struct on every frame anyway.
  * The core loop stays numpy-only (no torch/ultralytics), so the tracker
    is fast to import, fast to test, and CI-friendly.
  * The algorithm is short and auditable: two-stage Hungarian/IoU
    association over a constant-velocity Kalman predictor plus a small
    lost-track buffer. That's the whole trick.

If the team later wants an ultralytics BYTETracker behind this same API,
the swap is localized to VehicleTracker.update(); the VehicleTracker
signature is the stable seam.

Two-stage (BYTE) association
----------------------------
Detection scores above ``track_thresh`` are "high" and drive the first
Hungarian/IoU matching round. Unmatched tracks are then re-attempted
against detections whose score is below ``track_thresh`` (the "byte"
pool): a vehicle under occlusion often still yields a low-confidence box,
and matching it keeps the track alive instead of breaking the ID. Because
detection/detector.py already filters everything below its own
confidence_threshold (default 0.5 == track_thresh), the low-score pool is
empty in the default pipeline; the branch exists so that running the
detector at a lower confidence_threshold automatically enables the byte
recovery the paper describes.

Occlusion handling follows the stub's intent: a track whose detection is
absent stays recoverable for ``track_buffer`` frames (its Kalman filter
keeps predicting the box), during which it is excluded from output; if it
reappears in time it comes back with the same track_id. After
``track_buffer`` frames it is removed, and its ID is never reused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from detection.detector import Detection

DEFAULT_CONFIG_PATH = "configs/model.yaml"


# ---- GEOMETRY HELPERS ----


def _iou_xyxy(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes; 0.0 if separate or degenerate."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _iou_matrix(pred_boxes, det_boxes) -> np.ndarray:
    """(len(pred_boxes) x len(det_boxes)) IoU matrix."""
    ious = np.zeros((len(pred_boxes), len(det_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, db in enumerate(det_boxes):
            ious[i, j] = _iou_xyxy(pb, db)
    return ious


def _det_xyxy(det: Detection) -> tuple[float, float, float, float]:
    return (det.x1, det.y1, det.x2, det.y2)


def _center(box_xyxy) -> tuple[float, float]:
    return ((box_xyxy[0] + box_xyxy[2]) / 2.0, (box_xyxy[1] + box_xyxy[3]) / 2.0)


def _xywh_from_xyxy(box_xyxy) -> tuple[float, float, float, float]:
    return (
        (box_xyxy[0] + box_xyxy[2]) / 2.0,
        (box_xyxy[1] + box_xyxy[3]) / 2.0,
        box_xyxy[2] - box_xyxy[0],
        box_xyxy[3] - box_xyxy[1],
    )


def _box_from_cxcywh(cx, cy, w, h) -> tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


# ---- KALMAN FILTER ----


class _KalmanBox:
    """Constant-velocity Kalman filter over the box state.

    State vector: [cx, cy, w, h, vcx, vcy, vw, vh] — center, width, height
    and their per-frame velocities. Frame-to-frame step is dt=1. The
    prediction on occluded frames is what lets a track re-associate after
    a gap instead of dying.
    """

    def __init__(self, process_std: float, measurement_std: float):
        self._F = np.eye(8)
        self._F[:4, 4:] = np.eye(4)  # position += velocity
        self._H = np.hstack([np.eye(4), np.zeros((4, 4))])
        self._Q = (process_std ** 2) * np.eye(8)
        self._R = (measurement_std ** 2) * np.eye(4)
        self._started = False

    def init(self, cx: float, cy: float, w: float, h: float) -> None:
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._P = np.eye(8) * 100.0  # confident first guess, loose covariance
        self._started = True

    def predict(self) -> None:
        if not self._started:
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, cx: float, cy: float, w: float, h: float) -> None:
        if not self._started:
            return
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self._H @ self._x                       # innovation
        s = self._H @ self._P @ self._H.T + self._R
        k = self._P @ self._H.T @ np.linalg.inv(s)      # Kalman gain
        self._x = self._x + k @ y
        self._P = (np.eye(8) - k @ self._H) @ self._P

    def box(self) -> tuple[float, float, float, float]:
        """Predicted/updated box as (x1, y1, x2, y2)."""
        cx, cy, w, h = self._x[:4]
        return _box_from_cxcywh(cx, cy, w, h)

    def center(self) -> tuple[float, float]:
        return (float(self._x[0]), float(self._x[1]))


# ---- HUNGARIAN ASSIGNMENT ----


def _hungarian_assignment(cost_matrix):
    """Minimum-cost one-to-one assignment for a rectangular cost matrix.

    Returns (rows, cols) index lists of assigned pairs. Empty rows/cols
    (the identity of the "unmatched" pairing) are handled by padding the
    matrix to square with zero-cost cells: the classic O(n^3) Hungarian
    with potentials then yields a perfect matching whose min(rows, cols)
    real pairs are the answer. The caller applies the IoU gate afterwards.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)

    # 1-indexed square working array (standard emaxx-style Hungarian).
    a = np.zeros((n + 1, n + 1), dtype=np.float64)
    a[1 : n_rows + 1, 1 : n_cols + 1] = cost

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 2, dtype=np.int64)
    way = np.zeros(n + 2, dtype=np.int64)
    INF = 1e18

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = a[i0, j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:  # augment along the stored path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows, cols = [], []
    for j in range(1, n + 1):
        if p[j] != 0:
            row = int(p[j]) - 1
            col = int(j) - 1
            if row < n_rows and col < n_cols:
                rows.append(row)
                cols.append(col)
    return rows, cols


# ---- TRACK STRUCT + PUBLIC DATACLASS ----


@dataclass
class TrackedVehicle:
    track_id: int
    cls: str
    confidence: float
    current_position: tuple[float, float]
    previous_position: tuple[float, float] | None
    first_seen_time: float
    last_seen_time: float
    road: str | None = None          # filled in by counting/roi.py
    movement: str | None = None      # filled in by counting/movement.py
    queue_entry_time: float | None = None
    waiting_time: float = 0.0


class _Track:
    """Internal per-vehicle state: Kalman filter + visibility bookkeeping."""

    def __init__(self, track_id: int, det: Detection, timestamp: float, kf: _KalmanBox):
        self.id = track_id
        self.cls = det.cls
        self.confidence = det.confidence
        cx, cy, w, h = _xywh_from_xyxy(_det_xyxy(det))
        self.kf = kf
        self.kf.init(cx, cy, w, h)
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.frames_missed = 0
        self.active = True
        self.previous_position: tuple[float, float] | None = None
        self.last_active_center = (cx, cy)

    def predict_frame(self) -> None:
        self.kf.predict()

    def match(
        self, det: Detection, timestamp: float
    ) -> tuple[tuple[float, float], tuple[float, float] | None, str, float]:
        """Apply a matched detection; return (current, previous, class, conf)."""
        cx, cy, w, h = _xywh_from_xyxy(_det_xyxy(det))
        self.kf.update(cx, cy, w, h)
        self.previous_position = self.last_active_center
        self.last_active_center = self.kf.center()
        self.cls = det.cls
        self.confidence = det.confidence
        self.frames_missed = 0
        self.active = True
        self.last_seen = timestamp
        return (self.last_active_center, self.previous_position, self.cls, self.confidence)


def _predict_logic(tracks: list[_Track]) -> None:
    """Advance every track's Kalman guess one frame."""
    for track in tracks:
        track.predict_frame()


# ---- VEHICLE TRACKER ----


class VehicleTracker:
    """ByteTrack-style multi-object tracker over Detection lists.

    Params (all config-driven via VehicleTracker.from_config, defaults shown):
      track_buffer:     frames a lost track stays recoverable (config 30)
      match_thresh:     BYTE cost gate for the first (high-score) stage; a
                        pair is kept when 1 - IoU <= match_thresh, so the
                        default 0.8 accepts anything with IoU >= 0.2
      track_thresh:     detection-score split; >= is "high" (config 0.5)
      low_match_thresh: BYTE cost gate for the byte (low-score) stage
    """

    def __init__(
        self,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        *,
        track_thresh: float = 0.5,
        low_match_thresh: float = 0.5,
        kalman_process_std: float = 0.05,
        kalman_measurement_std: float = 0.5,
    ):
        if track_buffer < 1:
            raise ValueError(f"track_buffer must be >= 1, got {track_buffer}")
        for name, value in (
            ("match_thresh", match_thresh),
            ("track_thresh", track_thresh),
            ("low_match_thresh", low_match_thresh),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")

        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.track_thresh = float(track_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.kalman_process_std = float(kalman_process_std)
        self.kalman_measurement_std = float(kalman_measurement_std)

        self._tracks: list[_Track] = []
        self._next_id = 1

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_CONFIG_PATH) -> "VehicleTracker":
        """Build a tracker from the `tracker` block of configs/model.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        tracker_cfg = cfg["tracker"]
        kalman_cfg = tracker_cfg.get("kalman", {})
        return cls(
            track_buffer=tracker_cfg["track_buffer"],
            match_thresh=tracker_cfg["match_thresh"],
            track_thresh=tracker_cfg.get("track_thresh", 0.5),
            low_match_thresh=tracker_cfg.get("low_match_thresh", 0.5),
            kalman_process_std=float(kalman_cfg.get("process_std", 0.05)),
            kalman_measurement_std=float(kalman_cfg.get("measurement_std", 0.5)),
        )

    def reset(self) -> None:
        """Clear all tracks; used when switching to a new video/sequence."""
        self._tracks = []
        self._next_id = 1

    # ---- UPDATE CONTINUES BELOW ----

    def update(self, detections: list[Detection], timestamp: float) -> list[TrackedVehicle]:
        """Feed one frame's detections in, get tracked vehicles with IDs out.

        Returns the currently-visible (matched) tracks as TrackedVehicle,
        sorted by track_id. Lost-but-recoverable tracks are withheld until
        re-associated; after track_buffer frames they are dropped for good.
        """
        high: list[Detection] = []
        low: list[Detection] = []
        for det in detections:
            if det.confidence >= self.track_thresh:
                high.append(det)
            else:
                low.append(det)

        # Advance every track's prediction; build the recoverable-tracks pool.
        pool_indices: list[int] = []
        for i, track in enumerate(self._tracks):
            track.predict_frame()
            if track.frames_missed <= self.track_buffer:
                pool_indices.append(i)

        # Stage 1: pool tracks vs high-score detections (the "head" dets).
        stage1 = self._associate(pool_indices, high, self.match_thresh)
        still_pool = [i for i in pool_indices if i not in {pos for pos, _ in stage1}]
        for track_pos, det_pos in stage1:
            self._tracks[track_pos].match(high[det_pos], timestamp)

        # Stage 2 (byte recovery): leftover tracks vs low-score detections.
        stage2 = self._associate(still_pool, low, self.low_match_thresh)
        for track_pos, det_pos in stage2:
            self._tracks[track_pos].match(low[det_pos], timestamp)

        # Anything still unmatched dies one frame at a time; prune past buffer.
        unmatched = set(still_pool) - {pos for pos, _ in stage2}
        for track_pos in unmatched:
            self._tracks[track_pos].active = False
            self._tracks[track_pos].frames_missed += 1
        self._tracks = [
            t
            for t in self._tracks
            if (t.frames_missed <= self.track_buffer) or t.active
        ]

        # High-score detections that matched nothing become new tracks.
        used_high = {det_pos for _, det_pos in stage1}
        for det_pos in range(len(high)):
            if det_pos not in used_high:
                self._spawn(high[det_pos], timestamp)

        results = [self._to_tracked_vehicle(t, timestamp) for t in self._tracks if t.active]
        results.sort(key=lambda tv: tv.track_id)
        return results

    # ---- MATCHING + OUTPUT HELPERS BELOW ----

    def _spawn(self, det: Detection, timestamp: float) -> None:
        kf = _KalmanBox(self.kalman_process_std, self.kalman_measurement_std)
        self._tracks.append(_Track(self._next_id, det, timestamp, kf))
        self._next_id += 1

    def _associate(
        self,
        track_positions: list[int],
        detections: list[Detection],
        threshold: float,
    ) -> list[tuple[int, int]]:
        """Hungarian/IoU association; returns (track_list_pos, det_index) pairs.

        ``threshold`` is the BYTE cost threshold: pairs are kept when
        ``1 - IoU <= threshold``. With the config default match_thresh=0.8
        this accepts IoU >= 0.2, which is ByteTrack's exact semantics and
        the reason a fast-moving vehicle whose box only overlaps ~0.7
        between frames keeps its track instead of fragmenting.
        """
        if not track_positions or not detections:
            return []
        pred_boxes = [self._tracks[i].kf.box() for i in track_positions]
        det_boxes = [_det_xyxy(d) for d in detections]
        ious = _iou_matrix(pred_boxes, det_boxes)
        cost = 1.0 - ious
        rows, cols = _hungarian_assignment(cost)
        pairs = []
        for r, c in zip(rows, cols):
            if cost[r][c] <= threshold:
                pairs.append((track_positions[r], c))
        return pairs

    def _to_tracked_vehicle(self, track: _Track, timestamp: float) -> TrackedVehicle:
        return TrackedVehicle(
            track_id=track.id,
            cls=track.cls,
            confidence=track.confidence,
            current_position=track.last_active_center,
            previous_position=track.previous_position,
            first_seen_time=float(track.first_seen),
            last_seen_time=float(track.last_seen),
        )
