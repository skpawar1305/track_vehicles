import json, threading, time, os, sys
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty, Full
from urllib.parse import urlsplit, unquote


def _enable_relay_capture_opts():
    """Allow FFmpeg to open the RTP source of the local relay SDP.

    OpenCV/FFmpeg snapshot OPENCV_FFMPEG_CAPTURE_OPTIONS at process start, so
    changing it in-process has no effect: when the SHA-256 relay is enabled we
    re-exec once with the variable set. Runs before OpenCV is imported.
    """
    if os.environ.get("TRACKER_RELAY_REEXEC") == "1":
        print("[relay] FFmpeg options:",
              os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", ""), flush=True)
        return
    try:
        with open("config.json") as f:
            relay = bool(json.load(f).get("relay", False))
    except Exception:
        relay = False
    if not (relay or os.environ.get("TRACKER_RTSP_RELAY") == "1"):
        return
    opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    if "protocol_whitelist" in opts:
        return
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        opts + "|" if opts else "") + (
        "protocol_whitelist;file,udp,rtp,rtsp,tcp|rw_timeout;15000000")
    os.environ["TRACKER_RELAY_REEXEC"] = "1"
    print("[relay] re-exec with FFmpeg options:", opts, flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)


_enable_relay_capture_opts()

import cv2
import numpy as np
import ncnn
from store import Store
from rtsp_relay import RtspRelay

# Rate-limited logging for background workers: never let a recurring failure
# either spam the journal every second or disappear silently.
_throttle_last = {}


def _log_throttled(key, msg, every=60.0):
    now = time.time()
    if now - _throttle_last.get(key, 0.0) >= every:
        _throttle_last[key] = now
        print(msg, flush=True)


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PARAM = str(MODEL_DIR / "nanodet_plus_m_1.5x_416.ncnn.param")
# Speed/accuracy ladder: pick the highest resolution that keeps enough FPS on the Pi.
# TRACKER_MODEL_IMGSZ=640|512|416 selects models/yolo11n_ncnn_<size>/.
def _parse_imgsz(s):
    """'640' -> (640, 640); '384x640' -> (384, 640)  (returns h, w)."""
    s = str(s).lower()
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    n = int(s)
    return n, n


YOLO_MODEL = os.environ.get("TRACKER_MODEL", "yolo26n")   # e.g. yolo26n | yolo11n
YOLO_IMGSZ = os.environ.get("TRACKER_MODEL_IMGSZ", "320x320")
YOLO_H, YOLO_W = _parse_imgsz(YOLO_IMGSZ)
YOLO_INPUT = YOLO_W  # unqualified legacy alias (width)
YOLO_DIR = MODEL_DIR / f"{YOLO_MODEL}_ncnn_{YOLO_IMGSZ}"
YOLO_INT8 = os.environ.get("TRACKER_INT8", "0") == "1"
YOLO_PARAM = str(YOLO_DIR / ("model_int8.ncnn.param" if YOLO_INT8 else "model.ncnn.param"))
NUM_THREADS = int(os.environ.get("TRACKER_THREADS", "4"))
# A healthy frame yields a handful of vehicles; a corrupt/desynced HEVC frame can
# make the detector hallucinate hundreds. Treat such frames as invalid.
MAX_DETECTIONS_PER_FRAME = int(os.environ.get("TRACKER_MAX_DETS", "50"))
DEFAULT_ENABLED_CLASSES = [2, 5, 7]  # cars, buses, and trucks; exclude bicycles/motorcycles
EXCLUDED_TWO_WHEELER_CLASSES = {1, 3}  # COCO bicycle and motorcycle
# One vehicle is often localised as two overlapping boxes whose winning classes
# differ (car vs truck). At IoU above this we treat them as the same object: in
# the detector (class-agnostic NMS, the Ultralytics `agnostic_nms` equivalent)
# and at the crossing (one count). Genuinely adjacent vehicles overlap far less.
DUP_IOU = float(os.environ.get("TRACKER_DUP_IOU", "0.6"))
# A duplicate crossing from a different track within this window (and overlapping
# the previous box) is suppressed. 0 disables the safety net.
CROSS_DEDUP_S = float(os.environ.get("TRACKER_CROSS_DEDUP_S", "1.0"))


def _counting_classes(classes):
    return [int(c) for c in classes if int(c) not in EXCLUDED_TWO_WHEELER_CLASSES]

# ── Config ────────────────────────────────────────────────────────────
CONFIG_PATH = "config.json"

class Config:
    def __init__(self):
        self.lock = threading.RLock()
        self.stream_url = ""
        self.line = None          # legacy pixel coords
        self.roi = None           # legacy pixel coords
        self.norm_line = None     # normalized [[x,y],[x,y]] (0..1)
        self.norm_roi = None      # normalized [[x,y],...] (0..1)
        self.conf_thresh = 0.35
        self.flip_sides = False
        self.capture_dir = "captures"
        self.max_captures = 1000
        self.enabled_classes = DEFAULT_ENABLED_CLASSES.copy()
        self.detector = os.environ.get("TRACKER_DETECTOR", "hybrid")  # "hybrid" | "motion" | "yolo"
        self.motion_scale = float(os.environ.get("TRACKER_MOTION_SCALE", "0.5"))
        self.motion_min_area = int(os.environ.get("TRACKER_MOTION_MIN_AREA", "12"))
        self.motion_hold = float(os.environ.get("TRACKER_MOTION_HOLD", "60"))
        self.motion_heartbeat = float(os.environ.get("TRACKER_MOTION_HEARTBEAT", "3"))
        self.min_track_age = int(os.environ.get("TRACKER_MIN_TRACK_AGE", "1"))
        self.min_travel_frac = float(os.environ.get("TRACKER_MIN_TRAVEL_FRAC", "0.02"))
        # Hysteresis past the line (fraction of frame height) to reject jitter.
        self.cross_margin_frac = float(os.environ.get("TRACKER_CROSS_MARGIN_FRAC", "0.015"))
        self.running = True
        self._reconnect = False
        self.captures = []
        self.token = os.environ.get("TRACKER_TOKEN", "")
        _live = os.environ.get("TRACKER_LIVE_URL", "")
        self.config_url = os.environ.get(
            "TRACKER_CONFIG_URL", _live.replace("/api/live", "/api/config") if _live else "")
        self.config_updated_at = None
        self.live_wanted = False   # set true by server while a viewer is connected
        # Cameras whose RTSP Digest is SHA-256 (OpenCV's FFmpeg only does MD5)
        # are reached through the local RtspRelay UDP->SDP shim.
        self.relay = os.environ.get("TRACKER_RTSP_RELAY", "0") == "1"
        # Optional second camera: its raw substream is shown on the dashboard with
        # no detection. It stays fully idle (no RTSP session, no decode, no upload)
        # until someone is watching /live.
        self.cam2_url = os.environ.get("TRACKER_CAM2_URL", "")
        self.cam2_sdp = os.environ.get("TRACKER_CAM2_SDP", "/tmp/tracker_cam2.sdp")
        self.cam2_port = int(os.environ.get("TRACKER_CAM2_PORT", "5006"))
        self.live2_url = os.environ.get("TRACKER_LIVE2_URL", "")
        if not self.live2_url and _live:
            self.live2_url = _live.replace("/api/live", "/api/live2")
        self.live2_wanted = False
        # Full-resolution evidence: at each crossing fetch a still from the
        # camera's main-profile ONVIF snapshot (the camera encodes the JPEG, so
        # no Pi decode). Falls back to the processed substream frame if the fetch
        # fails. URL/user/password default to the stream URL's; override with
        # TRACKER_CAPTURE_URL / _USER / _PASSWORD.
        self.capture_url = os.environ.get("TRACKER_CAPTURE_URL", "")
        self.capture_user = os.environ.get("TRACKER_CAPTURE_USER", "")
        self.capture_password = os.environ.get("TRACKER_CAPTURE_PASSWORD", "")
        self._load()

    def _load(self):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            with self.lock:
                self.stream_url = data.get("stream_url", "")
                self.line = data.get("line")
                self.roi = data.get("roi")
                self.norm_line = data.get("norm_line")
                self.norm_roi = data.get("norm_roi")
                self.conf_thresh = data.get("conf_thresh", 0.5)
                self.flip_sides = data.get("flip_sides", False)
                self.capture_dir = os.environ.get(
                    "TRACKER_CAPTURE_DIR", data.get("capture_dir", "captures"))
                self.max_captures = data.get("max_captures", 1000)
                self.enabled_classes = _counting_classes(
                    data.get("enabled_classes", DEFAULT_ENABLED_CLASSES))
                self.relay = bool(data.get("relay", self.relay))
                self.cam2_url = data.get("cam2_url", self.cam2_url)
                self.capture_url = data.get("capture_url", self.capture_url)
            self._derive_capture()
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _derive_capture(self):
        """Default the full-res capture URL/creds from the RTSP stream URL."""
        if not self.stream_url.startswith("rtsp://"):
            return
        u = urlsplit(self.stream_url)
        if not u.hostname:
            return
        if not self.capture_user:
            self.capture_user = unquote(u.username or "")
        if not self.capture_password:
            self.capture_password = unquote(u.password or "")
        if not self.capture_url:
            # Prama/PT-NC cameras serve a hardware-encoded 1080p still here.
            self.capture_url = f"https://{u.hostname}/onvif-http/snapshot?Profile_1"

    def save(self):
        with self.lock:
            data = {"stream_url": self.stream_url, "line": self.line, "roi": self.roi,
                    "norm_line": self.norm_line, "norm_roi": self.norm_roi,
                    "conf_thresh": self.conf_thresh, "flip_sides": self.flip_sides,
                    "capture_dir": self.capture_dir, "max_captures": self.max_captures,
                    "enabled_classes": self.enabled_classes, "relay": self.relay,
                    "cam2_url": self.cam2_url, "capture_url": self.capture_url}
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def set_line(self, line):
        with self.lock: self.line = line
        self.save()
    def set_roi(self, roi):
        with self.lock: self.roi = roi
        self.save()
    def set_stream_url(self, url):
        with self.lock: self.stream_url = url; self._reconnect = True
        self.save()
    def check_reconnect(self):
        with self.lock:
            if self._reconnect: self._reconnect = False; return True
            return False
    def set_flip_sides(self, flip):
        with self.lock: self.flip_sides = flip
        self.save()
    def set_enabled_classes(self, classes):
        with self.lock: self.enabled_classes = _counting_classes(classes)
        self.save()

    def apply_remote(self, data):
        """Apply a config pulled from the server (normalized coords)."""
        with self.lock:
            self.norm_line = data.get("line")
            self.norm_roi = data.get("roi")
            self.flip_sides = bool(data.get("flip_sides", False))
            if data.get("enabled_classes"):
                self.enabled_classes = _counting_classes(data["enabled_classes"])
            self.config_updated_at = data.get("updated_at")
        self.save()

    def pixel_line(self, w, h):
        with self.lock:
            if self.norm_line and len(self.norm_line) >= 2:
                a, b = self.norm_line[0], self.norm_line[1]
                return (int(a[0] * w), int(a[1] * h), int(b[0] * w), int(b[1] * h))
            return self.line

    def pixel_roi(self, w, h):
        with self.lock:
            if self.norm_roi and len(self.norm_roi) >= 3:
                return [[int(p[0] * w), int(p[1] * h)] for p in self.norm_roi]
            return self.roi
    def reset_counts(self):
        with self.lock: self.captures.clear()
        self.save()
    def add_capture(self, entry):       # FIX 3 (carried over): thread-safe insert
        with self.lock:
            self.captures.insert(0, entry)
            if len(self.captures) > self.max_captures:
                self.captures.pop()

cfg = Config()

# ── Line crossing ─────────────────────────────────────────────────────
CROSSING_NONE, CROSSING_IN, CROSSING_OUT = 0, 1, 2

def line_geometry(line, point):
    """Signed perpendicular distance (px) of `point` to `line`, and its
    projection parameter t along the segment (0 at (x1,y1), 1 at (x2,y2)).

    Returns (None, None) for a degenerate line/point.
    """
    if line is None or point is None:
        return None, None
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 <= 0:
        return None, None
    px, py = point
    dist = (dx * (py - y1) - dy * (px - x1)) / (l2 ** 0.5)
    t = ((px - x1) * dx + (py - y1) * dy) / l2
    return dist, t


def box_iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def detect_crossing(line, old_centroid, new_centroid, flip=False):
    if line is None or old_centroid is None or new_centroid is None:
        return CROSSING_NONE
    x1, y1, x2, y2 = line
    cp_old = (x2 - x1) * (old_centroid[1] - y1) - (y2 - y1) * (old_centroid[0] - x1)
    cp_new = (x2 - x1) * (new_centroid[1] - y1) - (y2 - y1) * (new_centroid[0] - x1)
    old_side = 1 if cp_old >= 0 else -1
    new_side = 1 if cp_new >= 0 else -1
    if old_side != new_side:
        if flip: return CROSSING_OUT if new_side == 1 else CROSSING_IN
        return CROSSING_IN if new_side == 1 else CROSSING_OUT
    return CROSSING_NONE

class CrossingGate:
    """Suppresses spurious crossings (shadow/lighting/compression flicker).

    A track must have existed for `min_age` frames and travelled at least
    `min_travel_frac` of the frame height (net displacement from where it was
    first seen) before a crossing is accepted, plus a per-track cooldown.
    """

    def __init__(self, min_age=4, min_travel_frac=0.04, cooldown=2.0):
        self.min_age = min_age
        self.min_travel_frac = min_travel_frac
        self.cooldown = cooldown
        self.first = {}
        self.last_cross = {}
        self.side = {}        # tid -> last committed side (+1/-1) outside the band

    def update_first(self, tid, cen):
        if tid not in self.first:
            self.first[tid] = (float(cen[0]), float(cen[1]))

    def drop(self, dead_ids):
        for tid in dead_ids:
            self.first.pop(tid, None)
            self.last_cross.pop(tid, None)
            self.side.pop(tid, None)

    def crossing(self, line, obj, flip=False, margin=0.0, seg_tol=0.02):
        """Stateful line-crossing test with hysteresis.

        Returns CROSSING_NONE/IN/OUT. A crossing is registered only when the
        object's centroid moves from one committed side of the line to the
        other, landing at least `margin` px past it, with the crossing point
        inside the drawn segment. Committing to a side means a low-confidence
        box whose centroid jitters across the line by a pixel or two never
        registers a crossing.
        """
        dist, t = line_geometry(line, obj.centroid)
        if dist is None:
            return CROSSING_NONE
        if abs(dist) < margin:
            return CROSSING_NONE
        new_side = 1 if dist >= 0 else -1
        prev = self.side.get(obj.track_id)
        self.side[obj.track_id] = new_side
        if prev is None or prev == new_side:
            return CROSSING_NONE
        if t < -seg_tol or t > 1 + seg_tol:
            return CROSSING_NONE
        if flip:
            return CROSSING_OUT if new_side == 1 else CROSSING_IN
        return CROSSING_IN if new_side == 1 else CROSSING_OUT

    def allow(self, obj, now, frame_h):
        if obj.age < self.min_age:
            return False
        fx, fy = self.first.get(obj.track_id, obj.centroid)
        travel = ((obj.centroid[0] - fx) ** 2 + (obj.centroid[1] - fy) ** 2) ** 0.5
        if travel < self.min_travel_frac * frame_h:
            return False
        if now - self.last_cross.get(obj.track_id, 0.0) < self.cooldown:
            return False
        self.last_cross[obj.track_id] = now
        return True

# ── Capture Manager ───────────────────────────────────────────────────
class CaptureManager:
    def __init__(self, capture_dir="captures", max_captures=1000):
        self.capture_dir = capture_dir
        self.max_captures = max_captures
        os.makedirs(capture_dir, exist_ok=True)
        os.makedirs(os.path.join(capture_dir, "thumb"), exist_ok=True)

    THUMB_WIDTH = 640
    THUMB_QUALITY = 80

    @staticmethod
    def _ts(crossed_at=None):
        if crossed_at:
            try:
                return datetime.fromisoformat(crossed_at).strftime(
                    "%Y%m%d_%H%M%S_%f")[:-3]
            except ValueError:
                pass
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _thumb(self, frame, filename):
        # Full-frame 16:9 thumbnail (context beats a tight, blurry crop).
        h, w = frame.shape[:2]
        tw = min(self.THUMB_WIDTH, w)
        th = max(1, int(h * tw / w))
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(self.capture_dir, "thumb", filename), thumb,
                    [cv2.IMWRITE_JPEG_QUALITY, self.THUMB_QUALITY])

    def store_crossing(self, frame, jpeg, track_id, direction, crossed_at=None):
        """Persist one crossing.

        Saves two images when the main-stream still is available: the full-size
        `<id>.jpg` (the camera's main profile, best resolution) and the
        `<id>_sub.jpg` substream frame captured at the crossing (the accurate
        moment the vehicle was on the line). The thumbnail is ALWAYS built from
        the substream frame, so the timeline shows the vehicle at the line
        rather than the delayed main-stream moment. When `jpeg` is None the
        substream frame is the only source and is saved as `<id>.jpg` (no
        separate `_sub` copy, to avoid storing the same pixels twice).
        """
        ts = self._ts(crossed_at)
        direction_label = "in" if direction == 1 else "out"
        filename = f"{ts}_id{track_id}_{direction_label}.jpg"
        sub = None
        if jpeg is not None:
            with open(os.path.join(self.capture_dir, filename), "wb") as f:
                f.write(jpeg)
            sub = f"{ts}_id{track_id}_{direction_label}_sub.jpg"
            cv2.imwrite(os.path.join(self.capture_dir, sub), frame)
        else:
            cv2.imwrite(os.path.join(self.capture_dir, filename), frame)
        self._thumb(frame, filename)
        return {"filename": filename, "thumb": f"thumb/{filename}", "sub": sub,
                "timestamp": ts, "track_id": track_id, "direction": direction_label}

# ── NanoDet ncnn Detector ─────────────────────────────────────────────
_VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', -1: 'motion'}
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_STD  = np.array([57.375,  57.12,  58.395], dtype=np.float32)
_STRIDES = [8, 16, 32, 64]
_INPUT_SIZE = 416

class NanoDetNcnn:
    def __init__(self, model_param=MODEL_PARAM, model_bin=None, conf_thresh=0.5,
                 iou_thresh=0.45, num_threads=NUM_THREADS):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        if model_bin is None:
            model_bin = model_param[:-6] + ".bin"
        self.model_param = model_param
        self.model_bin   = model_bin
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.load_param(model_param)
        self.net.load_model(model_bin)
        self.input_name  = "in0"
        self.output_name = "out0"
        priors = []
        for stride in _STRIDES:
            h = int(np.ceil(_INPUT_SIZE / stride))
            w = int(np.ceil(_INPUT_SIZE / stride))
            for i in range(h):
                for j in range(w):
                    priors.append((j * stride, i * stride, stride))
        self.priors = np.array(priors, dtype=np.float32)

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(_INPUT_SIZE / h, _INPUT_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (_INPUT_SIZE - new_h) // 2
        pad_w = (_INPUT_SIZE - new_w) // 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        blob = canvas.astype(np.float32)
        blob = (blob - _MEAN) / _STD
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, pad_w, pad_h

    @staticmethod
    def _softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _nms(self, boxes, scores):
        if len(boxes) == 0: return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= self.iou_thresh)[0] + 1]
        return keep

    def _run_on_roi(self, roi, enabled_classes, offset_x, offset_y):
        h, w = roi.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(roi)
        ex = self.net.create_extractor()
        in_mat = ncnn.Mat(np.ascontiguousarray(blob[0])).clone()
        ex.input(self.input_name, in_mat)
        _, out_mat = ex.extract(self.output_name)
        out = np.array(out_mat)
        if out.ndim != 2 or out.shape[1] < 112:
            return []

        # Fully vectorised NanoDet decode (no per-prior Python loop).
        cls_scores = out[:, :80]
        reg_dists  = out[:, 80:112].reshape(-1, 4, 8)

        cls_ids = cls_scores.argmax(axis=1)
        scores  = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        mask = scores >= self.conf_thresh
        mask &= ~np.isin(cls_ids, tuple(EXCLUDED_TWO_WHEELER_CLASSES))
        if enabled_classes:
            mask &= np.isin(cls_ids, enabled_classes)
        if not mask.any():
            return []
        idx = np.nonzero(mask)[0]

        d = reg_dists[idx]
        e = np.exp(d - d.max(axis=2, keepdims=True))
        soft = e / e.sum(axis=2, keepdims=True)
        bins = np.arange(8, dtype=np.float32)
        dist = (soft * bins).sum(axis=2) * self.priors[idx, 2:3]   # (M, 4)

        cx = self.priors[idx, 0]; cy = self.priors[idx, 1]
        x1 = np.clip((cx - dist[:, 0] - pad_w) / scale, 0, w - 1)
        y1 = np.clip((cy - dist[:, 1] - pad_h) / scale, 0, h - 1)
        x2 = np.clip((cx + dist[:, 2] - pad_w) / scale, 0, w - 1)
        y2 = np.clip((cy + dist[:, 3] - pad_h) / scale, 0, h - 1)
        valid = (x2 > x1) & (y2 > y1)
        if not valid.any():
            return []

        x1 = x1[valid].astype(np.int32); y1 = y1[valid].astype(np.int32)
        x2 = x2[valid].astype(np.int32); y2 = y2[valid].astype(np.int32)
        scores = scores[idx][valid]; cls_ids = cls_ids[idx][valid]

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(boxes, scores)
        out_list = []
        for k in keep:
            bx1, by1, bx2, by2 = int(boxes[k, 0]), int(boxes[k, 1]), int(boxes[k, 2]), int(boxes[k, 3])
            cid = int(cls_ids[k])
            out_list.append({
                'bbox': (bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y),
                'centroid': ((bx1 + bx2) // 2 + offset_x, (by1 + by2) // 2 + offset_y),
                'confidence': float(scores[k]), 'class_id': cid,
                'label': _VEHICLE_NAMES.get(cid, f'cls_{cid}'),
            })
        return out_list

    def detect(self, frame, enabled_classes=None):
        return self._run_on_roi(frame, enabled_classes, 0, 0)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        if roi_points is None or len(roi_points) < 3:
            return self.detect(frame, enabled_classes)
        h, w = frame.shape[:2]
        pts = np.array(roi_points, dtype=np.int32)
        # int(): numpy min()/max() leak np.int32 offsets into every bbox, which
        # later blows up json.dumps() when the crossing is persisted.
        cx1, cy1 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        cx2, cy2 = int(min(w, pts[:, 0].max())), int(min(h, pts[:, 1].max()))
        if cx2 <= cx1 or cy2 <= cy1: return []
        return self._run_on_roi(frame[cy1:cy2, cx1:cx2], enabled_classes, cx1, cy1)


class YoloNcnn:
    """YOLO11 detector via ncnn. Output out0 is (4+nc, anchors); letterboxed 640."""

    def __init__(self, model_param=YOLO_PARAM, model_bin=None, conf_thresh=0.30,
                 iou_thresh=0.45, num_threads=NUM_THREADS, cross_nms_iou=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.cross_nms_iou = DUP_IOU if cross_nms_iou is None else cross_nms_iou
        if model_bin is None:
            model_bin = model_param[:-6] + ".bin"
        self.model_param = model_param
        self.model_bin   = model_bin
        self.h, self.w   = YOLO_H, YOLO_W
        self.size        = YOLO_W
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        # Optional GPU compute (Raspberry Pi 5 has Vulkan 1.2). INT8 models run on
        # CPU regardless; use the fp32 model when TRACKER_VULKAN=1.
        if os.environ.get("TRACKER_VULKAN", "0") == "1":
            self.net.opt.use_vulkan_compute = True
            try:
                self.net.set_vulkan_device(int(os.environ.get("TRACKER_VULKAN_DEVICE", "0")))
            except Exception as e:
                print("[yolo] vulkan device set failed:", e)
        self.net.load_param(model_param)
        self.net.load_model(model_bin)
        self.input_name  = "in0"
        self.output_name = "out0"

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.w / w, self.h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.w - nw) // 2
        pad_h = (self.h - nh) // 2
        canvas = np.full((self.h, self.w, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return blob.transpose(2, 0, 1), scale, pad_w, pad_h

    def _nms(self, boxes, scores):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= self.iou_thresh)[0] + 1]
        return keep

    def _merge_cross_class(self, boxes, scores, keep):
        """Greedily drop kept boxes that overlap (any class) above the threshold.

        Highest score wins, so a car/truck duplicate pair collapses to the one
        the model was more sure about.
        """
        if self.cross_nms_iou <= 0 or len(keep) < 2:
            return keep
        out = []
        for k in sorted(keep, key=lambda k: float(scores[k]), reverse=True):
            if all(box_iou(boxes[k], boxes[j]) <= self.cross_nms_iou for j in out):
                out.append(k)
        return out

    def _run_on_roi(self, roi, enabled_classes, offset_x, offset_y):
        h, w = roi.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(roi)
        ex = self.net.create_extractor()
        # Keep the input Mat alive until *after* extract(): the extractor holds a
        # reference to its memory, and an inline temporary gets GC'd -> use-after-free
        # (a single bad frame then poisons the net into returning garbage boxes).
        in_mat = ncnn.Mat(np.ascontiguousarray(blob)).clone()
        ex.input(self.input_name, in_mat)
        _, out_mat = ex.extract(self.output_name)
        out = np.array(out_mat)              # (4+nc, anchors)
        if out.ndim != 2 or out.shape[0] < 5:
            return []
        preds = out.T                        # (anchors, 4+nc)
        xywh = preds[:, :4]
        cls_scores = preds[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        mask = scores >= self.conf_thresh
        mask &= ~np.isin(cls_ids, tuple(EXCLUDED_TWO_WHEELER_CLASSES))
        if enabled_classes:
            mask &= np.isin(cls_ids, enabled_classes)
        if not mask.any():
            return []
        idx = np.nonzero(mask)[0]

        b = xywh[idx]
        cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = np.clip((cx - bw / 2 - pad_w) / scale, 0, w - 1)
        y1 = np.clip((cy - bh / 2 - pad_h) / scale, 0, h - 1)
        x2 = np.clip((cx + bw / 2 - pad_w) / scale, 0, w - 1)
        y2 = np.clip((cy + bh / 2 - pad_h) / scale, 0, h - 1)
        valid = (x2 > x1) & (y2 > y1)
        if not valid.any():
            return []
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        scores = scores[idx][valid]; cls_ids = cls_ids[idx][valid]
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Class-aware NMS (so overlapping vehicles of different classes survive)
        keep = []
        for c in np.unique(cls_ids):
            m = np.nonzero(cls_ids == c)[0]
            keep.extend(m[self._nms(boxes[m], scores[m])].tolist())

        # Cross-class merge: the model often fires two near-identical boxes for
        # ONE vehicle with different winning classes (car+truck, bus+truck). The
        # per-class NMS above keeps both, the centroid tracker then mints a
        # second track, and both cross the line -> a double count. Suppress
        # across classes at a high IoU so only genuine adjacent vehicles (low
        # overlap) both survive.
        keep = self._merge_cross_class(boxes, scores, keep)

        out_list = []
        for k in keep:
            bx1, by1, bx2, by2 = (int(boxes[k, 0]), int(boxes[k, 1]),
                                  int(boxes[k, 2]), int(boxes[k, 3]))
            cid = int(cls_ids[k])
            out_list.append({
                'bbox': (bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y),
                'centroid': ((bx1 + bx2) // 2 + offset_x, (by1 + by2) // 2 + offset_y),
                'confidence': float(scores[k]), 'class_id': cid,
                'label': _VEHICLE_NAMES.get(cid, f'cls_{cid}'),
            })
        return out_list

    def detect(self, frame, enabled_classes=None):
        return self._run_on_roi(frame, enabled_classes, 0, 0)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        if roi_points is None or len(roi_points) < 3:
            return self.detect(frame, enabled_classes)
        h, w = frame.shape[:2]
        pts = np.array(roi_points, dtype=np.int32)
        # int(): numpy min()/max() leak np.int32 offsets into every bbox, which
        # later blows up json.dumps() when the crossing is persisted.
        cx1, cy1 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        cx2, cy2 = int(min(w, pts[:, 0].max())), int(min(h, pts[:, 1].max()))
        if cx2 <= cx1 or cy2 <= cy1:
            return []
        return self._run_on_roi(frame[cy1:cy2, cx1:cx2], enabled_classes, cx1, cy1)


class MotionDetector:
    """Cheap background-subtraction motion detector (very low CPU).

    Runs on a downscaled frame, so it is affordable at full camera frame rate —
    which keeps tracking stable where a heavy detector would be too slow.
    """

    def __init__(self, scale=0.5, min_area=12, var_threshold=12, history=200, warmup=12,
                 diff_threshold=12):
        self.scale = scale
        self.min_area = min_area
        self.warmup = warmup
        self.diff_threshold = diff_threshold
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._prev = None

    def _mask(self, frame):
        h, w = frame.shape[:2]
        sw, sh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # MOG2 alone absorbs slow movers; OR it with a plain frame difference so
        # any real change (however smooth) still counts as motion.
        mog = cv2.threshold(self.bg.apply(gray), 200, 255, cv2.THRESH_BINARY)[1]
        if self._prev is not None:
            diff = cv2.absdiff(gray, self._prev)
            diff = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
            mog = cv2.bitwise_or(mog, diff)
        self._prev = gray

        # No morphological open (it erases small/distant movers); just grow a bit.
        return cv2.dilate(mog, self.kernel, iterations=2)

    def detect(self, frame, enabled_classes=None):
        fg = self._mask(frame)
        if self.warmup > 0:          # let the background model settle first
            self.warmup -= 1
            return []
        inv = 1.0 / self.scale
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            x1, y1 = int(x * inv), int(y * inv)
            x2, y2 = int((x + w) * inv), int((y + h) * inv)
            out.append({'bbox': (x1, y1, x2, y2), 'centroid': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'confidence': 1.0, 'class_id': -1, 'label': 'motion'})
        return out

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        dets = self.detect(frame)
        if not roi_points or len(roi_points) < 3:
            return dets
        poly = np.array(roi_points, dtype=np.int32)
        return [d for d in dets
                if cv2.pointPolygonTest(poly, (float(d['centroid'][0]), float(d['centroid'][1])), False) >= 0]


class HybridDetector:
    """Low-CPU vehicle detector: cheap motion *arms* the heavy detector.

    The motion pass runs every frame. The moment motion is seen it opens a hold
    window (TRACKER_MOTION_HOLD seconds, default 60); while that window is open
    YOLO runs on every frame — so a vehicle that pauses, or whose motion blob is
    momentarily missed, is still detected and tracked. Once the window lapses
    with no motion, detection idles again. Output is real vehicle detections
    (car/motorcycle/...), never raw motion blobs.
    """

    def __init__(self, conf_thresh=0.30, motion_scale=0.5, motion_min_area=12,
                 margin=0.08, hold_seconds=None, heartbeat=None):
        if hold_seconds is None:
            hold_seconds = float(os.environ.get("TRACKER_MOTION_HOLD", "60"))
        if heartbeat is None:
            heartbeat = float(os.environ.get("TRACKER_MOTION_HEARTBEAT", "3"))
        self.motion = MotionDetector(scale=motion_scale, min_area=motion_min_area)
        self.yolo = YoloNcnn(conf_thresh=conf_thresh)
        self.margin = margin
        self.hold = hold_seconds
        self.heartbeat = heartbeat
        self.active_until = 0.0
        self.last_run = 0.0

    def _union_poly(self, blobs, w, h):
        xs1 = min(b['bbox'][0] for b in blobs); ys1 = min(b['bbox'][1] for b in blobs)
        xs2 = max(b['bbox'][2] for b in blobs); ys2 = max(b['bbox'][3] for b in blobs)
        mx = int((xs2 - xs1) * self.margin) + 8
        my = int((ys2 - ys1) * self.margin) + 8
        x1 = max(0, xs1 - mx); y1 = max(0, ys1 - my)
        x2 = min(w, xs2 + mx); y2 = min(h, ys2 + my)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _gate(self, frame):
        blobs = self.motion.detect(frame)
        now = time.time()
        if blobs:
            self.active_until = now + self.hold
        active = now < self.active_until
        # Heartbeat: even with no motion, run YOLO at least every N seconds so a
        # crossing can never be missed entirely (motion is only an optimisation).
        if not active and (now - self.last_run) >= self.heartbeat:
            active = True
        if active:
            self.last_run = now
        if os.environ.get("TRACKER_DEBUG") == "1" and blobs:
            print(f"[motion] blobs={len(blobs)} armed={active} hold={now < self.active_until}")
        return blobs, active

    def detect(self, frame, enabled_classes=None):
        blobs, active = self._gate(frame)
        if not active:
            return []
        h, w = frame.shape[:2]
        if blobs:
            return self.yolo.detect_roi(frame, self._union_poly(blobs, w, h), enabled_classes)
        return self.yolo.detect(frame, enabled_classes)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        blobs, active = self._gate(frame)
        if not active:
            return []
        if roi_points and len(roi_points) >= 3:
            return self.yolo.detect_roi(frame, roi_points, enabled_classes)
        h, w = frame.shape[:2]
        if blobs:
            return self.yolo.detect_roi(frame, self._union_poly(blobs, w, h), enabled_classes)
        return self.yolo.detect(frame, enabled_classes)


def bbox_touches_line(line, bbox):
    """Check if a bbox overlaps the line's bounding rectangle in both axes."""
    x1, y1, x2, y2 = line
    bx1, by1, bx2, by2 = bbox
    return (bx1 <= max(x1, x2) and bx2 >= min(x1, x2) and
            by1 <= max(y1, y2) and by2 >= min(y1, y2))


def segment_crosses_line(line, old_centroid, new_centroid):
    """True if the motion segment old->new crosses the *drawn* line segment.

    Better than both alternatives for low-FPS counting:
      - the infinite-line side test alone fires far outside the line;
      - a bbox-overlap test misses a fast vehicle that jumps clean over the
        line between two sampled frames (centroid crosses, no bbox overlaps).
    Requires the intersection to lie within the line's endpoints.
    """
    if line is None or old_centroid is None or new_centroid is None:
        return False
    ax, ay, bx, by = line
    ox, oy = old_centroid
    nx, ny = new_centroid

    def side(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = side(ax, ay, bx, by, ox, oy)
    o2 = side(ax, ay, bx, by, nx, ny)
    o3 = side(ox, oy, nx, ny, ax, ay)
    o4 = side(ox, oy, nx, ny, bx, by)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# ── Centroid Tracker (low-fps safe) ───────────────────────────────────
class TrackInfo:
    """Lightweight tracked object wrapper for crossing detection & annotation."""
    __slots__ = ('track_id', 'bbox', 'label', 'class_id', 'confidence', 'centroid',
                 'prev_centroid', 'age', 'last_crossing_frame', 'last_crossing')
    def __init__(self, track_id, bbox, label, class_id, confidence, centroid, prev_centroid, age):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.class_id = int(class_id)
        self.confidence = confidence
        self.centroid = centroid
        self.prev_centroid = prev_centroid
        self.age = age
        self.last_crossing_frame = -60
        self.last_crossing = None


class CentroidTracker:
    """Velocity-aware nearest-centroid tracker for low-fps streams.

    The packaged BYTETracker associates boxes by IoU. On a 10 fps substream a
    vehicle can move farther than its own box between two samples, so IoU is 0:
    ByteTrack drops the track — often returning nothing for every frame after the
    first — and the crossing age/travel gates can never be satisfied. This
    associates by *predicted centroid distance* instead, so fast movers keep
    their id and their prev->cur centroid segment can be tested against the line.
    It is intentionally simple; it suits sparse scenes (a road, not a crowd).
    """

    def __init__(self, max_age=25, assoc_frac=0.2, vel_smooth=0.6,
                 gap_frac=0.5, gap_max_frac=1.0, vel_reset=0.4, pred_cap=0.5,
                 debug=False):
        self.tracks = {}          # id -> state
        self.next_id = 1
        self.max_age = max_age    # coasted frames before a track is dropped
        self.assoc_frac = assoc_frac   # baseline association distance = frac * frame_w
        self.vel_smooth = vel_smooth
        # A frozen/reconnected stream delivers the next frame seconds later, by
        # which time the vehicle has moved far more than one frame's worth. Widen
        # the association gate with the elapsed gap and extrapolate by wall-clock
        # so the track (and its crossing side history) survives the gap.
        self.gap_frac = gap_frac        # extra gate per second of gap = frac * frame_w
        self.gap_max_frac = gap_max_frac  # cap, as a fraction of the frame diagonal
        self.vel_reset = vel_reset      # gap (s) after which velocity is re-measured
        # Cap extrapolation: over a multi-second gap the pre-gap velocity is stale
        # and linear extrapolation can fling the prediction clean off-frame, so far
        # from the real detection that even a widened gate misses it. Extrapolating
        # at most `pred_cap` seconds keeps the anchor near the vehicle; the widened
        # gate then re-acquires it wherever it actually is.
        self.pred_cap = pred_cap
        self.debug = debug

    def _info(self, tid, det, prev, cur, age):
        return TrackInfo(tid, det['bbox'], det['label'], det['class_id'],
                         det['confidence'], cur, prev, age)

    def _advance(self, t, dt):
        """Extrapolate a track by at most `pred_cap` seconds of its velocity."""
        dt = min(dt, self.pred_cap)
        return (t['cen'][0] + t['vel'][0] * dt,
                t['cen'][1] + t['vel'][1] * dt)

    def update(self, dets, w, h, now=None):
        if now is None:
            now = time.time()
        # Predict each track's centroid. Velocity is px/second, so time spent
        # coasting (including a multi-second stream gap) is extrapolated
        # correctly rather than treated as a single frame of motion.
        preds = {}
        for tid, t in self.tracks.items():
            dt = now - t['last_t']
            if dt < 0.0:
                dt = 0.0
            px, py = self._advance(t, dt)
            preds[tid] = (px, py, dt)
        cand = []
        for tid, (px, py, dt) in preds.items():
            for i, d in enumerate(dets):
                cx, cy = d['centroid']
                cand.append((((px - cx) ** 2 + (py - cy) ** 2) ** 0.5, tid, i, dt))
        cand.sort()

        base = self.assoc_frac * w
        diag = (w * w + h * h) ** 0.5
        max_gate = max(base, self.gap_max_frac * diag)
        used_t, used_d, pairs = set(), set(), []
        for dist, tid, i, dt in cand:
            if tid in used_t or i in used_d:
                continue
            gate = min(base + self.gap_frac * w * dt, max_gate)
            if dist > gate:
                continue
            if dt >= 1.0 and self.debug:
                print(f"[track] re-acquired #{tid} after {dt:.1f}s gap "
                      f"(dist={dist:.0f} gate={gate:.0f})")
            used_t.add(tid); used_d.add(i); pairs.append((tid, i))

        objects = []
        for tid, i in pairs:
            tr, det = self.tracks[tid], dets[i]
            prev = tr['meas']
            cur = det['centroid']
            dt = now - tr['meas_t']
            if dt <= 1e-3:
                dt = 1e-3
            vx, vy = (cur[0] - prev[0]) / dt, (cur[1] - prev[1]) / dt
            if dt > self.vel_reset:      # long gap: trust the fresh average
                tr['vel'] = (vx, vy)
            else:
                s = self.vel_smooth
                tr['vel'] = (s * vx + (1 - s) * tr['vel'][0],
                             s * vy + (1 - s) * tr['vel'][1])
            tr['cen'] = tr['meas'] = cur
            tr['last_t'] = tr['meas_t'] = now
            tr['age'] += 1
            tr['missed'] = 0
            tr['det'] = det
            objects.append(self._info(tid, det, prev, cur, tr['age']))

        # Coast tracks with no detection this frame (predicted position only).
        for tid, tr in self.tracks.items():
            if tid in used_t:
                continue
            dt = now - tr['last_t']
            if dt < 0.0:
                dt = 0.0
            tr['cen'] = self._advance(tr, dt)
            tr['last_t'] = now
            tr['age'] += 1
            tr['missed'] += 1
            # prev_centroid=None marks a prediction so crossing is not evaluated.
            objects.append(self._info(tid, tr['det'], None, tr['cen'], tr['age']))

        # New tracks for unmatched detections.
        for i, det in enumerate(dets):
            if i in used_d:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'cen': det['centroid'], 'meas': det['centroid'],
                                'vel': (0.0, 0.0), 'age': 0, 'missed': 0, 'det': det,
                                'last_t': now, 'meas_t': now}
            objects.append(self._info(tid, det, None, det['centroid'], 0))

        for tid in [t for t, tr in self.tracks.items() if tr['missed'] > self.max_age]:
            del self.tracks[tid]
        return objects

# ── Annotation ────────────────────────────────────────────────────────
def annotate_frame(frame, line, roi, objects, counts, fps, det_count, flip_sides=False, simple=False):
    h, w = frame.shape[:2]
    if not simple:
        if roi is not None and len(roi) >= 3:
            pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (100, 180, 255), 2)
            xs = np.array([p[0] for p in roi]); ys = np.array([p[1] for p in roi])
            cv2.rectangle(frame, (xs.min(), ys.min()), (xs.max(), ys.max()), (80, 80, 80), 1)
        if line:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (59, 130, 246), 3)
            dx, dy = x2 - x1, y2 - y1
            length = (dx*dx + dy*dy) ** 0.5
            if length > 0:
                ux, uy = dx/length, dy/length
                px, py = -uy*30, ux*30
                mx, my = (x1+x2)//2, (y1+y2)//2
                in_pos  = (mx + int(px), my + int(py))
                out_pos = (mx - int(px), my - int(py))
                if flip_sides: in_pos, out_pos = out_pos, in_pos
                cv2.putText(frame, "IN",  (in_pos[0]-10,  in_pos[1]-6),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34,197,94),  2)
                cv2.putText(frame, "OUT", (out_pos[0]-20, out_pos[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239,68,68),  2)
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (34,197,94), 2)
            cv2.putText(frame, f"{obj.label} #{obj.track_id}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34,197,94), 2)
            cv2.circle(frame, obj.centroid, 4, (251,191,36), -1)
        cv2.putText(frame, f"{fps:.1f} FPS  Det: {det_count}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136,136,136), 1)
    cv2.putText(frame, f"IN: {counts['in']}",   (w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34,197,94),  2)
    cv2.putText(frame, f"OUT: {counts['out']}", (w-150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239,68,68),  2)
    return frame

class LatestFrame:
    """Thread-safe holder for the most recent frame (reader -> detect + tracking).

    `fresh` is False for a frame the NVR repeated verbatim while the stream was
    stalled. Consumers keep relaying it (so the dashboard does not time out) but
    must not advance the tracker on it, or wall-clock-time-based velocity decays
    to zero and the next genuinely-new frame looks like an unexplained jump.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self.seq = 0
        self.fresh = True

    def put(self, frame, fresh=True):
        with self._lock:
            self._frame = frame
            self.fresh = fresh
            self.seq += 1

    def get(self):
        with self._lock:
            return self._frame, self.seq, self.fresh


def detect_loop(detector, cfg, latest, shared, stop):
    """Runs detection as fast as the CPU allows on the newest frame.

    Decoupled from the tracking loop: tracking/crossing proceeds at the camera's
    full sample rate even while a slow (YOLO) inference is still running, because
    the tracker coasts on velocity prediction between fresh detections.
    """
    last = -1
    while not stop.is_set():
        frame, seq, fresh = latest.get()
        if frame is None or seq == last:
            time.sleep(0.005)
            continue
        last = seq
        if not fresh:                        # NVR repeat; don't burn CPU on it
            continue
        if float(frame.std()) <= 5:          # warm-up / dead frames
            continue
        h, w = frame.shape[:2]
        roi_px = cfg.pixel_roi(w, h)
        # Read the class filter every frame so a remote config change applies live.
        classes = None if cfg.detector == "motion" else cfg.enabled_classes
        try:
            if roi_px and len(roi_px) >= 3:
                dets = detector.detect_roi(frame, roi_px, enabled_classes=classes)
            else:
                dets = detector.detect(frame, enabled_classes=classes)
        except Exception as e:
            print("[detect] error:", e)
            continue
        if len(dets) > MAX_DETECTIONS_PER_FRAME:
            print(f"[detect] implausible detections ({len(dets)}) — glitch, ignoring")
            dets = []
        with shared["lock"]:
            shared["dets"] = dets
            shared["counter"] += 1


# ── Reader thread: RTSP capture with reconnect ────────────────────────
# If the stream only hands back repeated/stale frames (or none) for this long,
# force a reconnect: the NVR can stall silently while the process looks healthy.
READ_TIMEOUT = float(os.environ.get("TRACKER_READ_TIMEOUT", "15"))


def reader_loop(cfg, latest):
    """Reads camera frames and stores the latest one for detect + tracking.

    Must never die: a raise here silently stops all counting while the process
    stays up. Any unexpected error is logged and the stream is reopened.
    """
    cap = None
    relay = None
    relay_url = None
    last_read_t = 0.0
    read_interval = 0.0
    prev_gray = None
    last_fresh = time.time()

    while cfg.running:
        try:
            if cap is not None and cfg.check_reconnect():
                print("[reader] URL changed, reconnecting...")
                cap.release(); cap = None

            # Stall watchdog: reconnect if no genuinely-new frame has arrived
            # for too long (`stimeout`/`rw_timeout` bound a blocked read; this
            # catches a stream that keeps returning the same stale frame).
            if cap is not None and time.time() - last_fresh > READ_TIMEOUT:
                print(f"[reader] no new frames for {READ_TIMEOUT:.0f}s, reconnecting...")
                cap.release(); cap = None

            if cap is None:
                url = cfg.stream_url
                if not url:
                    time.sleep(1); continue
                # Optional GStreamer pipeline (e.g. Pi hardware H.264 decode via
                # v4l2h264dec) to move decode off the CPU that YOLO needs.
                pipeline = os.environ.get("TRACKER_CAPTURE_PIPELINE", "")
                if pipeline:
                    source, api = pipeline, cv2.CAP_GSTREAMER
                elif cfg.relay:
                    # SHA-256 RTSP Digest camera: FFmpeg can't authenticate, so an
                    # RtspRelay terminates auth and re-serves RTP as a local SDP.
                    if relay is not None and relay_url != url:
                        relay.stop(); relay = None
                    if relay is None:
                        relay = RtspRelay(
                            url,
                            sdp_path=os.environ.get(
                                "TRACKER_RELAY_SDP", "/tmp/tracker_rtsp_relay.sdp"),
                            local_port=int(os.environ.get("TRACKER_RELAY_PORT", "5004")),
                            auth=os.environ.get("TRACKER_RELAY_AUTH", "sha256"))
                        relay_url = url
                        relay.start()
                    if not relay.ready.wait(timeout=12):
                        print("[reader] RTSP relay not ready, retrying in 5s...")
                        relay.stop(); relay = None
                        time.sleep(5); continue
                    opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
                    if "protocol_whitelist" not in opts:
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                            opts + "|" if opts else "") + "protocol_whitelist;file,udp,rtp"
                    source, api = relay.sdp_path, cv2.CAP_FFMPEG
                else:
                    source, api = url, cv2.CAP_FFMPEG
                cap = cv2.VideoCapture(source, api)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the newest frame
                if not cap.isOpened():
                    print("[reader] Failed to open stream, retrying in 5s...")
                    cap.release(); cap = None; time.sleep(5); continue
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                read_interval = 1.0 / video_fps if video_fps > 0 else 0.0
                last_read_t = time.time()
                last_fresh = time.time()
                if cfg.relay and not pipeline:
                    print(f"[reader] Stream via RTSP relay ({video_fps:.2f} fps)")
                else:
                    print(f"[reader] Stream connected ({video_fps:.2f} fps)")

            # Pace reads to match video's native framerate
            if read_interval > 0:
                elapsed = time.time() - last_read_t
                if elapsed < read_interval:
                    time.sleep(read_interval - elapsed)

            ret, frame = cap.read()
            last_read_t = time.time()
            if not ret:
                print("[reader] Stream lost, reconnecting...")
                cap.release(); cap = None; time.sleep(1); continue

            # Flag frames the NVR repeats while stalled. A repeated frame is
            # bit-identical, so *no* pixels change; count significantly-changed pixels
            # (rather than a mean) so even a small/distant mover still reads as fresh.
            # Copy so the decoder can't overwrite the buffer a consumer still holds.
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                fresh = True
            else:
                changed = int((cv2.absdiff(gray, prev_gray) > 6).sum())
                fresh = changed > 3
            prev_gray = gray
            if fresh:
                last_fresh = time.time()
            latest.put(frame.copy(), fresh)
        except Exception as e:
            print(f"[reader] error: {e}; reconnecting...", flush=True)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            time.sleep(1)

    if cap is not None:
        cap.release()
    if relay is not None:
        relay.stop()
    print("[reader] Done")

# Live feed target: 360p, best-effort (drop frames rather than block).
LIVE_HEIGHT  = int(os.environ.get("TRACKER_LIVE_HEIGHT", "360"))
LIVE_QUALITY = int(os.environ.get("TRACKER_LIVE_QUALITY", "65"))

def annotate_live(frame, line, objects, counts, target_h=LIVE_HEIGHT, roi=None, flip=False):
    """Downscale to the target height and draw a crisp overlay (boxes scaled to match)."""
    h, w = frame.shape[:2]
    s = target_h / h if h > target_h else 1.0
    img = (cv2.resize(frame, (max(1, int(w * s)), target_h), interpolation=cv2.INTER_AREA)
           if s < 1.0 else frame.copy())
    if roi is not None and len(roi) >= 3:
        pts = np.array([[int(p[0] * s), int(p[1] * s)] for p in roi], dtype=np.int32)
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, (100, 180, 255), 1, cv2.LINE_AA)
    if line:
        x1, y1, x2, y2 = (int(v * s) for v in line)
        cv2.line(img, (x1, y1), (x2, y2), (59, 130, 246), 2)
        dx, dy = x2 - x1, y2 - y1
        L = (dx * dx + dy * dy) ** 0.5
        if L > 1:
            ux, uy = dx / L, dy / L
            off = 16
            px, py = -uy * off, ux * off
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            in_pos = (int(mx + px), int(my + py))
            out_pos = (int(mx - px), int(my - py))
            if flip:
                in_pos, out_pos = out_pos, in_pos
            cv2.putText(img, "IN", in_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2, cv2.LINE_AA)
            cv2.putText(img, "OUT", out_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2, cv2.LINE_AA)
    for obj in objects:
        x1, y1, x2, y2 = (int(v * s) for v in obj.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cv2.putText(img, f"{obj.label} #{obj.track_id}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1, cv2.LINE_AA)
    return img

def live_loop(url, token, live_q, cfg):
    """POSTs annotated JPEG frames to the VPS live endpoint (outbound only)."""
    import requests
    headers = {"Authorization": f"Bearer {token}"}
    while cfg.running:
        try:
            jpeg = live_q.get(timeout=1.0)
        except Empty:
            continue
        try:
            requests.post(url, data=jpeg, headers=headers, timeout=10)
        except Exception as e:
            _log_throttled("live", f"[live] post failed: {e}", every=60)

def cam2_loop(cfg, live2_q, stop):
    """Relay camera 2's substream to the dashboard as raw JPEGs (no detection).

    An idle edge consumes nothing: the RTSP session is only opened while
    ``live2_wanted`` is set (someone is watching /live) and is torn down a few
    seconds after it clears, so the camera stops sending and no decode runs.
    Frames are only decoded to re-encode a small JPEG; there is no inference.
    """
    relay = None
    cap = None
    idle_since = None
    last_live = 0.0
    while not stop.is_set():
        if not cfg.live2_wanted:
            # Drop the session after a short grace so brief reconnects don't flap.
            if cap is not None:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since >= 5.0:
                    cap.release(); cap = None
                    if relay is not None: relay.stop(); relay = None
                    idle_since = None
                    print("[cam2] idle, stream closed")
            time.sleep(0.2)
            continue
        idle_since = None
        try:
            if cap is None:
                relay = RtspRelay(cfg.cam2_url, sdp_path=cfg.cam2_sdp,
                                  local_port=cfg.cam2_port, auth="sha256")
                relay.start()
                if not relay.ready.wait(timeout=12):
                    print("[cam2] relay not ready, retrying in 5s")
                    relay.stop(); relay = None
                    time.sleep(5); continue
                cap = cv2.VideoCapture(relay.sdp_path, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print("[cam2] open failed, retrying in 5s")
                    cap.release(); cap = None; relay.stop(); relay = None
                    time.sleep(5); continue
                print("[cam2] live (substream)")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("read failed")
            now = time.time()
            if now - last_live >= 0.2:
                last_live = now
                h, w = frame.shape[:2]
                if h > LIVE_HEIGHT:
                    frame = cv2.resize(frame, (max(1, int(w * LIVE_HEIGHT / h)), LIVE_HEIGHT),
                                       interpolation=cv2.INTER_AREA)
                ok2, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, LIVE_QUALITY])
                if ok2:
                    try: live2_q.put_nowait(jpg.tobytes())
                    except Full: pass
        except Exception as e:
            if stop.is_set():
                break
            print(f"[cam2] {e}; reconnecting in 3s")
            if cap is not None:
                cap.release(); cap = None
            if relay is not None:
                try: relay.stop()
                except Exception: pass
                relay = None
            time.sleep(3)
    if cap is not None:
        cap.release()
    if relay is not None:
        relay.stop()
    print("[cam2] Done")

def fetch_capture_jpeg(cfg, timeout=6.0):
    """GET one full-resolution still from the camera's main profile.

    The camera hardware-encodes the JPEG, so this costs the Pi one HTTP GET and
    no video decode. Returns bytes, or None on any failure.

    A persistent Session keeps the TLS connection and the digest nonce alive
    between crossings, so the still is fetched with (ideally) a single
    round-trip and lands as close as possible to the crossing instant.
    """
    import requests
    from requests.auth import HTTPDigestAuth
    global _CAPTURE_SESSION
    if not cfg.capture_url:
        return None
    if _CAPTURE_SESSION is None:
        _CAPTURE_SESSION = requests.Session()
    try:
        r = _CAPTURE_SESSION.get(
            cfg.capture_url,
            auth=HTTPDigestAuth(cfg.capture_user, cfg.capture_password),
            verify=False, timeout=timeout)
        if r.ok and r.content and r.headers.get("content-type", "").startswith("image/"):
            return r.content
    except Exception as e:
        _log_throttled("capture_fetch", f"[capture] snapshot fetch failed: {e}", every=60)
    return None


_CAPTURE_SESSION = None

# How long a queued crossing may wait for the main-stream snapshot before we
# stop trusting the fetch and fall back to the (accurately-timed) substream
# frame for the full image as well.
CAPTURE_MAX_AGE = float(os.environ.get("TRACKER_CAPTURE_MAX_AGE", "1.5"))


def _safe_store_add(store, ev, tries=3):
    """INSERT one event, retrying a briefly-locked DB. Never raises."""
    for attempt in range(tries):
        try:
            store.add(ev)
            return True
        except Exception as e:
            if "locked" in str(e).lower() and attempt < tries - 1:
                time.sleep(0.5)
                continue
            print(f"[capture] store.add failed for {ev.get('id')}: {e}", flush=True)
            return False
    return False


def persist_capture(cfg, capture_mgr, store, item, jpeg):
    """Save the images + DB row for one crossing. Never raises.

    The capture worker (and the inline overflow path in the track loop) must not
    die on a disk-full / locked-DB / bad-value error: that would silently stop
    every future crossing from being persisted while run.py keeps counting.
    """
    try:
        entry = capture_mgr.store_crossing(
            item["frame"], jpeg, item["track_id"], item["direction"],
            item["crossed_at"])
    except Exception as e:
        print(f"[capture] save failed for ID#{item['track_id']}: {e}", flush=True)
        return None
    cfg.add_capture(entry)
    _safe_store_add(store, {
        "id": entry["filename"].rsplit(".", 1)[0],
        "track_id": item["track_id"],
        "class_id": item["class_id"],
        "label": item["label"],
        "confidence": item["confidence"],
        "direction": entry["direction"],
        "crossed_at": item["crossed_at"],
        "bbox": item["bbox"],
        "line": item["line"],
        "image_path": entry["filename"],
        "thumb_path": entry["thumb"],
        "sub_path": entry["sub"],
    })
    return entry


def capture_loop(cfg, capture_mgr, store, capture_q):
    """Persist crossings on a worker thread (fetch/save off the track loop).

    Prefers a full-resolution main-stream still for the full image; the
    thumbnail always comes from the substream frame captured at the crossing.
    If the snapshot is stale (the worker was busy with earlier crossings) or
    unavailable, the substream frame is used for the full image too — an
    accurately-timed low-res frame beats a crisp frame of the wrong moment.
    """
    while cfg.running:
        try:
            item = capture_q.get(timeout=1.0)
        except Empty:
            continue
        try:
            age = time.time() - item.get("enqueued_at", time.time())
            jpeg = fetch_capture_jpeg(cfg) if age <= CAPTURE_MAX_AGE else None
            if jpeg is not None:
                print(f"[capture] ID#{item['track_id']} main-stream ({len(jpeg) // 1024}KB)")
            else:
                why = "stale" if age > CAPTURE_MAX_AGE else "fallback"
                print(f"[capture] ID#{item['track_id']} substream ({why}, {age:.2f}s)")
            persist_capture(cfg, capture_mgr, store, item, jpeg)
        except Exception as e:
            print(f"[capture] unexpected error for ID#{item.get('track_id')}: {e}",
                  flush=True)


def config_loop(cfg):
    """Polls the server for line/scan-area config and applies changes."""
    import requests
    headers = {"Authorization": f"Bearer {cfg.token}"}
    while cfg.running:
        try:
            r = requests.get(cfg.config_url, headers=headers, timeout=5)
            if r.ok:
                data = r.json()
                cfg.live_wanted = bool(data.get("live_wanted", False))
                cfg.live2_wanted = bool(data.get("live2_wanted", False))
                if data.get("updated_at") != cfg.config_updated_at:
                    cfg.apply_remote(data)
                    print(f"[config] applied remote config (line={cfg.norm_line}, roi={bool(cfg.norm_roi)}, "
                          f"flip={cfg.flip_sides})")
        except Exception as e:
            _log_throttled("config", f"[config] poll failed: {e}", every=60)
        time.sleep(3)

def prune_loop(store, capture_dir, retention_days):
    """Hourly sweep: delete local events/captures older than retention_days."""
    while cfg.running:
        try:
            paths = store.prune(retention_days)
            for rel in paths:
                try:
                    os.remove(os.path.join(capture_dir, rel))
                except OSError:
                    pass
            if paths:
                print(f"[prune] expired {len(paths)} files older than {retention_days}d")
        except Exception as e:
            _log_throttled("prune", f"[prune] error: {e}", every=60)
        time.sleep(3600)

# ── Main: detect + track loop ─────────────────────────────────────────
def main():
    print("[main] Starting Vehicle Line Counter")
    print(f"[main] Stream: {cfg.stream_url}")

    if cfg.detector == "motion":
        detector    = MotionDetector(scale=cfg.motion_scale, min_area=cfg.motion_min_area)
        print(f"[main] Motion detector (scale={cfg.motion_scale}, min_area={cfg.motion_min_area})")
    elif cfg.detector == "yolo":
        detector    = YoloNcnn(conf_thresh=cfg.conf_thresh)
        print(f"[main] YOLO11 ncnn loaded ({YOLO_PARAM.split('/')[-2]}, {NUM_THREADS} threads)")
    else:
        detector    = HybridDetector(conf_thresh=cfg.conf_thresh,
                                     motion_scale=cfg.motion_scale,
                                     motion_min_area=cfg.motion_min_area,
                                     hold_seconds=cfg.motion_hold,
                                     heartbeat=cfg.motion_heartbeat)
        print(f"[main] Hybrid detector: motion-gated YOLO11 "
              f"({YOLO_PARAM.split('/')[-2]}, {NUM_THREADS} threads)")

    # Low-FPS friendly: associate by predicted centroid distance, not IoU, so a
    # fast vehicle that clears its own box between 10 fps samples keeps its id.
    tracker     = CentroidTracker(
        max_age=int(os.environ.get("TRACKER_TRACK_MAX_AGE", "25")),
        assoc_frac=float(os.environ.get("TRACKER_ASSOC_FRAC", "0.2")),
        gap_frac=float(os.environ.get("TRACKER_ASSOC_GAP_FRAC", "0.5")),
        gap_max_frac=float(os.environ.get("TRACKER_ASSOC_MAX_FRAC", "1.0")),
        vel_reset=float(os.environ.get("TRACKER_VEL_RESET", "0.4")),
        pred_cap=float(os.environ.get("TRACKER_ASSOC_PRED_CAP", "0.5")),
        debug=os.environ.get("TRACKER_DEBUG") == "1")
    capture_mgr = CaptureManager(cfg.capture_dir, cfg.max_captures)

    # Offline-first event store (SQLite); survives restarts and outages
    store = Store(os.environ.get("TRACKER_DB_PATH", "events.db"))

    last_cross_info = {}
    # Recent crossings, to collapse one vehicle counted twice (two tracks from
    # two class-boxes) into a single event.
    recent_cross    = []
    gate_ids        = set()
    cross_gate      = CrossingGate(min_age=cfg.min_track_age,
                                   min_travel_frac=cfg.min_travel_frac)

    latest = LatestFrame()
    shared = {"dets": [], "counter": 0, "lock": threading.Lock()}
    stop = threading.Event()

    threading.Thread(target=reader_loop, args=(cfg, latest), daemon=True).start()
    threading.Thread(target=detect_loop,
                     args=(detector, cfg, latest, shared, stop), daemon=True).start()
    print("[main] Reader + detect threads started")

    # Full-res evidence: a worker fetches the camera's main-stream still at each
    # crossing (substream fallback) so a network fetch never stalls tracking.
    capture_q = Queue(maxsize=64)
    threading.Thread(target=capture_loop, args=(cfg, capture_mgr, store, capture_q),
                     daemon=True).start()
    print("[main] Capture worker started")

    # Live relay to the VPS dashboard (optional; outbound POST)
    live_url   = os.environ.get("TRACKER_LIVE_URL", "")
    live_token = os.environ.get("TRACKER_TOKEN", "")
    live_q     = Queue(maxsize=1)
    if live_url:
        threading.Thread(target=live_loop, args=(live_url, live_token, live_q, cfg),
                         daemon=True).start()
        print(f"[main] Live relay -> {live_url}")

    # Second camera (raw substream) — only streams while someone watches /live.
    live2_q = Queue(maxsize=1)
    if cfg.cam2_url and cfg.live2_url:
        threading.Thread(target=live_loop, args=(cfg.live2_url, live_token, live2_q, cfg),
                         daemon=True).start()
        threading.Thread(target=cam2_loop, args=(cfg, live2_q, stop), daemon=True).start()
        print(f"[main] Camera 2 live -> {cfg.live2_url}")

    # Pull line / scan-area config from the server (Pi is outbound-only)
    if cfg.config_url:
        threading.Thread(target=config_loop, args=(cfg,), daemon=True).start()
        print(f"[main] Config poll -> {cfg.config_url}")

    # Auto-expire local events/captures older than the retention window
    retention_days = int(os.environ.get("TRACKER_RETENTION_DAYS", "30"))
    threading.Thread(target=prune_loop, args=(store, cfg.capture_dir, retention_days),
                     daemon=True).start()
    print(f"[main] Retention: {retention_days} days")

    _counts = store.total_counts()
    c_in  = _counts.get("in", 0)
    c_out = _counts.get("out", 0)
    total_frames = 0   # non-resetting counter for crossing debounce
    last_live = 0.0
    last_frame_seq = -1
    used_det = -1
    stat_t = time.time()
    stat_frames = 0
    stat_dets = 0
    objects = []          # carried across stalled (repeated) frames for the live feed
    prev_frame_t = 0.0

    try:
        while cfg.running:
            frame, fseq, fresh = latest.get()
            if frame is None or fseq == last_frame_seq:
                time.sleep(0.01)
                continue
            last_frame_seq = fseq
            if float(frame.std()) <= 5:
                continue

            now_t = time.time()
            h, w = frame.shape[:2]
            line_px = cfg.pixel_line(w, h)
            roi_px  = cfg.pixel_roi(w, h)

            if fresh:
                if prev_frame_t and now_t - prev_frame_t > 2.5:
                    print(f"[gap] no new frames for {now_t - prev_frame_t:.1f}s")
                prev_frame_t = now_t
                total_frames += 1

                # Consume fresh detections if the detect thread produced any; else
                # feed an empty list so the tracker coasts between detector updates.
                with shared["lock"]:
                    dcount = shared["counter"]
                    raw_detections = list(shared["dets"])
                if dcount == used_det:
                    raw_detections = []
                else:
                    used_det = dcount

                # Update the centroid tracker with fresh detections (empty list
                # just coasts tracks between detector updates).
                objects = tracker.update(raw_detections, w, h, now=now_t)

                # Flicker gate keeps its "first seen" position per live track id.
                present = set()
                for obj in objects:
                    present.add(obj.track_id)
                    cross_gate.update_first(obj.track_id, obj.centroid)
                cross_gate.drop(gate_ids - present)
                gate_ids = present

            # Push annotated frame to the VPS live feed (~5 fps, non-blocking).
            # Keep doing this even on a repeated frame so a static scene does not
            # time out the dashboard (it only relays, it does not track).
            if live_url and cfg.live_wanted and now_t - last_live >= 0.2:
                last_live = now_t
                ann = annotate_live(frame, line_px, objects, {"in": c_in, "out": c_out},
                                    roi=roi_px, flip=cfg.flip_sides)
                ok, jpeg = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, LIVE_QUALITY])
                if ok:
                    try: live_q.put_nowait(jpeg.tobytes())
                    except Full: pass

            # Periodic rate report (tracking is decoupled from detection).
            if time.time() - stat_t >= 10:
                el = time.time() - stat_t
                with shared["lock"]:
                    dc = shared["counter"]
                print(f"[stat] track={(total_frames - stat_frames) / el:.1f} fps  "
                      f"detect={(dc - stat_dets) / el:.1f} fps  tracks={len(objects)}")
                stat_t, stat_frames, stat_dets = time.time(), total_frames, dc

            if not fresh:
                continue

            debug = os.environ.get("TRACKER_DEBUG") == "1"
            for obj in objects:
                if obj.age < cfg.min_track_age:
                    if debug:
                        print(f"[dbg] #{obj.track_id} {obj.label} age={obj.age} "
                              f"c={obj.centroid} (wait)")
                    continue
                crossing = cross_gate.crossing(line_px, obj, flip=cfg.flip_sides,
                                               margin=cfg.cross_margin_frac * h)
                if debug:
                    print(f"[dbg] #{obj.track_id} {obj.label} age={obj.age} c={obj.centroid} "
                          f"side={cross_gate.side.get(obj.track_id)} cross={crossing}")
                if crossing == CROSSING_NONE:
                    continue
                # Reject flicker: real age + net travel + cooldown
                if not cross_gate.allow(obj, now_t, h):
                    continue
                info = last_cross_info.get(obj.track_id, {'frame': -60, 'dir': None})
                if total_frames - info['frame'] < 15:
                    continue
                direction = 'IN' if crossing == CROSSING_IN else 'OUT'
                # Safety net: if another track crossed the same way here moments
                # ago (the duplicate track from a second class-box), count once.
                if CROSS_DEDUP_S > 0:
                    recent_cross = [rc for rc in recent_cross
                                    if now_t - rc[0] <= CROSS_DEDUP_S]
                    if any(rc[1] == crossing and box_iou(obj.bbox, rc[2]) > DUP_IOU
                           for rc in recent_cross):
                        if debug:
                            print(f"[dbg] #{obj.track_id} duplicate crossing "
                                  f"{direction} suppressed")
                        continue
                    recent_cross.append((now_t, crossing, tuple(obj.bbox)))
                last_cross_info[obj.track_id] = {'frame': total_frames, 'dir': crossing}
                # Hand the capture to a worker: it fetches a full-res main-stream
                # still (substream fallback) without stalling the track loop.
                item = {
                    "track_id": obj.track_id, "class_id": obj.class_id,
                    "label": obj.label, "confidence": obj.confidence,
                    "direction": crossing, "bbox": list(obj.bbox),
                    "line": line_px, "frame": frame.copy(),
                    "crossed_at": datetime.now(timezone.utc).isoformat(),
                    "enqueued_at": time.time(),
                }
                try:
                    capture_q.put_nowait(item)
                except Full:
                    # Worker backlogged: persist inline from the substream frame
                    # so the in-memory count and the DB can never diverge.
                    _log_throttled(
                        "capture_full",
                        "[capture] queue full — persisting crossing inline (substream only)",
                        every=30)
                    persist_capture(cfg, capture_mgr, store, item, None)
                if crossing == CROSSING_IN:
                    c_in += 1
                else:
                    c_out += 1
                print(f"[cross] ID#{obj.track_id} {direction}")

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
    except Exception as e:
        print(f"[main] Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        cfg.running = False
        print("[main] Done")

if __name__ == "__main__":
    main()
