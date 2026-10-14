import json, threading, time, os, sys, signal
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


# ── Worker liveness ───────────────────────────────────────────────────
# Every long-lived loop stamps its name here each iteration. A thread that
# *raises* is already guarded inside its loop; this catches the other failure
# mode — a thread wedged on a lock or a stuck C call — which otherwise leaves
# the process alive but no longer counting.
_HEARTBEAT = {}


def _beat(name):
    _HEARTBEAT[name] = time.time()


# Staleness budget per worker (seconds). Must exceed the worker's own longest
# idle wait plus its slowest single step, or it is a false positive.
WORKER_TIMEOUTS = {
    "reader": 60.0, "detect": 60.0, "capture": 60.0, "config": 60.0,
    "live": 60.0, "cam2": 60.0, "prune": 7200.0,
}
# These three are on the counting path: a stall means crossings may be missed,
# so the supervisor forces a (draining) restart rather than just logging.
CRITICAL_WORKERS = {"reader", "detect", "capture"}


def supervisor_loop(cfg, stop):
    """Force a draining restart if a counting worker wedges.

    Setting ``cfg.running=False`` lets the main loop exit and drain the capture
    queue first; systemd ``Restart=always`` then brings us back with fresh
    threads. Without this, a deadlock in one worker leaves the process looking
    healthy while counts silently stop.
    """
    while cfg.running and not stop.is_set():
        now = time.time()
        for name, budget in WORKER_TIMEOUTS.items():
            last = _HEARTBEAT.get(name)
            if last is None:
                continue
            age = now - last
            if age <= budget:
                continue
            if name in CRITICAL_WORKERS:
                print(f"[supervisor] critical worker '{name}' stalled {age:.0f}s; "
                      f"restarting", flush=True)
                cfg.running = False
                stop.set()
                return
            _log_throttled(f"hb_{name}",
                           f"[supervisor] worker '{name}' stalled {age:.0f}s",
                           every=60)
        stop.wait(10)


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
# Overlap alone can also suppress a genuinely separate vehicle queued right
# behind another. So the de-dupe additionally demands the two boxes be the *same
# vehicle*: their centres within this fraction of the frame width AND one box
# almost entirely inside the other. A follower's centre is displaced along the
# road, so it survives.
DEDUP_CENTER_FRAC = float(os.environ.get("TRACKER_DEDUP_CENTER_FRAC", "0.03"))
# Evidence: the saved full image comes from the camera's main profile; the
# thumbnail is always built from the substream crossing frame, so the timeline
# shows the vehicle at the line.
#   "1"/"snap" (default) — the camera's ONVIF main-profile snapshot; the camera
#                          encodes the JPEG (no Pi video decode) and it tracks the
#                          live stream (measured live via the burned-in OSD)
#   "rtsp"               — decode a fresh main-stream frame per crossing. Measured
#                          NOT to help: a fresh RTSP connect starts at the NVR's
#                          last I-frame (main GOP=50 ≈ 2 s) and costs ~1.4 s open
#                          + 1.2 MB, so it is no fresher than the snapshot.
#   "0"/"sub"            — use the substream crossing frame as the full image
# The real evidence-vs-crossing mismatch was the *reader* drifting behind live
# (see reader_loop), not the snapshot; fixing the reader is what aligns them.
# Also settable in config.json as "capture_main_stream" (no root needed).
CAPTURE_MAIN_MODE = os.environ.get("TRACKER_CAPTURE_MAIN_STREAM", "1").lower()


def _derive_main_rtsp(url):
    """Map a substream RTSP URL's channel to its main-stream sibling.

    ``…/Streaming/Channels/102`` -> ``…/Streaming/Channels/101`` (and ``202`` ->
    ``201``); the NVR/camera encode the full field of view on both profiles.
    Returns "" if the URL doesn't look like a channel path.
    """
    if not url.startswith("rtsp://"):
        return ""
    base, _, _ = url.partition("?")           # drop any query string
    import re
    m = re.search(r"(/Streaming/Channels/[12])02?$", base)
    if not m:
        return ""
    return base[:m.start()] + m.group(1) + "01"



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
        self.detector = os.environ.get("TRACKER_DETECTOR", "hybrid")  # "hybrid" | "motion" | "yolo" | "opencv"
        # OpenCV DNN (cv2.dnn) ONNX path, used when detector == "opencv".
        self.dnn_model = os.environ.get(
            "TRACKER_DNN_MODEL", "/opt/tracker/models/object_detection_nanodet_2022nov.onnx")
        self.motion_scale = float(os.environ.get("TRACKER_MOTION_SCALE", "0.5"))
        self.motion_min_area = int(os.environ.get("TRACKER_MOTION_MIN_AREA", "12"))
        self.motion_hold = float(os.environ.get("TRACKER_MOTION_HOLD", "60"))
        self.motion_heartbeat = float(os.environ.get("TRACKER_MOTION_HEARTBEAT", "3"))
        self.min_track_age = int(os.environ.get("TRACKER_MIN_TRACK_AGE", "1"))
        # Hysteresis past the line (fraction of frame height) to reject jitter.
        # Wide enough that a stationary/working vehicle parked on the line (its
        # bbox centroid wandering as the detector re-fits it) can't flip sides.
        self.cross_margin_frac = float(os.environ.get("TRACKER_CROSS_MARGIN_FRAC", "0.06"))
        # After a crossing commits, the track must move this fraction of the
        # frame height *away from the crossing point* before another crossing
        # can commit. A vehicle that only drifts back and forth at the line
        # never re-arms, so it is counted once (see CrossingGate).
        self.cross_rearm_frac = float(os.environ.get("TRACKER_CROSS_REARM_FRAC", "0.2"))
        # Confirmed transit: a crossing only *counts* once the centroid reaches
        # this fraction of the frame height past the line — deeper than the
        # hysteresis margin — so a machine oscillating on the line (or a bbox
        # refit that flicks the centroid across) never commits. A return before
        # the confirm distance cancels the pending crossing. 0 disables it
        # (legacy: count as soon as the margin is cleared).
        self.cross_confirm_frac = float(os.environ.get("TRACKER_CONFIRM_FRAC", "0.2"))
        # Which point of the box crosses the line: "centroid" (default), or the
        # box's "left"/"right" edge. Use an edge when another vehicle occludes
        # the crossing and the detector merges them into one box (the merged
        # box's leading edge still tracks the moving vehicle). Also configurable
        # in config.json as "probe".
        self.cross_probe = os.environ.get("TRACKER_PROBE", "centroid")
        # No-count polygon (normalized). Crossings are frozen while the centroid
        # is inside it — e.g. the excavator/loader's working area. Pi-local, set
        # from config.json; a remote config may also provide it.
        self.keepout = None       # legacy pixel coords
        self.norm_keepout = None  # normalized [[x,y],...] (0..1)
        # ncnn inference threads; also read from config.json so it can be lowered
        # on-device (heat/CPU) without editing the root-owned tracker.env.
        self.threads = int(os.environ.get("TRACKER_THREADS", str(NUM_THREADS)))
        # Model selection, overridable per-device from config.json so the model
        # can be A/B'd without editing the root-owned tracker.env.
        self.model = YOLO_MODEL
        self.model_imgsz = YOLO_IMGSZ
        self.int8 = YOLO_INT8
        # Square input of the `nanodet`/`picodet` ncnn graph (must match its
        # export). Independent of the YOLO model_imgsz above.
        self.dnn_imgsz = os.environ.get("TRACKER_DNN_IMGSZ", "416")
        # Tracker backend: "centroid" (the in-house IoU+distance tracker) or
        # "bytetrack" (Kalman + two-stage IoU). Overridable from config.json so
        # it can be A/B'd on the edge without editing the root-owned tracker.env.
        self.tracker = os.environ.get("TRACKER_TRACKER", "centroid")
        # PicoDet/NanoDet ncnn detection settings. NMS follows the PicoDet repo
        # (per-class, IoU 0.3, mean/norm already matched). Score defaults to 0.3,
        # not the repo's 0.5: on 97 real crossing frames 0.5 lost 4/97 frames and
        # a truck box, while 0.3 kept 97/97 (a crossing spans ~50 frames, but the
        # low-confidence frames matter near the line). Set dnn_iou/dnn_conf from
        # config.json to A/B on the edge.
        self.dnn_conf = float(os.environ.get("TRACKER_DNN_CONF", "0.3"))
        self.dnn_iou  = float(os.environ.get("TRACKER_DNN_IOU", "0.3"))
        # Optional class->label remap (config.json "vehicle_labels"), e.g.
        # {"2": "truck", "5": "truck"} on a site where every vehicle is a truck
        # and the detector's car/bus boxes are just truck mislabels.
        self.vehicle_labels = None
        # Debug overlay (hysteresis band + per-track re-arm state) on the live feed.
        self.debug_overlay = os.environ.get("TRACKER_DEBUG_OVERLAY", "0") == "1"
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
        # Evidence saves a full-resolution main-profile frame; the thumbnail
        # comes from the substream crossing frame. Default mode "1" uses the
        # ONVIF main-profile snapshot (no Pi decode); see CAPTURE_MAIN_MODE. Any
        # failure falls back to the substream frame. URL/user/password default to
        # the stream URL's; override with TRACKER_CAPTURE_URL / _USER /
        # _PASSWORD, and the main RTSP URL with TRACKER_CAPTURE_RTSP.
        self.capture_url = os.environ.get("TRACKER_CAPTURE_URL", "")
        self.capture_rtsp = os.environ.get("TRACKER_CAPTURE_RTSP", "")
        self.capture_user = os.environ.get("TRACKER_CAPTURE_USER", "")
        self.capture_password = os.environ.get("TRACKER_CAPTURE_PASSWORD", "")
        self.capture_main_mode = CAPTURE_MAIN_MODE
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
                self.keepout = data.get("keepout")
                self.norm_keepout = data.get("norm_keepout")
                self.conf_thresh = data.get("conf_thresh", 0.5)
                self.flip_sides = data.get("flip_sides", False)
                self.capture_dir = os.environ.get(
                    "TRACKER_CAPTURE_DIR", data.get("capture_dir", "captures"))
                self.max_captures = data.get("max_captures", 1000)
                self.enabled_classes = _counting_classes(
                    data.get("enabled_classes", DEFAULT_ENABLED_CLASSES))
                self.relay = bool(data.get("relay", self.relay))
                self.threads = int(data.get("threads", self.threads))
                self.debug_overlay = bool(data.get("debug_overlay", self.debug_overlay))
                self.cross_rearm_frac = float(
                    data.get("cross_rearm_frac", self.cross_rearm_frac))
                self.cross_confirm_frac = float(
                    data.get("cross_confirm_frac", self.cross_confirm_frac))
                self.cross_probe = data.get("probe", self.cross_probe)
                self.model = data.get("model", self.model)
                self.model_imgsz = data.get("model_imgsz", self.model_imgsz)
                self.int8 = bool(data.get("int8", self.int8))
                self.dnn_imgsz = data.get("dnn_imgsz", self.dnn_imgsz)
                self.detector = data.get("detector", self.detector)
                self.dnn_model = data.get("dnn_model", self.dnn_model)
                self.tracker = data.get("tracker", self.tracker)
                self.dnn_conf = float(data.get("dnn_conf", self.dnn_conf))
                self.dnn_iou  = float(data.get("dnn_iou", self.dnn_iou))
                _vl = data.get("vehicle_labels")
                self.vehicle_labels = ({int(k): str(v) for k, v in _vl.items()}
                                       if isinstance(_vl, dict) else None)
                self.cam2_url = data.get("cam2_url", self.cam2_url)
                self.capture_url = data.get("capture_url", self.capture_url)
                self.capture_rtsp = data.get("capture_rtsp", self.capture_rtsp)
                self.capture_main_mode = str(
                    data.get("capture_main_stream", self.capture_main_mode)).lower()
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
        if not self.capture_rtsp:
            # Main-stream sibling of the substream channel (…/102 -> …/101).
            self.capture_rtsp = _derive_main_rtsp(self.stream_url)
        if not self.capture_url and self.capture_main_mode in ("1", "snap", "snapshot"):
            # Fallback still: the camera's main-profile ONVIF snapshot; the
            # thumbnail still comes from the substream crossing frame.
            self.capture_url = f"https://{u.hostname}/onvif-http/snapshot?Profile_1"

    def save(self):
        with self.lock:
            data = {"stream_url": self.stream_url, "line": self.line, "roi": self.roi,
                    "norm_line": self.norm_line, "norm_roi": self.norm_roi,
                    "keepout": self.keepout, "norm_keepout": self.norm_keepout,
                    "conf_thresh": self.conf_thresh, "flip_sides": self.flip_sides,
                    "capture_dir": self.capture_dir, "max_captures": self.max_captures,
                    "enabled_classes": self.enabled_classes, "relay": self.relay,
                    "threads": self.threads,
                    "debug_overlay": self.debug_overlay,
                    "cross_rearm_frac": self.cross_rearm_frac,
                    "cross_confirm_frac": self.cross_confirm_frac,
                    "probe": self.cross_probe,
                    "model": self.model,
                    "model_imgsz": self.model_imgsz,
                    "int8": self.int8,
                    "dnn_imgsz": self.dnn_imgsz,
                    "detector": self.detector,
                    "dnn_model": self.dnn_model,
                    "tracker": self.tracker,
                    "dnn_conf": self.dnn_conf,
                    "dnn_iou": self.dnn_iou,
                    "vehicle_labels": ({str(k): v for k, v in self.vehicle_labels.items()}
                                       if self.vehicle_labels else None),
                    "cam2_url": self.cam2_url, "capture_url": self.capture_url,
                    "capture_rtsp": self.capture_rtsp,
                    "capture_main_stream": self.capture_main_mode}
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
            # The dashboard is the source of truth for the keep-out polygon too
            # (drawn in Camera setup). Apply it, including a clear back to None,
            # when the key is present so the UI can both set and remove it.
            if "keepout" in data:
                self.norm_keepout = data["keepout"]
                self.keepout = None
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

    def pixel_keepout(self, w, h):
        with self.lock:
            if self.norm_keepout and len(self.norm_keepout) >= 3:
                return [[int(p[0] * w), int(p[1] * h)] for p in self.norm_keepout]
            return self.keepout
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
CROSSING_NONE, CROSSING_IN, CROSSING_OUT, CROSSING_BLOCKED = 0, 1, 2, 3

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


def same_vehicle_box(a, b, frame_w):
    """True when two crossing boxes are almost certainly one vehicle detected as
    two boxes, rather than two separate vehicles (a queue). Their centres must
    coincide, and one box must sit almost entirely inside the other — which
    covers both a near-identical duplicate and a tight cab/front box nested in
    the full-truck box (low IoU, so plain IoU-NMS misses it). A genuine follower,
    even one mostly occluded by the leader, has its centre displaced along the
    road. Kept conservative: a missed vehicle is worse than a duplicate."""
    if frame_w <= 0:
        return True   # no scale to judge by: fall back to overlap only
    cxa, cya = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    cxb, cyb = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    if ((cxa - cxb) ** 2 + (cya - cyb) ** 2) ** 0.5 > DEDUP_CENTER_FRAC * frame_w:
        return False
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    # Conservative on purpose: a missed vehicle is worse than a duplicate, so
    # only suppress a box that is *almost entirely* inside the other (>= 0.9 of
    # its area). A partly-visible follower never qualifies.
    return small > 0 and (iw * ih) / small >= 0.9


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

def point_in_poly(poly, point):
    """True when ``point`` lies inside the polygon ``poly`` (list of [x, y])."""
    if not poly or len(poly) < 3 or point is None:
        return False
    pts = np.asarray(poly, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


class CrossingGate:
    """Committed-side line crossing with hysteresis and a confirmed transit.

    ``update()`` is the whole rule set. A track commits to the side of the line
    it is first seen on. A crossing is only *counted* once the centroid lands
    ``confirm_frac`` of the frame height past the line — deeper than the
    hysteresis ``margin_frac`` — inside the drawn segment, on the opposite side
    to the committed one, at least ``min_age`` frames old, and ``rearm_frac`` of
    the frame height away from the point of the previous crossing. Short of
    that the crossing is only *pending*: a return toward the original side
    before it reaches the confirm distance cancels it. So a machine wobbling on
    the line (or oscillating under the confirm distance) never counts, while a
    genuine pass-through — and a genuine later return — does.

    When the centroid is inside ``keepout`` (the working-machine area) no side
    change is committed at all, so a loader/excavator parked on the line can
    never manufacture a crossing.

    ``confirm_frac=0`` reproduces the legacy behaviour (count as soon as the
    margin is cleared).

    Returns CROSSING_NONE, CROSSING_IN, CROSSING_OUT, or CROSSING_BLOCKED — the
    last is a real cross the age/re-arm guard refused, kept only so the caller
    can report the ``rearm_block`` stat.
    """

    def __init__(self, min_age=4, margin_frac=0.06, rearm_frac=0.2,
                 confirm_frac=0.2, seg_tol=0.02, keepout=None,
                 handoff_s=2.0, handoff_frac=0.0, probe_edge=None):
        self.min_age = min_age
        self.margin_frac = margin_frac
        self.rearm_frac = rearm_frac
        self.confirm_frac = confirm_frac   # 0 disables the confirmed transit
        self.seg_tol = seg_tol
        self.keepout = keepout             # pixel polygon [[x,y],...] or None
        # Which point of the box is tested against the line. The centroid is
        # dragged across the line when a vehicle is occluded by another and the
        # detector merges them into one box; the *leading* edge of that merged
        # box still tracks the moving vehicle, so "left"/"right" fix the count
        # for a right->left / left->right flow respectively. None = centroid.
        self.probe_edge = probe_edge
        self.last_pt = {}     # tid -> centroid where the last crossing committed
        self.side = {}        # tid -> last committed side (+1/-1) outside the band
        self.pending = {}     # tid -> side awaiting confirmation (diagnostics)
        # Occlusion hand-off: when a vehicle is split into a new track that is
        # *born already past the line* (its old track was dragged/lost by a
        # merge with another vehicle), inherit the crossing if the opposite side
        # was occupied moments ago. 0 disables it.
        self.handoff_s = handoff_s
        self.handoff_frac = handoff_frac   # spatial gate as a fraction of frame width
        self.recent = {}      # side (+1/-1) -> (t, x, y) last credible sighting

    def _probe(self, obj):
        if self.probe_edge == "left":
            return (float(obj.bbox[0]), (obj.bbox[1] + obj.bbox[3]) / 2.0)
        if self.probe_edge == "right":
            return (float(obj.bbox[2]), (obj.bbox[1] + obj.bbox[3]) / 2.0)
        return obj.centroid

    def drop(self, dead_ids):
        for tid in dead_ids:
            self.last_pt.pop(tid, None)
            self.side.pop(tid, None)
            self.pending.pop(tid, None)

    def update(self, line, obj, flip=False, frame_h=360, now=None, frame_w=None):
        if now is None:
            now = time.time()
        if point_in_poly(self.keepout, obj.centroid):
            # Inside the no-count work area: freeze, so a machine working there
            # can neither commit a crossing nor disturb the committed side.
            return CROSSING_NONE
        pt = self._probe(obj)
        dist, t = line_geometry(line, pt)
        if dist is None:
            return CROSSING_NONE
        margin = self.margin_frac * frame_h
        if abs(dist) < margin:
            # Inside the hysteresis band: not a crossing. Returning to the line
            # cancels any pending crossing (the track came back).
            self.pending.pop(obj.track_id, None)
            return CROSSING_NONE
        new_side = 1 if dist >= 0 else -1
        # Remember the last credible sighting on each side (outside the band),
        # for the occlusion hand-off below.
        self.recent[new_side] = (now, float(pt[0]), float(pt[1]))
        confirm = self.confirm_frac * frame_h if self.confirm_frac > 0 else margin
        prev = self.side.get(obj.track_id)
        if (prev is None and self.handoff_frac > 0 and abs(dist) >= confirm):
            # A *new* track born already at/inside the confirm depth on one side.
            # If the opposite side was occupied within handoff_s and the gap is
            # spatially plausible, this is the same vehicle reappearing across
            # the line after an occluding merge -> hand off the crossing.
            r = self.recent.get(-new_side)
            if r is not None:
                gap = ((pt[0] - r[1]) ** 2 + (pt[1] - r[2]) ** 2) ** 0.5
                gate = self.handoff_frac * (frame_w or 640) + self.rearm_frac * frame_h
                if (now - r[0]) <= self.handoff_s and gap <= gate:
                    self.side[obj.track_id] = new_side
                    self.last_pt[obj.track_id] = (float(pt[0]), float(pt[1]))
                    self.pending.pop(obj.track_id, None)
                    if flip:
                        return CROSSING_OUT if new_side == 1 else CROSSING_IN
                    return CROSSING_IN if new_side == 1 else CROSSING_OUT
        if prev is None or prev == new_side:
            # First sight, or still on the same committed side.
            self.side[obj.track_id] = new_side
            self.pending.pop(obj.track_id, None)
            return CROSSING_NONE
        if t < -self.seg_tol or t > 1 + self.seg_tol:
            # Crossed the infinite line, but outside the drawn segment: take the
            # new side without counting.
            self.side[obj.track_id] = new_side
            self.pending.pop(obj.track_id, None)
            return CROSSING_NONE
        # Candidate crossing: on the opposite side to the committed one. In
        # confirmed mode the centroid must reach confirm_frac past the line
        # before it counts; short of that it only becomes pending, and a return
        # to the original side (or into the band) cancels it.
        confirm = self.confirm_frac * frame_h if self.confirm_frac > 0 else margin
        if abs(dist) < confirm:
            self.pending[obj.track_id] = new_side
            return CROSSING_NONE
        # A genuine cross. Commit only if the track is old enough and has
        # re-armed away from its previous crossing; otherwise leave the side
        # unchanged so the cross can still commit on a later frame.
        if obj.age < self.min_age or self.rearm_remaining(obj, frame_h, pt):
            self.pending[obj.track_id] = new_side
            return CROSSING_BLOCKED
        self.side[obj.track_id] = new_side
        self.pending.pop(obj.track_id, None)
        self.last_pt[obj.track_id] = (float(pt[0]), float(pt[1]))
        if flip:
            return CROSSING_OUT if new_side == 1 else CROSSING_IN
        return CROSSING_IN if new_side == 1 else CROSSING_OUT

    def rearm_remaining(self, obj, frame_h, pt=None):
        """Px the track still has to move from its last crossing before it can
        cross again (0.0 once re-armed; None if it has not crossed yet)."""
        lp = self.last_pt.get(obj.track_id)
        if lp is None:
            return None
        p = pt if pt is not None else obj.centroid
        d = ((p[0] - lp[0]) ** 2 + (p[1] - lp[1]) ** 2) ** 0.5
        return max(0.0, self.rearm_frac * frame_h - d)

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

        When `jpeg` is None (the substream fallback, or `TRACKER_CAPTURE_MAIN_STREAM=0`)
        the substream frame captured at the crossing is the source for both the
        full image `<id>.jpg` and the thumbnail — no separate `_sub` copy, since
        storing the same pixels twice is waste. When a main-stream `jpeg` is
        supplied (the default path), `<id>.jpg` holds that full-size still and
        `<id>_sub.jpg` keeps the substream frame; the thumbnail is ALWAYS built
        from the substream frame, so the timeline shows the vehicle at the line
        rather than the delayed main-stream moment.
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
_VEHICLE_NAME_IDS = {v: k for k, v in _VEHICLE_NAMES.items()}


def apply_vehicle_labels(dets, mapping):
    """Relabel detections by class id (config ``vehicle_labels``).

    On a truck-only site the detector's occasional ``car``/``bus`` is a truck
    mislabel, so map those classes to ``truck`` (e.g. ``{"5": "truck"}``). The
    matching COCO id is applied too, so the stored ``class_id`` and ``label``
    stay consistent. No mapping -> detections are returned unchanged.
    """
    if not mapping:
        return dets
    out = []
    for d in dets:
        lbl = mapping.get(d["class_id"])
        if lbl:
            d = dict(d)
            d["label"] = lbl
            d["class_id"] = _VEHICLE_NAME_IDS.get(lbl, d["class_id"])
        out.append(d)
    return out
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_STD  = np.array([57.375,  57.12,  58.395], dtype=np.float32)
_STRIDES = [8, 16, 32, 64]
_INPUT_SIZE = 416


def _ncnn_io_names(param_path):
    """Return (input_blob, output_blob) for an ncnn .param.

    ncnn exports name the blobs inconsistently (RangiLyu's NanoDet release uses
    ``data``/``output``; our earlier export used ``in0``/``out0``), so read them
    from the graph: input = blob produced by the ``Input`` layer; output = blob
    produced but never consumed.
    """
    produced, consumed, inp = [], set(), None
    try:
        with open(param_path) as f:
            lines = f.read().splitlines()
        for ln in lines[2:]:
            t = ln.split()
            if len(t) < 4:
                continue
            typ, bc, tc = t[0], int(t[2]), int(t[3])
            bottoms = t[4:4 + bc]
            tops = t[4 + bc:4 + bc + tc]
            consumed.update(bottoms)
            produced.extend(tops)
            if typ == "Input" and tops:
                inp = tops[0]
    except (OSError, ValueError, IndexError):
        pass
    outs = [b for b in produced if b not in consumed]
    return inp or "in0", (outs[-1] if outs else "out0")


class NanoDetNcnn:
    def __init__(self, model_param=MODEL_PARAM, model_bin=None, conf_thresh=0.5,
                 iou_thresh=0.45, num_threads=NUM_THREADS, input_size=_INPUT_SIZE):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        # Square network input the priors decode against (320 or 416); must match
        # the ncnn graph's export size.
        self.input_size  = int(input_size)
        # Containment suppression: drop a box that mostly sits inside a
        # higher-confidence one (NanoDet emits "container"/fragment boxes that
        # IoU-NMS misses because their IoU is low).
        self.contain_iou = 0.6
        if model_bin is None:
            model_bin = model_param[:-6] + ".bin"
        self.model_param = model_param
        self.model_bin   = model_bin
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.load_param(model_param)
        self.net.load_model(model_bin)
        self.input_name, self.output_name = _ncnn_io_names(model_param)
        priors = []
        for stride in _STRIDES:
            h = int(np.ceil(self.input_size / stride))
            w = int(np.ceil(self.input_size / stride))
            for i in range(h):
                for j in range(w):
                    priors.append((j * stride, i * stride, stride))
        self.priors = np.array(priors, dtype=np.float32)

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        s = self.input_size
        scale = min(s / h, s / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (s - new_h) // 2
        pad_w = (s - new_w) // 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
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
        # Drop containers: a lower-confidence box mostly inside a kept box.
        order = np.array(keep)[np.argsort(-scores[keep])]
        keep2 = []
        for i in order:
            ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            redundant = False
            for j in keep2:
                aj = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
                iw = max(0.0, min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0]))
                ih = max(0.0, min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]))
                small = min(ai, aj)
                if small > 0 and (iw * ih) / small > self.contain_iou:
                    redundant = True
                    break
            if not redundant:
                keep2.append(i)
        keep = keep2
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


# ── PP-PicoDet (PaddleDetection) ncnn Detector ────────────────────────
# FCOS/GFL head: per level a class-score map (num_class) and a bbox-distance
# distribution (4*(reg_max+1)), decoded by a softmax expectation over the bins.
# Uses Qengineering's ncnn export (`picodet_s_320_coco_lcnet`): input ``image``,
# outputs ``transpose_<2k>.tmp_0`` (cls) and ``transpose_<2k+1>.tmp_0`` (dis),
# for strides 8/16/32/64.
_PICO_MEAN = [103.53, 116.28, 123.675]
_PICO_NORM = [0.017429, 0.017507, 0.017125]
# LibreYOLO's PP-PicoDet export is RGB (channel 0 = R), unlike the Qengineering
# export above which is BGR. Feeding BGR silently shifts the channel stats and
# returns different (wrong) boxes.
_LIBRE_PICO_MEAN = [123.675, 116.28, 103.53]
_LIBRE_PICO_NORM = [1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375]
_PICO_HEADS = [(0, 1, 8), (2, 3, 16), (4, 5, 32), (6, 7, 64)]


class PicodetNcnn:
    def __init__(self, model_param, model_bin=None, conf_thresh=0.5,
                 nms_thresh=0.3, dup_iou=None, nms_agnostic=True,
                 num_threads=NUM_THREADS, input_size=320,
                 reg_max=7, num_class=80):
        if model_bin is None:
            model_bin = model_param[:-6] + ".bin"
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh  = float(nms_thresh)
        # Second, class-agnostic pass: collapse one vehicle's two class-boxes
        # (car+truck) the way Ultralytics' agnostic_nms does. The PicoDet repo
        # does per-class NMS only, so this is our addition to protect the count.
        self.nms_agnostic = bool(nms_agnostic)
        self.dup_iou = DUP_IOU if dup_iou is None else float(dup_iou)
        self.input_size  = int(input_size)
        self.reg_max     = int(reg_max)
        self.num_class   = int(num_class)
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.opt.use_fp16_arithmetic = True   # PicoDet repo: fp16 arithmetic
        self.net.load_param(model_param)
        self.net.load_model(model_bin)
        self.bins = np.arange(self.reg_max + 1, dtype=np.float32)

    def _nms(self, boxes, scores, labels):
        """Per-class NMS at ``nms_thresh`` — the PicoDet repo calls nms() once
        per class (threshold 0.3) and keeps the +1-pixel area convention."""
        if not boxes:
            return []
        b = np.asarray(boxes, dtype=np.float32)
        sc = np.asarray(scores, dtype=np.float32)
        lb = np.asarray(labels)
        keep = []
        for c in np.unique(lb):
            sub = np.nonzero(lb == c)[0]
            bb, ss = b[sub], sc[sub]
            x1, y1, x2, y2 = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = ss.argsort()[::-1]
            while order.size:
                i = order[0]
                keep.append(int(sub[i]))
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                inter = w * h
                iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
                order = order[np.where(iou <= self.nms_thresh)[0] + 1]
        return keep

    def _merge_cross_class(self, boxes, scores, keep):
        """Class-agnostic pass: drop kept boxes overlapping (> dup_iou) a higher
        score box of any class, so one vehicle's car+truck pair collapses."""
        if not self.nms_agnostic or self.dup_iou <= 0 or len(keep) < 2:
            return keep
        out = []
        for k in sorted(keep, key=lambda k: float(scores[k]), reverse=True):
            if all(box_iou(boxes[k], boxes[j]) <= self.dup_iou for j in out):
                out.append(k)
        return out

    def _run(self, frame, enabled_classes, ox, oy):
        h, w = frame.shape[:2]
        s = self.input_size
        img = np.ascontiguousarray(frame)   # ROI slices are non-contiguous views
        mat = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR,
                                          w, h, s, s)
        mat.substract_mean_normalize(_PICO_MEAN, _PICO_NORM)
        ex = self.net.create_extractor()
        ex.input("image", mat)
        boxes, scores, labels = [], [], []
        sx, sy = w / s, h / s
        for ci, di, stride in _PICO_HEADS:
            cls = np.array(ex.extract(f"transpose_{ci}.tmp_0")[1], dtype=np.float32)
            dis = np.array(ex.extract(f"transpose_{di}.tmp_0")[1], dtype=np.float32)
            cls = cls.reshape(-1, self.num_class)
            dis = dis.reshape(-1, 4 * (self.reg_max + 1))
            fw = int(np.ceil(s / stride))
            lab = cls.argmax(axis=1)
            sc = cls[np.arange(cls.shape[0]), lab]
            idx = np.nonzero(sc > self.conf_thresh)[0]
            if not idx.size:
                continue
            d = dis[idx].reshape(-1, 4, self.reg_max + 1)
            e = np.exp(d - d.max(axis=2, keepdims=True))
            sm = e / e.sum(axis=2, keepdims=True)
            dist = (sm * self.bins).sum(axis=2) * stride
            cx = (idx % fw + 0.5) * stride
            cy = (idx // fw + 0.5) * stride
            x1 = np.clip(cx - dist[:, 0], 0, s)
            y1 = np.clip(cy - dist[:, 1], 0, s)
            x2 = np.clip(cx + dist[:, 2], 0, s)
            y2 = np.clip(cy + dist[:, 3], 0, s)
            sc = sc[idx]
            lab = lab[idx]
            for k in range(idx.shape[0]):
                cid = int(lab[k])
                if cid in EXCLUDED_TWO_WHEELER_CLASSES:
                    continue
                if enabled_classes and cid not in enabled_classes:
                    continue
                boxes.append((x1[k] * sx + ox, y1[k] * sy + oy,
                              x2[k] * sx + ox, y2[k] * sy + oy))
                scores.append(float(sc[k]))
                labels.append(cid)
        out = []
        keep = self._nms(boxes, scores, labels)
        keep = self._merge_cross_class(boxes, scores, keep)
        for k in keep:
            bx1, by1, bx2, by2 = (int(boxes[k][0]), int(boxes[k][1]),
                                  int(boxes[k][2]), int(boxes[k][3]))
            cid = labels[k]
            out.append({'bbox': (bx1, by1, bx2, by2),
                        'centroid': ((bx1 + bx2) // 2, (by1 + by2) // 2),
                        'confidence': scores[k], 'class_id': cid,
                        'label': _VEHICLE_NAMES.get(cid, f'cls_{cid}')})
        return out

    def detect(self, frame, enabled_classes=None):
        return self._run(frame, enabled_classes, 0, 0)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        if roi_points is None or len(roi_points) < 3:
            return self.detect(frame, enabled_classes)
        h, w = frame.shape[:2]
        pts = np.array(roi_points, dtype=np.int32)
        cx1, cy1 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        cx2, cy2 = int(min(w, pts[:, 0].max())), int(min(h, pts[:, 1].max()))
        if cx2 <= cx1 or cy2 <= cy1:
            return []
        return self._run(frame[cy1:cy2, cx1:cx2], enabled_classes, cx1, cy1)


class LibrePicoNcnn:
    """PP-PicoDet-S (single class) exported by LibreYOLO to ncnn.

    Selected with ``"detector": "picodet_libre"`` in config.json. The graph is
    input ``in0`` (1x3xSxS) with a single output ``out0`` already decoded to
    ``(anchors, 5)`` = ``[x1, y1, x2, y2, score]`` in the SxS input space, so no
    FCOS/GFL head decode is needed here (unlike ``PicodetNcnn``). Preprocess is a
    plain stretch to SxS with the PicoDet mean/norm.

    It has one class (0 = truck); every box is reported as COCO class 7 (truck)
    so it passes the dashboard's ``enabled_classes`` filter and the UI label is
    right. Use this for a truck-only fine-tune (see truck_ft/). Pair it with
    ``"probe": "left"`` so a vehicle occluded at the line still counts.
    """

    def __init__(self, model_param, model_bin=None, conf_thresh=0.30,
                 nms_thresh=0.6, num_threads=NUM_THREADS, input_size=320):
        if model_bin is None:
            model_bin = model_param[:-6] + ".bin"
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.input_size = int(input_size)
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.opt.use_fp16_arithmetic = True
        self.input_name = "in0"
        self.output_name = "out0"
        self.net.load_param(model_param)
        self.net.load_model(model_bin)

    def _nms(self, boxes, scores):
        order = scores.argsort()[::-1]
        keep = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        while order.size:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= self.nms_thresh)[0] + 1]
        return keep

    def _run(self, frame, ox, oy):
        h, w = frame.shape[:2]
        s = self.input_size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mat = ncnn.Mat.from_pixels_resize(np.ascontiguousarray(rgb),
                                          ncnn.Mat.PixelType.PIXEL_RGB, w, h, s, s)
        mat.substract_mean_normalize(_LIBRE_PICO_MEAN, _LIBRE_PICO_NORM)
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat)
        _, out = ex.extract(self.output_name)
        a = np.array(out)
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 5:
            return []
        scores = a[:, 4]
        mask = scores >= self.conf_thresh
        if not mask.any():
            return []
        b = a[mask, :4].astype(np.float32)
        sc = scores[mask]
        keep = self._nms(b, sc)
        sx, sy = w / s, h / s
        res = []
        for k in keep:
            bx1 = max(0, min(w - 1, int(b[k, 0] * sx))) + ox
            by1 = max(0, min(h - 1, int(b[k, 1] * sy))) + oy
            bx2 = max(0, min(w - 1, int(b[k, 2] * sx))) + ox
            by2 = max(0, min(h - 1, int(b[k, 3] * sy))) + oy
            if bx2 <= bx1 or by2 <= by1:
                continue
            res.append({'bbox': (bx1, by1, bx2, by2),
                        'centroid': ((bx1 + bx2) // 2, (by1 + by2) // 2),
                        'confidence': float(sc[k]), 'class_id': 7, 'label': 'truck'})
        return res

    def detect(self, frame, enabled_classes=None):
        return self._run(frame, 0, 0)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        if roi_points is None or len(roi_points) < 3:
            return self.detect(frame)
        h, w = frame.shape[:2]
        pts = np.array(roi_points, dtype=np.int32)
        cx1, cy1 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        cx2, cy2 = int(min(w, pts[:, 0].max())), int(min(h, pts[:, 1].max()))
        if cx2 <= cx1 or cy2 <= cy1:
            return []
        return self._run(frame[cy1:cy2, cx1:cx2], cx1, cy1)


class CvDnnNanoDet:
    """NanoDet-Plus (OpenCV Zoo) via cv2.dnn on the ONNX graph.

    Selected with ``"detector": "opencv"`` in config.json. It is a CPU-only
    OpenCV-5 DNN path: measured ~12 fps at 416 on the Pi 5, i.e. slower than the
    ncnn YOLO path — kept as an opt-in alternative, not the default.

    The ONNX is a 3-level NanoDet (strides 8/16/32) whose six outputs are grouped
    ``[cls8, cls16, cls32, bbox8, bbox16, bbox32]`` (each cls is (A, nc), each
    bbox is (A, 4*(reg_max+1))). We pair cls/bbox by their anchor count, so the
    decoder is not tied to a particular level count or output order. Input is
    stretched to 416x416 (no letterbox), matching the OpenCV Zoo demo.
    """

    def __init__(self, model_path, conf_thresh=0.3, iou_thresh=0.6,
                 num_threads=NUM_THREADS):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # Containment (intersection-over-min-area) suppression threshold: a box
        # that mostly sits inside a higher-confidence one is dropped. NanoDet
        # emits such "container"/fragment boxes that plain IoU-NMS misses.
        self.contain_iou = 0.6
        self.model_path = model_path
        self.w = self.h = 416
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        # ImageNet BGR mean/std used by the NanoDet OpenCV-Zoo demo.
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        cv2.setNumThreads(int(num_threads))
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.out_names = list(self.net.getUnconnectedOutLayersNames())
        self._acache = {}

    def _anchors(self, n, stride):
        a = self._acache.get(n)
        if a is None:
            f = int(round(n ** 0.5))
            xv, yv = np.meshgrid(np.arange(f) * stride, np.arange(f) * stride)
            cx = xv.flatten() + 0.5 * (stride - 1)
            cy = yv.flatten() + 0.5 * (stride - 1)
            a = np.column_stack((cx, cy)).astype(np.float32)
            self._acache[n] = a
        return a

    def _run_on_roi(self, roi, enabled_classes, offset_x, offset_y):
        h, w = roi.shape[:2]
        r = cv2.resize(roi, (self.w, self.h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        r = (r - self.mean) / self.std
        self.net.setInput(cv2.dnn.blobFromImage(r))
        outs = self.net.forward(self.out_names)
        return self._decode(outs, w / self.w, h / self.h, w, h,
                            enabled_classes, offset_x, offset_y)

    def _decode(self, outs, sx, sy, w, h, enabled_classes, ox, oy):
        bd = 4 * (self.reg_max + 1)
        cls = {o.shape[1]: o for o in outs if o.shape[-1] != bd}
        bbx = {o.shape[1]: o for o in outs if o.shape[-1] == bd}
        boxes, scores, clsids = [], [], []
        for n in sorted(cls, reverse=True):
            if n not in bbx:
                continue
            stride = self.w / (n ** 0.5)
            cs, bp = cls[n], bbx[n]
            if cs.ndim == 3:
                cs = cs[0]
            if bp.ndim == 3:
                bp = bp[0]
            A = cs.shape[0]
            b = bp.reshape(A, 4, self.reg_max + 1)
            e = np.exp(b - b.max(axis=2, keepdims=True))
            soft = e / e.sum(axis=2, keepdims=True)
            dist = (soft * self.project).sum(axis=2) * stride
            ci = cs.argmax(axis=1)
            sc = cs[np.arange(A), ci]
            m = sc >= self.conf_thresh
            m &= ~np.isin(ci, tuple(EXCLUDED_TWO_WHEELER_CLASSES))
            if enabled_classes:
                m &= np.isin(ci, list(enabled_classes))
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            an = self._anchors(n, stride)
            ax, ay, d = an[idx, 0], an[idx, 1], dist[idx]
            x1 = np.clip(ax - d[:, 0], 0, self.w); y1 = np.clip(ay - d[:, 1], 0, self.h)
            x2 = np.clip(ax + d[:, 2], 0, self.w); y2 = np.clip(ay + d[:, 3], 0, self.h)
            valid = (x2 > x1) & (y2 > y1)
            if not valid.any():
                continue
            boxes.append(np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1))
            scores.append(sc[idx][valid]); clsids.append(ci[idx][valid])
        if not boxes:
            return []
        B = np.concatenate(boxes); S = np.concatenate(scores); C = np.concatenate(clsids)
        wh = np.column_stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]])
        idxs = np.array(cv2.dnn.NMSBoxes(
            wh.tolist(), S.tolist(), self.conf_thresh, self.iou_thresh)).flatten()
        # IoU-NMS misses *containment*: NanoDet often emits a big box around a
        # vehicle (or spanning two adjacent trucks) whose IoU with the tight box
        # is low, so both survive and the tracker mints a second track. Drop any
        # box that is mostly inside a higher-confidence one.
        order = idxs[np.argsort(-S[idxs])]
        keep = []
        for i in order:
            area_i = (B[i, 2] - B[i, 0]) * (B[i, 3] - B[i, 1])
            redundant = False
            for j in keep:
                area_j = (B[j, 2] - B[j, 0]) * (B[j, 3] - B[j, 1])
                iw = max(0.0, min(B[i, 2], B[j, 2]) - max(B[i, 0], B[j, 0]))
                ih = max(0.0, min(B[i, 3], B[j, 3]) - max(B[i, 1], B[j, 1]))
                inter = iw * ih
                small = min(area_i, area_j)
                if small > 0 and inter / small > self.contain_iou:
                    redundant = True
                    break
            if not redundant:
                keep.append(i)
        out_list = []
        for i in keep:
            ix1, iy1 = int(max(0, B[i, 0] * sx)), int(max(0, B[i, 1] * sy))
            ix2, iy2 = int(min(w, B[i, 2] * sx)), int(min(h, B[i, 3] * sy))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            cid = int(C[i])
            out_list.append({
                'bbox': (ix1 + ox, iy1 + oy, ix2 + ox, iy2 + oy),
                'centroid': ((ix1 + ix2) // 2 + ox, (iy1 + iy2) // 2 + oy),
                'confidence': float(S[i]), 'class_id': cid,
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
        cx1, cy1 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        cx2, cy2 = int(min(w, pts[:, 0].max())), int(min(h, pts[:, 1].max()))
        if cx2 <= cx1 or cy2 <= cy1: return []
        return self._run_on_roi(frame[cy1:cy2, cx1:cx2], enabled_classes, cx1, cy1)


class YoloNcnn:
    """YOLO11 detector via ncnn. Output out0 is (4+nc, anchors); letterboxed 640."""

    def __init__(self, model_param=YOLO_PARAM, model_bin=None, conf_thresh=0.30,
                 iou_thresh=0.45, num_threads=NUM_THREADS, cross_nms_iou=None,
                 input_wh=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.cross_nms_iou = DUP_IOU if cross_nms_iou is None else cross_nms_iou
        if model_bin is None:
            model_bin = model_param[:-6] + ".bin"
        self.model_param = model_param
        self.model_bin   = model_bin
        # Network input size: from the config-selected model, else the env default.
        self.w, self.h   = (int(input_wh[0]), int(input_wh[1])) if input_wh \
            else (YOLO_W, YOLO_H)
        self.size        = self.w
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
        # per-class NMS above keeps both, the tracker then mints a
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
                 margin=0.08, hold_seconds=None, heartbeat=None, num_threads=None,
                 model_param=YOLO_PARAM, input_wh=None):
        if hold_seconds is None:
            hold_seconds = float(os.environ.get("TRACKER_MOTION_HOLD", "60"))
        if heartbeat is None:
            heartbeat = float(os.environ.get("TRACKER_MOTION_HEARTBEAT", "3"))
        self.motion = MotionDetector(scale=motion_scale, min_area=motion_min_area)
        self.yolo = YoloNcnn(model_param=model_param,
                             conf_thresh=conf_thresh,
                             num_threads=NUM_THREADS if num_threads is None else num_threads,
                             input_wh=input_wh)
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

# ── Velocity Tracker (IoU primary, centroid-distance fallback) ────────
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
    """Velocity-aware tracker: IoU primary, centroid-distance fallback.

    Association runs in two tiers so the common case is precise and the stalled
    case is still recoverable:

    1. **IoU** against the track's *velocity-predicted* box (last box translated
       by px/second velocity over the elapsed time). Frame-to-frame at ~25 fps the
       shift is tiny, so this is a standard IoU tracker; it resists swapping two
       adjacent vehicles.
    2. **Distance fallback**: if no IoU match, a detection whose centroid is within
       the gap-widened gate of the track's predicted centroid still re-acquires it.
       This is what keeps a count across a multi-second stream stall, where the
       predicted box can miss a vehicle that changed speed.

    No appearance model; intentionally simple and suited to sparse scenes.
    """

    def __init__(self, max_age=25, assoc_frac=0.2, vel_smooth=0.6,
                 gap_frac=0.5, gap_max_frac=1.0, vel_reset=0.4, pred_cap=0.5,
                 iou_gate=0.2, debug=False):
        self.tracks = {}          # id -> state
        self.next_id = 1
        self.max_age = max_age    # coasted frames before a track is dropped
        self.assoc_frac = assoc_frac   # fallback gate = frac * frame_w
        self.vel_smooth = vel_smooth
        self.vel_reset = vel_reset      # gap (s) after which velocity is re-measured
        # Fallback gate widens with the gap (gap_frac * w per second), capped at
        # gap_max_frac of the frame diagonal, so a coasted track can re-acquire a
        # vehicle that moved far during a stall.
        self.gap_frac = gap_frac
        self.gap_max_frac = gap_max_frac
        # Cap how far the coasted centroid is extrapolated for display; the IoU
        # box prediction and the fallback anchor use at most this too.
        self.pred_cap = pred_cap
        # Tier-1: minimum IoU with the velocity-predicted box to match
        # (0.2 == ByteTrack's match_thresh 0.8).
        self.iou_gate = iou_gate
        self.debug = debug

    def _info(self, tid, det, prev, cur, age):
        return TrackInfo(tid, det['bbox'], det['label'], det['class_id'],
                         det['confidence'], cur, prev, age)

    def _advance(self, t, dt):
        """Extrapolate a track's centroid by at most `pred_cap` seconds."""
        dt = min(dt, self.pred_cap)
        return (t['cen'][0] + t['vel'][0] * dt,
                t['cen'][1] + t['vel'][1] * dt)

    def _pred_box(self, t, dt):
        """Translate the track's last box by its velocity for IoU association.

        Uses the *full* gap: if the vehicle kept its velocity through a stall the
        predicted box still overlaps it (velocity is px/second); if it didn't, the
        IoU fails and the distance fallback (or coasting) takes over.
        """
        bx1, by1, bx2, by2 = t['det']['bbox']
        vx, vy = t['vel']
        return (bx1 + vx * dt, by1 + vy * dt, bx2 + vx * dt, by2 + vy * dt)

    def update(self, dets, w, h, now=None):
        if now is None:
            now = time.time()
        base = self.assoc_frac * w
        diag = (w * w + h * h) ** 0.5
        max_gate = max(base, self.gap_max_frac * diag)
        # Build candidates in two tiers: (tier 0 = IoU, 1 = distance fallback).
        # Sorting by (tier, cost) makes every IoU match win over any fallback;
        # within a tier the best match comes first.
        cand = []
        for tid, t in self.tracks.items():
            dt = now - t['last_t']
            if dt < 0.0:
                dt = 0.0
            pbox = self._pred_box(t, dt)
            pcen = self._advance(t, dt)
            gate = min(base + self.gap_frac * w * dt, max_gate)
            for i, d in enumerate(dets):
                iou = box_iou(pbox, d['bbox'])
                if iou >= self.iou_gate:
                    cand.append((0, -iou, tid, i, dt))
                cx, cy = d['centroid']
                dist = ((pcen[0] - cx) ** 2 + (pcen[1] - cy) ** 2) ** 0.5
                if dist <= gate:
                    cand.append((1, dist / gate if gate > 0 else 1.0, tid, i, dt))
        cand.sort(key=lambda c: (c[0], c[1]))

        used_t, used_d, pairs = set(), set(), []
        for tier, _cost, tid, i, dt in cand:
            if tid in used_t or i in used_d:
                continue
            if dt >= 1.0 and self.debug:
                how = "IoU" if tier == 0 else "dist"
                print(f"[track] matched #{tid} after {dt:.1f}s gap via {how}")
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

# ── ByteTrack adapter ────────────────────────────────────────────────
class ByteTrackAdapter:
    """Expose the vendored ByteTrack (Kalman + two-stage IoU) behind the same
    ``update(dets, w, h, now=) -> list[TrackInfo]`` interface as CentroidTracker.

    Selected with ``"tracker": "bytetrack"`` in config.json. dets is the same
    list of dicts the detectors emit; we hand ByteTrack an ``(N, 6)`` array of
    ``[x1, y1, x2, y2, score, class_id]``. ByteTrack only *outputs* live tracks,
    so we also coast its ``lost_stracks`` (Kalman-predicted, prev_centroid=None)
    to keep the live overlay and the gate bookkeeping from flickering while the
    detector catches up.
    """

    def __init__(self, track_thresh=0.3, track_buffer=40, match_thresh=0.8,
                 frame_rate=30, debug=False):
        from bytetracker import BYTETracker
        self.bt = BYTETracker(track_thresh=track_thresh, track_buffer=track_buffer,
                              match_thresh=match_thresh, frame_rate=frame_rate)
        self.prev = {}   # tid -> last centroid handed to the gate
        self.age = {}    # tid -> frames seen
        self.debug = debug

    def _info(self, tid, bbox, cls_id, score, cen, prev, age):
        label = _VEHICLE_NAMES.get(cls_id, f"cls_{cls_id}")
        return TrackInfo(tid, bbox, label, cls_id, float(score), cen, prev, age)

    def update(self, dets, w, h, now=None):
        if dets:
            arr = np.array([[float(d['bbox'][0]), float(d['bbox'][1]),
                             float(d['bbox'][2]), float(d['bbox'][3]),
                             float(d['confidence']), float(d['class_id'])]
                            for d in dets], dtype=np.float32)
        else:
            arr = np.empty((0, 6), dtype=np.float32)
        out = self.bt.update(arr, None)

        objects, seen = [], set()
        for row in out:
            x1, y1, x2, y2, tid, cls_id, score = row
            tid, cls_id = int(tid), int(cls_id)
            bbox = (float(x1), float(y1), float(x2), float(y2))
            cen = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            prev = self.prev.get(tid)
            self.prev[tid] = cen
            self.age[tid] = self.age.get(tid, 0) + 1
            objects.append(self._info(tid, bbox, cls_id, score, cen, prev, self.age[tid]))
            seen.add(tid)

        # Coast lost tracks (Kalman prediction) so the overlay does not blink.
        for st in self.bt.lost_stracks:
            tid = st.track_id
            if tid in seen:
                continue
            x1, y1, x2, y2 = (float(v) for v in st.tlbr)
            bbox = (x1, y1, x2, y2)
            cen = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.age[tid] = self.age.get(tid, 0) + 1
            objects.append(self._info(tid, bbox, int(st.cls), st.score, cen, None,
                                      self.age[tid]))

        # Track ids are never reused: drop per-id memory ByteTrack has retired,
        # or it leaks for the life of the process.
        live = ({t.track_id for t in self.bt.tracked_stracks}
                | {t.track_id for t in self.bt.lost_stracks})
        for tid in [t for t in self.prev if t not in live]:
            self.prev.pop(tid, None)
            self.age.pop(tid, None)
        return objects


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
        self.stream_fps = 0.0          # reader's actual delivered fps

    def put(self, frame, fresh=True):
        with self._lock:
            self._frame = frame
            self.fresh = fresh
            self.seq += 1

    def get(self):
        with self._lock:
            return self._frame, self.seq, self.fresh

    def set_stream_fps(self, fps):
        with self._lock:
            self.stream_fps = fps

    def get_stream_fps(self):
        with self._lock:
            return self.stream_fps


def detect_loop(detector, cfg, latest, shared, stop):
    """Runs detection as fast as the CPU allows on the newest frame.

    Decoupled from the tracking loop: tracking/crossing proceeds at the camera's
    full sample rate even while a slow (YOLO) inference is still running, because
    the tracker coasts on velocity prediction between fresh detections.
    """
    last = -1
    while not stop.is_set():
        _beat("detect")
        # The whole body is guarded: an unhandled raise here would silently stop
        # all detection while the process (and every other thread) looks healthy.
        try:
            frame, seq, fresh = latest.get()
            if frame is None or seq == last:
                time.sleep(0.005)
                continue
            last = seq
            if not fresh:                    # NVR repeat; don't burn CPU on it
                continue
            if float(frame.std()) <= 5:      # warm-up / dead frames
                continue
            h, w = frame.shape[:2]
            roi_px = cfg.pixel_roi(w, h)
            # Read the class filter every frame so a remote config applies live.
            classes = None if cfg.detector == "motion" else cfg.enabled_classes
            if roi_px and len(roi_px) >= 3:
                dets = detector.detect_roi(frame, roi_px, enabled_classes=classes)
            else:
                dets = detector.detect(frame, enabled_classes=classes)
            if len(dets) > MAX_DETECTIONS_PER_FRAME:
                _log_throttled("detect_glitch",
                               f"[detect] implausible detections ({len(dets)}) "
                               f"— glitch, ignoring", every=30)
                dets = []
            dets = apply_vehicle_labels(dets, cfg.vehicle_labels)
            with shared["lock"]:
                shared["dets"] = dets
                shared["counter"] += 1
        except Exception as e:
            _log_throttled("detect_err", f"[detect] error: {e}", every=30)
            time.sleep(0.1)
    print("[detect] Done")


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
    prev_gray = None
    last_fresh = time.time()
    last_pos = -1.0
    rate_t = time.time()
    rate_n = 0

    while cfg.running:
        _beat("reader")
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
                last_fresh = time.time()
                if cfg.relay and not pipeline:
                    print(f"[reader] Stream via RTSP relay ({video_fps:.2f} fps)")
                else:
                    print(f"[reader] Stream connected ({video_fps:.2f} fps)")

            # Read unpaced. Pacing to 1/fps made the loop slightly slower than
            # the true ~25 fps source, so the container timestamp fell behind
            # wall-clock a little every second (~3.5 s/min measured) — the reader
            # drifted seconds behind live and the evidence main image (fetched at
            # the delayed crossing) showed the scene after the vehicle had gone.
            # Unpaced, reads block on the source when caught up (lag stays ~0) and
            # drain fast after a reconnect, so the reader sits at the live edge.
            ret, frame = cap.read()
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
            # A genuinely-new frame (or an advancing container timestamp) means
            # the stream is alive. The timestamp matters for a legitimately
            # static, noise-free scene where every frame is bit-identical: that
            # is not a stall, so it must not trip the reconnect watchdog.
            try:
                pos = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            except Exception:
                pos = -1.0
            if fresh or pos > last_pos + 1.0:
                last_fresh = time.time()
            last_pos = pos if pos > last_pos else last_pos
            # Report the true stream rate (frames actually delivered), not the
            # fresh-frame rate: the NVR repeats frames on a static scene, and
            # those are skipped downstream, which made the header fps look broken.
            rate_n += 1
            now_r = time.time()
            if now_r - rate_t >= 2.0:
                latest.set_stream_fps(rate_n / (now_r - rate_t))
                rate_t, rate_n = now_r, 0
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

STATS_PATH = os.environ.get("TRACKER_STATS_PATH", "/tmp/tracker_stats.json")


def _cpu_temp():
    """SoC temperature in °C, or nan if unavailable."""
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except (OSError, ValueError):
        return float("nan")


def _write_stats(payload):
    """Atomic JSON side-channel so the edge's rates/temp are readable remotely
    (the service journal isn't). Best-effort: never disturb the counting loop."""
    try:
        data = dict(payload)
        data["ts"] = datetime.now(timezone.utc).isoformat()
        data["pid"] = os.getpid()
        tmp = STATS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, STATS_PATH)
    except Exception as e:
        _log_throttled("stats", f"[stats] write failed: {e}", every=60)


def _post_stats(cfg, payload):
    """Push the same stats to the VPS so the dashboard header can show them,
    instead of burning text into the live frame."""
    import requests
    try:
        requests.post(cfg.stats_url, json=payload,
                      headers={"Authorization": f"Bearer {cfg.token}"}, timeout=5)
    except Exception as e:
        _log_throttled("stats_post", f"[stats] post failed: {e}", every=60)


def draw_gate_overlay(img, line, objects, gate, frame_h, margin_px, scale=1.0):
    """Debug overlay for the crossing gate.

    Draws the two hysteresis band edges (±``margin_px`` around the line) and,
    per track, the re-arm anchor (the point of its last crossing) plus a circle
    of radius ``rearm_frac * frame_h``. The dot at the centroid is green when the
    track is re-armed (eligible to count) and amber while it still has to move
    away from the anchor before it can count again.
    """
    if line:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / L, dx / L
        for sgn in (1, -1):
            a = (int((x1 + nx * margin_px * sgn) * scale), int((y1 + ny * margin_px * sgn) * scale))
            b = (int((x2 + nx * margin_px * sgn) * scale), int((y2 + ny * margin_px * sgn) * scale))
            cv2.line(img, a, b, (170, 170, 170), 1, cv2.LINE_AA)
    if gate is None:
        return img
    for obj in objects:
        remaining = gate.rearm_remaining(obj, frame_h)
        armed = not remaining            # None (never crossed) or 0.0 => armed
        col = (34, 197, 94) if armed else (0, 165, 255)
        lp = gate.last_pt.get(obj.track_id)
        if lp is not None:
            ax, ay = int(lp[0] * scale), int(lp[1] * scale)
            r = max(2, int(gate.rearm_frac * frame_h * scale))
            # The re-arm rule is radial (distance from the crossing point), so
            # draw that true radius as a circle, not a skewed parallelogram.
            cv2.circle(img, (ax, ay), r, col, 1, cv2.LINE_AA)
            cv2.circle(img, (ax, ay), 3, col, -1, cv2.LINE_AA)
        cx, cy = int(obj.centroid[0] * scale), int(obj.centroid[1] * scale)
        cv2.circle(img, (cx, cy), 3, col, -1, cv2.LINE_AA)
        cv2.putText(img, "ARM" if armed else f"rearm {remaining:.0f}",
                    (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return img


def annotate_live(frame, line, objects, counts, target_h=LIVE_HEIGHT, roi=None, flip=False,
                  gate=None, margin_px=0.0, keepout=None):
    """Downscale to the target height and draw a crisp overlay (boxes scaled to match)."""
    h, w = frame.shape[:2]
    s = target_h / h if h > target_h else 1.0
    img = (cv2.resize(frame, (max(1, int(w * s)), target_h), interpolation=cv2.INTER_AREA)
           if s < 1.0 else frame.copy())
    if keepout is not None and len(keepout) >= 3:
        pts = np.array([[int(p[0] * s), int(p[1] * s)] for p in keepout], dtype=np.int32)
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, (60, 60, 220), 1, cv2.LINE_AA)
    if roi is not None and len(roi) >= 3:
        # Very faint so it reads as a hint, not part of the scene: draw on a copy
        # and alpha-blend back (OpenCV polylines has no alpha).
        pts = np.array([[int(p[0] * s), int(p[1] * s)] for p in roi], dtype=np.int32)
        overlay = img.copy()
        cv2.polylines(overlay, [pts.reshape((-1, 1, 2))], True, (100, 180, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
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
    if gate is not None:
        draw_gate_overlay(img, line, objects, gate, h, margin_px, s)
    return img

def live_loop(url, token, live_q, cfg):
    """POSTs annotated JPEG frames to the VPS live endpoint (outbound only)."""
    import requests
    headers = {"Authorization": f"Bearer {token}"}
    while cfg.running:
        _beat("live")
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
        _beat("cam2")
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

def _fetch_snapshot_jpeg(cfg, timeout=6.0):
    """GET the camera's main-profile ONVIF snapshot (no Pi video decode).

    Returns bytes, or None on any failure. The camera hardware-encodes the JPEG,
    but its still pipeline lags the live stream by ~2 s on the NVR, so this is
    now only a fallback when a live RTSP grab is unavailable.

    A persistent Session keeps the TLS connection and the digest nonce alive
    between calls.
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


def fetch_main_rtsp_jpeg(cfg, timeout=6.0, quality=85):
    """Decode a frame off the main RTSP stream and JPEG-encode it.

    Optional alternative to the ONVIF snapshot. Note a fresh RTSP connect starts
    at the NVR's last I-frame (main GOP=50 ≈ 2 s) and the open costs ~1.4 s, so
    it is no fresher than the snapshot and writes a bigger file; kept for
    cameras whose snapshot endpoint is unavailable or heavily cached. A
    connection is opened per crossing, a frame is read, then it is closed.
    Best-effort: returns None on any failure (the caller falls back to the
    snapshot, then the substream frame).

    Skipped when the substream needs the SHA-256 relay (`cfg.relay`); the
    snapshot fallback covers those cameras.
    """
    url = cfg.capture_rtsp
    if not url or cfg.relay:
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return None
        deadline = time.time() + timeout
        frame = None
        while time.time() < deadline:
            ok, f = cap.read()
            if ok and f is not None and getattr(f, "size", 0):
                frame = f
                break
        if frame is None:
            return None
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpg.tobytes() if ok else None
    except Exception as e:
        _log_throttled("capture_rtsp", f"[capture] main RTSP grab failed: {e}", every=60)
        return None
    finally:
        if cap is not None:
            cap.release()


def fetch_capture_jpeg(cfg, timeout=6.0):
    """Return a full-resolution main-profile JPEG, or None for the substream.

    Mode `cfg.capture_main_mode`:
      "rtsp" (default) — live main RTSP grab, falling back to the snapshot
      "1"/"snap"       — the ONVIF main-profile snapshot
      "0"/"sub"        — None (caller saves the substream crossing frame)
    """
    mode = cfg.capture_main_mode
    if mode in ("0", "sub", "off", "false", "no"):
        return None
    if mode in ("rtsp", "live"):
        jpeg = fetch_main_rtsp_jpeg(cfg, timeout=timeout)
        if jpeg is not None:
            return jpeg
    return _fetch_snapshot_jpeg(cfg, timeout=timeout)


_CAPTURE_SESSION = None

# How long a queued crossing may wait for the main-stream snapshot before we
# stop trusting the fetch and fall back to the (accurately-timed) substream
# frame for the full image as well.
CAPTURE_MAX_AGE = float(os.environ.get("TRACKER_CAPTURE_MAX_AGE", "1.5"))

# sync.py wakes early when this file's mtime changes, so a just-persisted
# crossing is uploaded within ~a second instead of the next period sync pass.
SYNC_NUDGE = os.environ.get("TRACKER_SYNC_NUDGE", "/tmp/tracker_sync.nudge")


def _nudge_sync():
    """Wake sync.py so a fresh crossing uploads without waiting a full interval.

    sync.py watches this file's mtime and cuts its sleep short on a change; a
    failure here is harmless (the periodic pass still carries the event).
    """
    try:
        with open(SYNC_NUDGE, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


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
    if _safe_store_add(store, {
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
    }):
        _nudge_sync()
    return entry


def capture_loop(cfg, capture_mgr, store, capture_q):
    """Persist crossings on a worker thread (fetch/save off the track loop).

    By default the full image is the camera's main-profile ONVIF snapshot and the
    thumbnail is always built from the substream frame captured at the crossing,
    so the timeline shows the vehicle at the line. It falls back to the substream
    frame for the full image too when the snapshot is stale (the worker was busy
    with earlier crossings) or unavailable — an accurately-timed low-res frame
    beats a crisp frame of the wrong moment. With `TRACKER_CAPTURE_MAIN_STREAM=0`
    the substream frame is saved as the full image and the camera's main encoder
    is never exercised; `=rtsp` decodes a main-stream frame instead of the
    snapshot.
    """
    # Drain on shutdown too: `cfg.running` goes False before the main loop
    # exits, but anything already counted and queued must still be persisted so
    # the DB can never fall behind the counts.
    while cfg.running or not capture_q.empty():
        _beat("capture")
        try:
            item = capture_q.get(timeout=1.0)
        except Empty:
            continue
        try:
            age = time.time() - item.get("enqueued_at", time.time())
            # During shutdown, skip the network still (it only delays the drain)
            # and persist from the substream frame we already hold.
            jpeg = (fetch_capture_jpeg(cfg)
                    if cfg.running and age <= CAPTURE_MAX_AGE else None)
            if jpeg is not None:
                print(f"[capture] ID#{item['track_id']} main-stream ({len(jpeg) // 1024}KB)")
            elif cfg.capture_main_mode in ("0", "sub", "off", "false", "no"):
                print(f"[capture] ID#{item['track_id']} substream (main-stream evidence off)")
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
        _beat("config")
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
        _beat("prune")
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

    # Model selected by config.json (falling back to the tracker.env defaults),
    # so the model can be A/B'd on-device without editing the root-owned env.
    model_dir   = MODEL_DIR / f"{cfg.model}_ncnn_{cfg.model_imgsz}"
    model_param = str(model_dir / ("model_int8.ncnn.param" if cfg.int8 else "model.ncnn.param"))
    model_wh    = _parse_imgsz(cfg.model_imgsz)
    model_label = f"{cfg.model}_ncnn_{cfg.model_imgsz}"

    if cfg.detector == "motion":
        detector    = MotionDetector(scale=cfg.motion_scale, min_area=cfg.motion_min_area)
        print(f"[main] Motion detector (scale={cfg.motion_scale}, min_area={cfg.motion_min_area})")
    elif cfg.detector == "yolo":
        detector    = YoloNcnn(model_param=model_param, conf_thresh=cfg.conf_thresh,
                               num_threads=cfg.threads, input_wh=model_wh)
        print(f"[main] YOLO11 ncnn loaded ({model_label}, {cfg.threads} threads)")
    elif cfg.detector == "opencv":
        detector    = CvDnnNanoDet(model_path=cfg.dnn_model,
                                   conf_thresh=cfg.conf_thresh, num_threads=cfg.threads)
        print(f"[main] OpenCV DNN NanoDet-Plus ({cfg.dnn_model}, {cfg.threads} threads)")
    elif cfg.detector == "nanodet":
        detector    = NanoDetNcnn(model_param=cfg.dnn_model,
                                   conf_thresh=cfg.conf_thresh, num_threads=cfg.threads,
                                   input_size=_parse_imgsz(cfg.dnn_imgsz)[0])
        print(f"[main] NanoDet-Plus ncnn ({cfg.dnn_model}, {cfg.dnn_imgsz}, {cfg.threads} threads)")
    elif cfg.detector == "picodet":
        detector    = PicodetNcnn(model_param=cfg.dnn_model,
                                   conf_thresh=cfg.dnn_conf, nms_thresh=cfg.dnn_iou,
                                   num_threads=cfg.threads,
                                   input_size=_parse_imgsz(cfg.dnn_imgsz)[0])
        print(f"[main] PP-PicoDet ncnn ({cfg.dnn_model}, {cfg.dnn_imgsz}, "
              f"score={cfg.dnn_conf}, nms_iou={cfg.dnn_iou}, {cfg.threads} threads)")
    elif cfg.detector in ("picodet_libre", "libre"):
        detector    = LibrePicoNcnn(model_param=cfg.dnn_model,
                                    conf_thresh=cfg.dnn_conf, nms_thresh=0.6,
                                    num_threads=cfg.threads,
                                    input_size=_parse_imgsz(cfg.dnn_imgsz)[0])
        print(f"[main] PicoDet (LibreYOLO, single-class) ncnn ({cfg.dnn_model}, "
              f"{cfg.dnn_imgsz}, score={cfg.dnn_conf}, {cfg.threads} threads)")
    else:
        detector    = HybridDetector(conf_thresh=cfg.conf_thresh,
                                     motion_scale=cfg.motion_scale,
                                     motion_min_area=cfg.motion_min_area,
                                     hold_seconds=cfg.motion_hold,
                                     heartbeat=cfg.motion_heartbeat,
                                     num_threads=cfg.threads,
                                     model_param=model_param, input_wh=model_wh)
        print(f"[main] Hybrid detector: motion-gated YOLO11 "
              f"({model_label}, {cfg.threads} threads)")

    # IoU association against each track's velocity-predicted box: frame-to-frame
    # at ~25 fps this is a standard IoU tracker; across a stream stall the box is
    # extrapolated forward so a constant-velocity detection still overlaps.
    if cfg.tracker == "bytetrack":
        tracker = ByteTrackAdapter(
            track_thresh=float(os.environ.get("TRACKER_BT_TRACK_THRESH", "0.3")),
            track_buffer=int(os.environ.get("TRACKER_TRACK_MAX_AGE", "40")),
            match_thresh=float(os.environ.get("TRACKER_BT_MATCH_THRESH", "0.8")),
            frame_rate=30,
            debug=os.environ.get("TRACKER_DEBUG") == "1")
        print("[main] Tracker: ByteTrack (Kalman + two-stage IoU)")
    else:
        tracker = CentroidTracker(
            max_age=int(os.environ.get("TRACKER_TRACK_MAX_AGE", "40")),
            iou_gate=float(os.environ.get("TRACKER_IOU_GATE", "0.2")),
            assoc_frac=float(os.environ.get("TRACKER_ASSOC_FRAC", "0.2")),
            gap_frac=float(os.environ.get("TRACKER_ASSOC_GAP_FRAC", "0.5")),
            gap_max_frac=float(os.environ.get("TRACKER_ASSOC_MAX_FRAC", "1.0")),
            vel_reset=float(os.environ.get("TRACKER_VEL_RESET", "0.4")),
            pred_cap=float(os.environ.get("TRACKER_ASSOC_PRED_CAP", "0.5")),
            debug=os.environ.get("TRACKER_DEBUG") == "1")
        print("[main] Tracker: CentroidTracker (IoU + distance fallback)")
    capture_mgr = CaptureManager(cfg.capture_dir, cfg.max_captures)

    # Offline-first event store (SQLite); survives restarts and outages
    store = Store(os.environ.get("TRACKER_DB_PATH", "events.db"))

    # Recent crossings, to collapse one vehicle counted twice (two tracks from
    # two class-boxes) into a single event.
    recent_cross    = []
    gate_ids        = set()
    last_stats_str  = ""
    cross_gate      = CrossingGate(min_age=cfg.min_track_age,
                                   margin_frac=cfg.cross_margin_frac,
                                   rearm_frac=cfg.cross_rearm_frac,
                                   confirm_frac=cfg.cross_confirm_frac,
                                   probe_edge=None if cfg.cross_probe in ("", "centroid") else cfg.cross_probe)

    latest = LatestFrame()
    shared = {"dets": [], "counter": 0, "lock": threading.Lock()}
    stop = threading.Event()

    # Seed each heartbeat at start so a worker that dies before its first beat
    # is still caught by the supervisor after its budget.
    _HEARTBEAT["reader"] = time.time()
    threading.Thread(target=reader_loop, args=(cfg, latest), daemon=True).start()
    _HEARTBEAT["detect"] = time.time()
    threading.Thread(target=detect_loop,
                     args=(detector, cfg, latest, shared, stop), daemon=True).start()
    print("[main] Reader + detect threads started")

    # Evidence: a worker persists each crossing from the substream frame (and an
    # optional main-stream still) so any network fetch never stalls tracking.
    capture_q = Queue(maxsize=64)
    _HEARTBEAT["capture"] = time.time()
    capture_thread = threading.Thread(
        target=capture_loop, args=(cfg, capture_mgr, store, capture_q), daemon=True)
    capture_thread.start()
    print("[main] Capture worker started")

    # Live relay to the VPS dashboard (optional; outbound POST)
    live_url   = os.environ.get("TRACKER_LIVE_URL", "")
    live_token = os.environ.get("TRACKER_TOKEN", "")
    live_q     = Queue(maxsize=1)
    cfg.stats_url = os.environ.get("TRACKER_STATS_URL") or (
        live_url.replace("/api/live", "/api/stats")
        if live_url.endswith("/api/live") else "")
    if live_url:
        _HEARTBEAT["live"] = time.time()
        threading.Thread(target=live_loop, args=(live_url, live_token, live_q, cfg),
                         daemon=True).start()
        print(f"[main] Live relay -> {live_url}")

    # Second camera (raw substream) — only streams while someone watches /live.
    live2_q = Queue(maxsize=1)
    if cfg.cam2_url and cfg.live2_url:
        _HEARTBEAT["live"] = time.time()
        threading.Thread(target=live_loop, args=(cfg.live2_url, live_token, live2_q, cfg),
                         daemon=True).start()
        _HEARTBEAT["cam2"] = time.time()
        threading.Thread(target=cam2_loop, args=(cfg, live2_q, stop), daemon=True).start()
        print(f"[main] Camera 2 live -> {cfg.live2_url}")

    # Pull line / scan-area config from the server (Pi is outbound-only)
    if cfg.config_url:
        _HEARTBEAT["config"] = time.time()
        threading.Thread(target=config_loop, args=(cfg,), daemon=True).start()
        print(f"[main] Config poll -> {cfg.config_url}")

    # Auto-expire local events/captures older than the retention window
    retention_days = int(os.environ.get("TRACKER_RETENTION_DAYS", "30"))
    _HEARTBEAT["prune"] = time.time()
    threading.Thread(target=prune_loop, args=(store, cfg.capture_dir, retention_days),
                     daemon=True).start()
    print(f"[main] Retention: {retention_days} days")

    # Worker watchdog: any counting worker that wedges triggers a draining exit
    # and systemd restarts us (a raise is already handled inside each loop).
    threading.Thread(target=supervisor_loop, args=(cfg, stop), daemon=True).start()
    print("[main] Supervisor started")

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
    # Session counters for the crossing gate's suppressors, so on-device stats
    # (the journal isn't readable) show whether the re-arm/dedup is blocking
    # would-be crossings — i.e. whether the scene-specific re-arm is too wide.
    n_rearm_block = 0
    n_dedup_block = 0
    objects = []          # carried across stalled (repeated) frames for the live feed
    prev_frame_t = 0.0

    # systemd stops/starts us with SIGTERM; handle it (and SIGINT) so the loop
    # exits through the finally below and drains the capture queue instead of
    # dropping already-counted crossings on every restart.
    def _shutdown(signum, _frame):
        print(f"[main] signal {signum}: graceful shutdown", flush=True)
        cfg.running = False
        stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

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
            # Refresh the gate's tunables/polygon each frame so a config change
            # (or the frame size) is picked up without a restart.
            cross_gate.keepout = cfg.pixel_keepout(w, h)
            cross_gate.confirm_frac = cfg.cross_confirm_frac
            cross_gate.rearm_frac = cfg.cross_rearm_frac

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

                # Update the IoU tracker with fresh detections (empty list
                # just coasts tracks between detector updates).
                objects = tracker.update(raw_detections, w, h, now=now_t)

                # Track ids are never reused, so drop per-id gate state for
                # tracks the tracker just retired — otherwise the gate's
                # per-track dicts grow without bound for the life of the process
                # (a slow leak over months).
                present = {obj.track_id for obj in objects}
                dead = gate_ids - present
                cross_gate.drop(dead)
                gate_ids = present

            # Push annotated frame to the VPS live feed (~5 fps, non-blocking).
            # Keep doing this even on a repeated frame so a static scene does not
            # time out the dashboard (it only relays, it does not track). The
            # whole block is best-effort: an annotation/encode failure must never
            # take down the counting loop (live is a view, not a source of truth).
            if live_url and cfg.live_wanted and now_t - last_live >= 0.2:
                last_live = now_t
                try:
                    ann = annotate_live(frame, line_px, objects, {"in": c_in, "out": c_out},
                                        roi=roi_px, flip=cfg.flip_sides,
                                        gate=(cross_gate if cfg.debug_overlay else None),
                                        margin_px=cfg.cross_margin_frac * h,
                                        keepout=cross_gate.keepout)
                    ok, jpeg = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, LIVE_QUALITY])
                    if ok:
                        try: live_q.put_nowait(jpeg.tobytes())
                        except Full: pass
                except Exception as e:
                    _log_throttled("live_ann", f"[live] annotation failed: {e}", every=60)

            # Periodic rate report (tracking is decoupled from detection).
            if time.time() - stat_t >= 10:
                el = time.time() - stat_t
                with shared["lock"]:
                    dc = shared["counter"]
                track_fps = (total_frames - stat_frames) / el
                detect_fps = (dc - stat_dets) / el
                temp_c = _cpu_temp()
                print(f"[stat] track={track_fps:.1f} fps  detect={detect_fps:.1f} fps  "
                      f"tracks={len(objects)}  temp={temp_c:.1f}C  "
                      f"rearm_block={n_rearm_block} dedup_block={n_dedup_block} "
                      f"pending={len(cross_gate.pending)}")
                payload = {
                    "track_fps": round(float(track_fps), 1),
                    "detect_fps": round(float(detect_fps), 1),
                    "stream_fps": round(float(latest.get_stream_fps()), 1),
                    "tracks": int(len(objects)),
                    "detector": type(detector).__name__,
                    "tracker": type(tracker).__name__,
                    "dets": int(len(shared.get("dets") or [])),
                    "temp_c": None if temp_c != temp_c else round(float(temp_c), 1),
                    "in": int(c_in), "out": int(c_out),
                    "rearm_block": int(n_rearm_block), "dedup_block": int(n_dedup_block),
                    "pending": int(len(cross_gate.pending)),
                }
                _write_stats(payload)
                if cfg.stats_url:
                    threading.Thread(target=_post_stats, args=(cfg, payload),
                                     daemon=True).start()
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
                crossing = cross_gate.update(line_px, obj, flip=cfg.flip_sides,
                                             frame_h=h, now=now_t, frame_w=w)
                if debug:
                    print(f"[dbg] #{obj.track_id} {obj.label} age={obj.age} c={obj.centroid} "
                          f"side={cross_gate.side.get(obj.track_id)} cross={crossing}")
                if crossing == CROSSING_NONE:
                    continue
                if crossing == CROSSING_BLOCKED:
                    # Real cross, refused by the age/re-arm guard (side unchanged).
                    n_rearm_block += 1
                    continue
                direction = 'IN' if crossing == CROSSING_IN else 'OUT'
                # Safety net: if another track crossed the same way here moments
                # ago (the duplicate track from a second box), count once.
                if CROSS_DEDUP_S > 0:
                    recent_cross = [rc for rc in recent_cross
                                    if now_t - rc[0] <= CROSS_DEDUP_S]
                    if any(rc[1] == crossing
                           and same_vehicle_box(obj.bbox, rc[2], w)
                           for rc in recent_cross):
                        if debug:
                            print(f"[dbg] #{obj.track_id} duplicate crossing "
                                  f"{direction} suppressed")
                        n_dedup_block += 1
                        continue
                    recent_cross.append((now_t, crossing, tuple(obj.bbox)))
                # Hand the capture to a worker: it persists the crossing (the
                # main-stream still, or the substream frame on fallback) without
                # stalling the track loop.
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
        stop.set()
        # Let the capture worker persist everything already counted and queued
        # before the process ends (counts==DB across restarts/updates).
        try:
            capture_thread.join(timeout=30)
        except Exception:
            pass
        print("[main] Done")

if __name__ == "__main__":
    main()
