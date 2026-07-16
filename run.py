import ctypes as _ctypes
for _lib in ['libcudart.so.13', 'libcublas.so.13', 'libcublasLt.so.13']:
    _ctypes.CDLL(f'/usr/local/lib/ollama/cuda_v13/{_lib}', mode=_ctypes.RTLD_GLOBAL)

import sys as _sys
_sys.path = [p for p in _sys.path if 'robostack' not in p and 'tyno_ws' not in p and 'python3.12' not in p]

import json, threading, time, os
from collections import deque
from datetime import datetime
from queue import Queue, Empty, Full

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, stream_with_context
import onnxruntime

# ── Config ────────────────────────────────────────────────────────────
CONFIG_PATH = "config.json"

class Config:
    def __init__(self):
        self.lock = threading.RLock()
        self.stream_url = ""
        self.line = None
        self.roi = None
        self.conf_thresh = 0.5
        self.flip_sides = False
        self.capture_dir = "captures"
        self.max_captures = 1000
        self.enabled_classes = [2, 3, 5, 7]
        self.running = True
        self._reconnect = False
        self.viewers = 0
        self.frame_buffer = deque(maxlen=1)
        self.frame_seq = 0          # FIX: sequence number for stream dedup
        self.captures = []
        self._load()

    def _load(self):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            with self.lock:
                self.stream_url = data.get("stream_url", "")
                self.line = data.get("line")
                self.roi = data.get("roi")
                self.conf_thresh = data.get("conf_thresh", 0.5)
                self.flip_sides = data.get("flip_sides", False)
                self.capture_dir = data.get("capture_dir", "captures")
                self.max_captures = data.get("max_captures", 1000)
                self.enabled_classes = data.get("enabled_classes", [2, 3, 5, 7])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self):
        with self.lock:
            data = {"stream_url": self.stream_url, "line": self.line, "roi": self.roi,
                    "conf_thresh": self.conf_thresh, "flip_sides": self.flip_sides,
                    "capture_dir": self.capture_dir, "max_captures": self.max_captures,
                    "enabled_classes": self.enabled_classes}
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
        with self.lock: self.enabled_classes = [int(c) for c in classes]
        self.save()
    def reset_counts(self):
        with self.lock: self.captures.clear()
        self.save()
    def has_viewers(self):
        with self.lock: return self.viewers > 0
    def viewer_connected(self):
        with self.lock: self.viewers += 1
    def viewer_disconnected(self):
        with self.lock:
            if self.viewers > 0: self.viewers -= 1
    def push_jpeg(self, jpeg_bytes):
        with self.lock:
            self.frame_buffer.append(jpeg_bytes)
            self.frame_seq += 1         # FIX: bump seq so generator knows it's new
    def pop_jpeg(self):
        with self.lock:
            if self.frame_buffer:
                return self.frame_seq, self.frame_buffer[-1]
            return None, None
    def add_capture(self, entry):       # FIX 3 (carried over): thread-safe insert
        with self.lock:
            self.captures.insert(0, entry)
            if len(self.captures) > self.max_captures:
                self.captures.pop()

cfg = Config()

# ── Line crossing ─────────────────────────────────────────────────────
CROSSING_NONE, CROSSING_IN, CROSSING_OUT = 0, 1, 2

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

# ── Capture Manager ───────────────────────────────────────────────────
class CaptureManager:
    def __init__(self, capture_dir="captures", max_captures=1000):
        self.capture_dir = capture_dir
        self.max_captures = max_captures
        os.makedirs(capture_dir, exist_ok=True)
        os.makedirs(os.path.join(capture_dir, "thumb"), exist_ok=True)

    def save(self, frame, track_id, direction, bbox):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        direction_label = "in" if direction == 1 else "out"
        filename = f"{ts}_id{track_id}_{direction_label}.jpg"
        cv2.imwrite(os.path.join(self.capture_dir, filename), frame)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop_size = min(max(x2 - x1, y2 - y1) * 2, w, h)
        crop_x1 = max(0, cx - crop_size // 2)
        crop_y1 = max(0, cy - crop_size // 2)
        thumb = frame[crop_y1:min(h, crop_y1 + crop_size), crop_x1:min(w, crop_x1 + crop_size)]
        if thumb.size > 0:
            cv2.imwrite(os.path.join(self.capture_dir, "thumb", filename), cv2.resize(thumb, (256, 256)))
        return {"filename": filename, "thumb": f"thumb/{filename}", "timestamp": ts,
                "track_id": track_id, "direction": direction_label}

# ── NanoDet ONNX Detector ─────────────────────────────────────────────
_VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_STD  = np.array([57.375,  57.12,  58.395], dtype=np.float32)
_STRIDES = [8, 16, 32, 64]
_INPUT_SIZE = 416

class NanoDetONNX:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.session = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
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
        out = self.session.run(None, {self.input_name: blob})[0][0]
        cls_scores = out[:, :80]
        reg_dists  = out[:, 80:]
        candidates = []
        for i, (cx, cy, stride) in enumerate(self.priors):
            scores = cls_scores[i]
            cls_id = int(np.argmax(scores))
            score  = float(scores[cls_id])
            if score < self.conf_thresh: continue
            if enabled_classes and cls_id not in enabled_classes: continue
            dist = reg_dists[i].reshape(4, 8)
            dist_softmax = self._softmax(dist, axis=1)
            dist_val = np.sum(dist_softmax * np.arange(8), axis=1) * stride
            x1 = max(0, min(int((cx - dist_val[0] - pad_w) / scale), w - 1))
            y1 = max(0, min(int((cy - dist_val[1] - pad_h) / scale), h - 1))
            x2 = max(0, min(int((cx + dist_val[2] - pad_w) / scale), w - 1))
            y2 = max(0, min(int((cy + dist_val[3] - pad_h) / scale), h - 1))
            if x2 <= x1 or y2 <= y1: continue
            candidates.append({'bbox': (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                               'centroid': ((x1 + x2) // 2 + offset_x, (y1 + y2) // 2 + offset_y),
                               'confidence': score, 'class_id': cls_id,
                               'label': _VEHICLE_NAMES.get(cls_id, f'cls_{cls_id}')})
        if not candidates: return []
        boxes      = np.array([c['bbox']       for c in candidates])
        scores_arr = np.array([c['confidence'] for c in candidates])
        keep = self._nms(boxes, scores_arr)
        return [candidates[i] for i in keep]

    def detect(self, frame, enabled_classes=None):
        return self._run_on_roi(frame, enabled_classes, 0, 0)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        if roi_points is None or len(roi_points) < 3:
            return self.detect(frame, enabled_classes)
        h, w = frame.shape[:2]
        pts = np.array(roi_points, dtype=np.int32)
        cx1, cy1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
        cx2, cy2 = min(w, pts[:, 0].max()), min(h, pts[:, 1].max())
        if cx2 <= cx1 or cy2 <= cy1: return []
        return self._run_on_roi(frame[cy1:cy2, cx1:cx2], enabled_classes, cx1, cy1)


# ── YOLO26n ONNX Detector ──────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.session = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 320

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, pad_w, pad_h

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
        out = self.session.run(None, {self.input_name: blob})[0][0]
        candidates = []
        for det in out:
            x1, y1, x2, y2, score, cls_id = det
            cls_id = int(cls_id)
            if score < self.conf_thresh: continue
            if enabled_classes and cls_id not in enabled_classes: continue
            # Undo letterbox
            x1 = max(0, min(int((x1 - pad_w) / scale), w - 1))
            y1 = max(0, min(int((y1 - pad_h) / scale), h - 1))
            x2 = max(0, min(int((x2 - pad_w) / scale), w - 1))
            y2 = max(0, min(int((y2 - pad_h) / scale), h - 1))
            if x2 <= x1 or y2 <= y1: continue
            candidates.append({'bbox': (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                               'centroid': ((x1 + x2) // 2 + offset_x, (y1 + y2) // 2 + offset_y),
                               'confidence': float(score), 'class_id': cls_id,
                               'label': _VEHICLE_NAMES.get(cls_id, f'cls_{cls_id}')})
        if not candidates: return []
        boxes      = np.array([c['bbox']       for c in candidates])
        scores_arr = np.array([c['confidence'] for c in candidates])
        keep = self._nms(boxes, scores_arr)
        return [candidates[i] for i in keep]

    def detect(self, frame, enabled_classes=None):
        return self._run_on_roi(frame, enabled_classes, 0, 0)

    def detect_roi(self, frame, roi_points, enabled_classes=None):
        if roi_points is None or len(roi_points) < 3:
            return self.detect(frame, enabled_classes)
        h, w = frame.shape[:2]
        pts = np.array(roi_points, dtype=np.int32)
        cx1, cy1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
        cx2, cy2 = min(w, pts[:, 0].max()), min(h, pts[:, 1].max())
        if cx2 <= cx1 or cy2 <= cy1: return []
        return self._run_on_roi(frame[cy1:cy2, cx1:cx2], enabled_classes, cx1, cy1)

def bbox_touches_line(line, bbox):
    """Check if a bbox overlaps the line's bounding rectangle in both axes."""
    x1, y1, x2, y2 = line
    bx1, by1, bx2, by2 = bbox
    return (bx1 <= max(x1, x2) and bx2 >= min(x1, x2) and
            by1 <= max(y1, y2) and by2 >= min(y1, y2))

# ── ByteTrack Tracker (bytetracker library via Kalman filter + Hungarian) ──
from bytetracker import BYTETracker

class TrackInfo:
    """Lightweight tracked object wrapper for crossing detection & annotation."""
    __slots__ = ('track_id', 'bbox', 'label', 'confidence', 'centroid',
                 'prev_centroid', 'age', 'last_crossing_frame', 'last_crossing')
    def __init__(self, track_id, bbox, label, confidence, centroid, prev_centroid, age):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.confidence = confidence
        self.centroid = centroid
        self.prev_centroid = prev_centroid
        self.age = age
        self.last_crossing_frame = -60
        self.last_crossing = None

# ── Web Server ────────────────────────────────────────────────────────
_web_app = Flask(__name__)

@_web_app.route('/')
def _index(): return render_template('index.html')
@_web_app.route('/captures')
def _captures_page(): return render_template('captures.html')
@_web_app.route('/analytics')
def _analytics_page(): return render_template('analytics.html')

@_web_app.route('/video_feed')
def _video_feed():
    _cfg = _web_app.config['cfg']
    _cfg.viewer_connected()
    def generate():
        last_seq = -1                   # FIX: skip duplicate frames
        try:
            while _cfg.running:
                seq, jpeg_bytes = _cfg.pop_jpeg()
                if seq is not None and seq != last_seq:
                    last_seq = seq
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                else:
                    time.sleep(0.02)
        finally:
            _cfg.viewer_disconnected()
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                             'Pragma': 'no-cache', 'Expires': '0'})

@_web_app.route('/api/line', methods=['GET', 'POST'])
def _api_line():
    _cfg = _web_app.config['cfg']
    if request.method == 'POST':
        data = request.get_json()
        if data and 'line' in data:
            line = data['line']
            if not line: _cfg.set_line(None); return jsonify({'status': 'ok', 'line': None})
            if len(line) == 4: _cfg.set_line([int(v) for v in line]); return jsonify({'status': 'ok', 'line': _cfg.line})
        return jsonify({'status': 'error'}), 400
    return jsonify({'line': _cfg.line})

@_web_app.route('/api/roi', methods=['GET', 'POST'])
def _api_roi():
    _cfg = _web_app.config['cfg']
    if request.method == 'POST':
        data = request.get_json()
        if data and 'roi' in data:
            roi = data['roi']
            if not roi: _cfg.set_roi(None); return jsonify({'status': 'ok', 'roi': None})
            if len(roi) == 4: _cfg.set_roi([[int(x), int(y)] for x, y in roi]); return jsonify({'status': 'ok', 'roi': _cfg.roi})
        return jsonify({'status': 'error'}), 400
    return jsonify({'roi': _cfg.roi})

@_web_app.route('/api/config', methods=['GET', 'POST'])
def _api_config():
    _cfg = _web_app.config['cfg']
    if request.method == 'POST':
        data = request.get_json()
        if not data: return jsonify({'status': 'error'}), 400
        if 'stream_url'      in data: _cfg.set_stream_url(data['stream_url']);           return jsonify({'status': 'ok'})
        if 'flip_sides'      in data: val = data['flip_sides']; _cfg.set_flip_sides(val if isinstance(val, bool) else str(val).lower() == 'true'); return jsonify({'status': 'ok'})
        if 'enabled_classes' in data: _cfg.set_enabled_classes(data['enabled_classes']); return jsonify({'status': 'ok'})
        return jsonify({'status': 'error'}), 400
    return jsonify({'stream_url': _cfg.stream_url, 'line': _cfg.line, 'roi': _cfg.roi,
                    'conf_thresh': _cfg.conf_thresh, 'flip_sides': _cfg.flip_sides,
                    'enabled_classes': _cfg.enabled_classes})

@_web_app.route('/api/counts')
def _api_counts():
    d = _web_app.config['cfg'].capture_dir
    c_in = c_out = 0
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.endswith('_in.jpg'):  c_in  += 1
            elif fn.endswith('_out.jpg'): c_out += 1
    return jsonify({"in": c_in, "out": c_out})

@_web_app.route('/api/reset', methods=['POST'])
def _api_reset(): return jsonify({'status': 'ok'})

@_web_app.route('/api/captures')
def _api_captures():
    _cfg = _web_app.config['cfg']
    base = request.host_url.rstrip('/')
    return jsonify([{**c, 'url': f"{base}/captures/{c['filename']}",
                     'thumb_url': f"{base}/captures/{c['thumb']}"} for c in _cfg.captures[:20]])

@_web_app.route('/api/captures/all')
def _api_captures_all():
    _cfg = _web_app.config['cfg']
    base = request.host_url.rstrip('/')
    results = []
    if os.path.isdir(_cfg.capture_dir):
        for fn in sorted(os.listdir(_cfg.capture_dir)):
            if not fn.endswith('.jpg') or fn.startswith('thumb'): continue
            # FIX 5: split before stripping extension so direction has no '.jpg'
            parts = fn.replace('.jpg', '').split('_')
            results.append({'filename': fn,
                            'url':       f"{base}/captures/{fn}",
                            'thumb_url': f"{base}/captures/thumb/{fn}",
                            'direction': parts[-1] if len(parts) > 3 else '?',
                            'timestamp': '_'.join(parts[:3]) if len(parts) > 3 else fn})
    return jsonify(results)

@_web_app.route('/api/captures/delete', methods=['POST'])
def _api_captures_delete():
    data = request.get_json()
    if not data or 'files' not in data: return jsonify({'status': 'error'}), 400
    _cfg = _web_app.config['cfg']; deleted = 0
    for fn in data['files']:
        for p in [fn, f"thumb/{fn}"]:
            fp = os.path.join(_cfg.capture_dir, p)
            if os.path.exists(fp): os.remove(fp); deleted += 1
    return jsonify({'status': 'ok', 'deleted': deleted})

@_web_app.route('/captures/<path:filename>')
def _serve_capture(filename):
    return send_from_directory(_web_app.config['cfg'].capture_dir, filename)

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

# ── Reader thread: cap management + MJPEG stream at fixed fps ─────────
_STREAM_INTERVAL = 1.0 / 30   # 30 fps stream target

def reader_loop(cfg, raw_q, latest_frame, frame_lock, tracker_state, tracker_lock):
    """Reads camera frames, feeds detect queue, updates latest frame for encoder."""
    cap = None
    last_read_t = 0.0
    read_interval = 0.0

    while cfg.running:
        if cap is not None and cfg.check_reconnect():
            print("[reader] URL changed, reconnecting...")
            cap.release(); cap = None

        if cap is None:
            url = cfg.stream_url
            if not url:
                time.sleep(1); continue
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("[reader] Failed to open stream, retrying in 5s...")
                cap.release(); cap = None; time.sleep(5); continue
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            read_interval = 1.0 / video_fps if video_fps > 0 else 0.0
            last_read_t = time.time()
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

        # Feed detect thread; drop if it's still busy with previous frame
        frame_shared = True
        try:
            raw_q.put_nowait(frame)
        except Full:
            frame_shared = False

        # Update latest frame for encoder thread (copy only if detector may use it)
        with frame_lock:
            latest_frame[0] = frame if not frame_shared else frame.copy()

    if cap is not None:
        cap.release()
    print("[reader] Done")


def encoder_loop(cfg, latest_frame, frame_lock, tracker_state, tracker_lock):
    """Encodes and pushes annotated JPEGs at the stream target FPS."""
    fps_timer   = time.time()
    stream_fps  = 0.0
    last_jpeg_t = time.time()
    stream_count = 0

    while cfg.running:
        # Sleep until next stream deadline
        next_deadline = last_jpeg_t + _STREAM_INTERVAL
        sleep_time = max(0.001, next_deadline - time.time())
        time.sleep(sleep_time)
        if not cfg.running:
            break

        now = time.time()
        if not cfg.has_viewers():
            last_jpeg_t = now
            continue

        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            continue

        with tracker_lock:
            objs      = list(tracker_state['objects'])
            counts    = {'in': tracker_state['c_in'], 'out': tracker_state['c_out']}
            det_count = tracker_state['det_count']

        last_jpeg_t = now
        annotated = annotate_frame(frame.copy(), cfg.line, cfg.roi, objs, counts,
                                   stream_fps, det_count, flip_sides=cfg.flip_sides, simple=True)
        # Resize to 640px wide before JPEG encode for speed
        h, w = annotated.shape[:2]
        if w > 640:
            scale = 640.0 / w
            annotated = cv2.resize(annotated, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        ok, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ok:
            cfg.push_jpeg(jpeg.tobytes())
            stream_count += 1

        if now - fps_timer >= 1.0:
            stream_fps  = stream_count / (now - fps_timer)
            stream_count = 0
            fps_timer   = now

    print("[encoder] Done")

# ── Main: detect + track loop ─────────────────────────────────────────
def main():
    print("[main] Starting Vehicle Line Counter")
    print(f"[main] Stream: {cfg.stream_url}")

    # FIX 4: pass conf_thresh from config
    detector    = NanoDetONNX('/home/skpawar1305/track_vehicles/models/nanodet-plus-m-1.5x_416.onnx',
                               conf_thresh=cfg.conf_thresh)
    print("[main] NanoDet loaded")

    tracker     = BYTETracker(track_thresh=cfg.conf_thresh, track_buffer=50,
                              match_thresh=0.7, frame_rate=30)
    capture_mgr = CaptureManager(cfg.capture_dir, cfg.max_captures)

    # Shared state written by detect loop, read by reader/encoder threads
    tracker_state = {'objects': [], 'c_in': 0, 'c_out': 0, 'det_count': 0}
    tracker_lock  = threading.Lock()
    latest_frame  = [None]
    frame_lock    = threading.Lock()

    # Tracked state per track_id for crossing detection
    prev_centroids = {}
    track_ages      = {}
    last_cross_info = {}

    # Queue passes raw frames from reader to detect thread (maxsize=1: always latest)
    raw_q = Queue(maxsize=1)

    _web_app.config['cfg'] = cfg
    threading.Thread(target=_web_app.run,
                     kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True,
                             'debug': False, 'use_reloader': False},
                     daemon=True).start()
    print("[main] Web UI at http://0.0.0.0:5000")

    threading.Thread(target=reader_loop,
                     args=(cfg, raw_q, latest_frame, frame_lock, tracker_state, tracker_lock),
                     daemon=True).start()
    print("[main] Reader thread started")

    threading.Thread(target=encoder_loop,
                     args=(cfg, latest_frame, frame_lock, tracker_state, tracker_lock),
                     daemon=True).start()
    print("[main] Encoder thread started")

    c_in = c_out = 0
    total_frames = 0   # FIX 2: non-resetting counter for crossing debounce

    try:
        while cfg.running:
            try:
                frame = raw_q.get(timeout=0.5)
            except Empty:
                continue

            total_frames += 1

            raw_detections = []
            if cfg.roi and len(cfg.roi) >= 3:
                raw_detections = detector.detect_roi(frame, cfg.roi,
                                                     enabled_classes=cfg.enabled_classes)
            det_count = len(raw_detections)

            # Convert detections to BYTETracker format: [[x1,y1,x2,y2,score,cls_id], ...]
            if raw_detections:
                dets_array = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3],
                                        d['confidence'], d['class_id']] for d in raw_detections],
                                      dtype=np.float32)
            else:
                dets_array = np.empty((0, 6), dtype=np.float32)

            tracked = tracker.update(torch.from_numpy(dets_array), None)

            # Convert BYTETracker output to list of tracked objects
            objects = []
            tracked_ids = set()
            for t in tracked:
                x1, y1, x2, y2, tid, cls_id, score = t
                x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
                tracked_ids.add(tid)
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                prev = prev_centroids.get(tid)
                age = track_ages.get(tid, 0)
                obj = TrackInfo(tid, (x1, y1, x2, y2),
                                _VEHICLE_NAMES.get(int(cls_id), f'cls_{int(cls_id)}'),
                                float(score), centroid, prev, age)
                objects.append(obj)
                prev_centroids[tid] = centroid
                track_ages[tid] = age + 1

            # Clean up stale tracks
            for tid in list(prev_centroids.keys()):
                if tid not in tracked_ids:
                    del prev_centroids[tid]
                    del track_ages[tid]

            # Publish tracker state for reader thread to annotate stream frames
            with tracker_lock:
                tracker_state['objects']   = list(objects)
                tracker_state['c_in']      = c_in
                tracker_state['c_out']     = c_out
                tracker_state['det_count'] = det_count

            for obj in objects:
                if obj.prev_centroid and obj.age >= 3:
                    crossing = detect_crossing(cfg.line, obj.prev_centroid, obj.centroid,
                                               flip=cfg.flip_sides)
                    if crossing == CROSSING_NONE: continue
                    # Only count if bbox physically touches the line too
                    if not cfg.line or not bbox_touches_line(cfg.line, obj.bbox):
                        continue
                    info = last_cross_info.get(obj.track_id, {'frame': -60, 'dir': None})
                    if total_frames - info['frame'] < 15: continue
                    last_cross_info[obj.track_id] = {'frame': total_frames, 'dir': crossing}
                    direction = 'IN' if crossing == CROSSING_IN else 'OUT'
                    # Annotate capture with current state
                    cap_frame = annotate_frame(frame.copy(), cfg.line, cfg.roi, objects,
                                               {"in": c_in, "out": c_out},
                                               0.0, det_count, flip_sides=cfg.flip_sides)
                    entry = capture_mgr.save(cap_frame, obj.track_id, crossing, obj.bbox)
                    cfg.add_capture(entry)
                    if crossing == CROSSING_IN: c_in += 1
                    else:                       c_out += 1
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