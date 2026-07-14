import json
import threading
from collections import deque

CONFIG_PATH = "config.json"


class Config:
    def __init__(self):
        self.lock = threading.RLock()
        self.stream_url = ""
        self.line = None
        self.counts = {"in": 0, "out": 0}
        self.conf_thresh = 0.5
        self.flip_sides = False
        self.motion_thresh = 500
        self.capture_dir = "captures"
        self.max_captures = 1000
        self.model_path = "models/yolo26n_ncnn_model"
        self.target_size = 320
        self.enabled_classes = [2, 3, 5, 7]
        self.running = True
        self._reconnect = False
        self.viewers = 0
        self.frame_buffer = deque(maxlen=1)
        self.captures = []
        self._load()

    def _load(self):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            with self.lock:
                self.stream_url = data.get("stream_url", "")
                self.line = data.get("line")
                c = data.get("counts", {})
                self.counts["in"] = c.get("in", 0)
                self.counts["out"] = c.get("out", 0)
                self.conf_thresh = data.get("conf_thresh", 0.5)
                self.flip_sides = data.get("flip_sides", False)
                self.motion_thresh = data.get("motion_thresh", 500)
                self.capture_dir = data.get("capture_dir", "captures")
                self.max_captures = data.get("max_captures", 1000)
                self.model_path = data.get("model_path", "models/yolo26n_ncnn_model")
                self.target_size = data.get("target_size", 320)
                self.enabled_classes = data.get("enabled_classes", [2, 3, 5, 7])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self):
        with self.lock:
            data = {
                "stream_url": self.stream_url,
                "line": self.line,
                "counts": dict(self.counts),
                "conf_thresh": self.conf_thresh,
                "flip_sides": self.flip_sides,
                "motion_thresh": self.motion_thresh,
                "capture_dir": self.capture_dir,
                "max_captures": self.max_captures,
                "model_path": self.model_path,
                "target_size": self.target_size,
                "enabled_classes": self.enabled_classes,
            }
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def set_line(self, line):
        with self.lock:
            self.line = line
        self.save()

    def set_stream_url(self, url):
        with self.lock:
            self.stream_url = url
            self._reconnect = True
        self.save()

    def check_reconnect(self):
        with self.lock:
            if self._reconnect:
                self._reconnect = False
                return True
            return False

    def set_flip_sides(self, flip):
        with self.lock:
            self.flip_sides = flip
        self.save()

    def set_enabled_classes(self, classes):
        with self.lock:
            self.enabled_classes = [int(c) for c in classes]
        self.save()

    def reset_counts(self):
        with self.lock:
            self.counts["in"] = 0
            self.counts["out"] = 0
        self.save()

    def get_counts(self):
        with self.lock:
            return dict(self.counts)

    def has_viewers(self):
        with self.lock:
            return self.viewers > 0

    def viewer_connected(self):
        with self.lock:
            self.viewers += 1

    def viewer_disconnected(self):
        with self.lock:
            if self.viewers > 0:
                self.viewers -= 1

    def push_jpeg(self, jpeg_bytes):
        with self.lock:
            self.frame_buffer.append(jpeg_bytes)

    def pop_jpeg(self):
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
            return None

    def add_capture(self, capture_info):
        with self.lock:
            self.captures.insert(0, capture_info)
            if len(self.captures) > 50:
                self.captures = self.captures[:50]
