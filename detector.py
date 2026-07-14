import os
import cv2
import numpy as np

try:
    import ncnn
except ImportError:
    ncnn = None

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


class YOLO26nDetector:
    def __init__(self, model_path, conf_thresh=0.5, target_size=320):
        self.conf_thresh = conf_thresh
        self.target_size = target_size
        self.vehicle_ids = set(VEHICLE_CLASSES.keys())
        self.net = None

        if ncnn is None:
            raise ImportError("ncnn not installed")

        param_path, bin_path = None, None
        if os.path.isdir(model_path):
            for f in os.listdir(model_path):
                if f.endswith('.param'):
                    param_path = os.path.join(model_path, f)
                elif f.endswith('.bin'):
                    bin_path = os.path.join(model_path, f)
        if param_path is None:
            param_path = f"{model_path}.param"
        if bin_path is None:
            bin_path = f"{model_path}.bin"

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def detect(self, frame, enabled_classes=None):
        h, w = frame.shape[:2]
        ts = self.target_size

        scale = min(ts / h, ts / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (ts - new_h) // 2
        pad_w = (ts - new_w) // 2

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((ts, ts, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        in_mat = ncnn.Mat.from_pixels(canvas, 2, ts, ts)
        in_mat.substract_mean_normalize([], [1 / 255.0] * 3)

        ex = self.net.create_extractor()
        ex.input("in0", in_mat)
        _, out = ex.extract("out0")

        detections = self._postprocess(out, scale, pad_w, pad_h, w, h)
        class_ids = set(enabled_classes) if enabled_classes else self.vehicle_ids
        return [d for d in detections if d['class_id'] in class_ids]

    def _postprocess(self, out, scale, pad_w, pad_h, orig_w, orig_h):
        out_np = np.array(out)
        if out_np.ndim == 3:
            out_np = out_np[0]
        out_np = out_np.T

        results = []
        for i in range(out_np.shape[0]):
            vals = out_np[i]
            cx, cy, w, h = vals[:4]
            scores = vals[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < self.conf_thresh:
                continue

            x1 = int(max(0, min((cx - w / 2 - pad_w) / scale, orig_w)))
            y1 = int(max(0, min((cy - h / 2 - pad_h) / scale, orig_h)))
            x2 = int(max(0, min((cx + w / 2 - pad_w) / scale, orig_w)))
            y2 = int(max(0, min((cy + h / 2 - pad_h) / scale, orig_h)))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                'bbox': (x1, y1, x2, y2),
                'centroid': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': conf,
                'class_id': cls_id,
                'label': VEHICLE_CLASSES.get(cls_id, f'cls_{cls_id}'),
            })

        results.sort(key=lambda r: r['confidence'], reverse=True)
        keep = []
        for r in results:
            if not any(iou(r['bbox'], k['bbox']) > 0.45 for k in keep):
                keep.append(r)
        return keep


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)
