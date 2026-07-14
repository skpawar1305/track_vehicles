import os
import cv2
from datetime import datetime


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
        thumb_filename = f"thumb/{filename}"

        full_path = os.path.join(self.capture_dir, filename)
        thumb_path = os.path.join(self.capture_dir, thumb_filename)

        cv2.imwrite(full_path, frame)

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop_size = max(x2 - x1, y2 - y1) * 2
        crop_size = min(crop_size, w, h)
        crop_x1 = max(0, cx - crop_size // 2)
        crop_y1 = max(0, cy - crop_size // 2)
        crop_x2 = min(w, crop_x1 + crop_size)
        crop_y2 = min(h, crop_y1 + crop_size)
        thumb = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if thumb.size > 0:
            thumb = cv2.resize(thumb, (256, 256))
            cv2.imwrite(thumb_path, thumb)

        return {
            "filename": filename,
            "thumb": thumb_filename,
            "timestamp": ts,
            "track_id": track_id,
            "direction": direction_label,
        }
