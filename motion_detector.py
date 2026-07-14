import cv2
import numpy as np


class MotionDetector:
    def __init__(self, padding=120, threshold=500, resize_width=320):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=False
        )
        self.padding = padding
        self.threshold = threshold
        self.resize_width = resize_width
        self._line = None
        self._zone = None
        self.motion_state = False

    @property
    def line(self):
        return self._line

    def update_line(self, line):
        self._line = line
        if line:
            x1, y1, x2, y2 = line
            pad = self.padding
            self._zone = (
                max(0, min(x1, x2) - pad),
                max(0, min(y1, y2) - pad),
                min(max(x1, x2) + pad, 99999),
                min(max(y1, y2) + pad, 99999),
            )

    def detect(self, frame):
        if self._line is None or self._zone is None:
            self.motion_state = False
            return False

        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = self._zone
        zx2 = min(zx2, w)
        zy2 = min(zy2, h)
        zx1 = max(zx1, 0)
        zy1 = max(zy1, 0)

        if zx2 <= zx1 or zy2 <= zy1:
            self.motion_state = False
            return False

        roi = frame[zy1:zy2, zx1:zx2]
        if roi.size == 0:
            self.motion_state = False
            return False

        scale = self.resize_width / roi.shape[1]
        new_w = self.resize_width
        new_h = int(roi.shape[0] * scale)
        if new_h == 0:
            self.motion_state = False
            return False
        small = cv2.resize(roi, (new_w, new_h))

        fgmask = self.subtractor.apply(small)
        fgmask = cv2.medianBlur(fgmask, 5)
        fg_pixels = np.count_nonzero(fgmask)

        self.motion_state = fg_pixels > self.threshold
        return self.motion_state
