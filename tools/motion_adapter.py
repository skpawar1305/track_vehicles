"""Motion-aware global-assignment tracker for the bake-off.

The production centroid tracker uses a greedy (tier, cost) sort. In a scene with
two overlapping vehicles this can let a track swap to the wrong box. This
variant keeps the same motion model but solves each association tier with the
Hungarian algorithm and adds a direction penalty when a track has a reliable
velocity, so a left-moving track is not matched to a box that reappeared behind
it.

It is deliberately a tracker-only experiment: the detector and crossing gate are
unchanged.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import TrackInfo, box_iou  # noqa: E402

BIG = 1e6


class MotionTracker:
    def __init__(self, max_age=40, assoc_frac=0.2, gap_frac=0.5,
                 gap_max_frac=1.0, iou_gate=0.2, vel_smooth=0.6,
                 vel_reset=0.4, pred_cap=0.5, direction_min_speed=8.0,
                 direction_penalty=0.8, fallback_max_cost=1.15,
                 appearance_weight=0.0):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.assoc_frac = assoc_frac
        self.gap_frac = gap_frac
        self.gap_max_frac = gap_max_frac
        self.iou_gate = iou_gate
        self.vel_smooth = vel_smooth
        self.vel_reset = vel_reset
        self.pred_cap = pred_cap
        self.direction_min_speed = direction_min_speed
        self.direction_penalty = direction_penalty
        self.fallback_max_cost = fallback_max_cost
        self.appearance_weight = float(appearance_weight)

    @staticmethod
    def _centroid(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _advance(self, t, dt):
        dt = min(dt, self.pred_cap)
        return (t["cen"][0] + t["vel"][0] * dt,
                t["cen"][1] + t["vel"][1] * dt)

    def _pred_box(self, t, dt):
        bx1, by1, bx2, by2 = t["det"]["bbox"]
        vx, vy = t["vel"]
        return (bx1 + vx * dt, by1 + vy * dt, bx2 + vx * dt, by2 + vy * dt)

    @staticmethod
    def _hist_dist(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))

    @staticmethod
    def _blend(a, b, alpha=0.8):
        if a is None:
            return b
        if b is None:
            return a
        return (alpha * a + (1.0 - alpha) * b).astype(np.float32)

    @staticmethod
    def _hungarian(cost, max_cost):
        if cost.size == 0:
            return []
        rows, cols = linear_sum_assignment(cost)
        out = []
        for r, c in zip(rows, cols):
            if cost[r, c] < BIG / 2 and cost[r, c] <= max_cost:
                out.append((int(r), int(c)))
        return out

    def update(self, dets, w, h, now=None):
        if now is None:
            now = time.time()
        base = self.assoc_frac * w
        diag = (w * w + h * h) ** 0.5
        max_gate = max(base, self.gap_max_frac * diag)

        tids = list(self.tracks)
        pred = {}
        for tid in tids:
            t = self.tracks[tid]
            dt = max(0.0, now - t["last_t"])
            gap = max(0.0, now - t["meas_t"])
            pred[tid] = {
                "box": self._pred_box(t, dt),
                "cen": self._advance(t, dt),
                "gate": min(base + self.gap_frac * w * gap, max_gate),
            }

        pairs, used_t, used_d = {}, set(), set()

        # Tier 1: IoU against the velocity-predicted box, global assignment.
        if tids and dets:
            cost = np.full((len(tids), len(dets)), BIG, dtype=np.float64)
            for r, tid in enumerate(tids):
                for c, d in enumerate(dets):
                    iou = box_iou(pred[tid]["box"], d["bbox"])
                    if iou >= self.iou_gate:
                        app = self.appearance_weight * self._hist_dist(
                            self.tracks[tid].get("hist"), d.get("hist"))
                        cost[r, c] = 1.0 - iou + app
            for r, c in self._hungarian(cost, 0.8 + self.appearance_weight):
                tid = tids[r]
                pairs[tid] = c
                used_t.add(tid)
                used_d.add(c)

        # Tier 2: distance fallback for the rest, with a direction penalty.
        left_t = [tid for tid in tids if tid not in used_t]
        left_d = [c for c in range(len(dets)) if c not in used_d]
        if left_t and left_d:
            cost = np.full((len(left_t), len(left_d)), BIG, dtype=np.float64)
            for r, tid in enumerate(left_t):
                t = self.tracks[tid]
                p = pred[tid]
                vx, vy = t["vel"]
                speed = (vx * vx + vy * vy) ** 0.5
                gap = max(0.0, now - t["meas_t"])
                for c, di in enumerate(left_d):
                    d = dets[di]
                    dx = d["centroid"][0] - p["cen"][0]
                    dy = d["centroid"][1] - p["cen"][1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > p["gate"]:
                        continue
                    penalty = 0.0
                    if speed >= self.direction_min_speed and gap < 2.0:
                        mx = d["centroid"][0] - t["meas"][0]
                        my = d["centroid"][1] - t["meas"][1]
                        m = (mx * mx + my * my) ** 0.5
                        if m > 4.0:
                            cos = (mx * vx + my * vy) / (m * speed)
                            if cos < 0.0:
                                penalty = self.direction_penalty * (-cos)
                    app = self.appearance_weight * self._hist_dist(
                        t.get("hist"), d.get("hist"))
                    cost[r, c] = dist / max(p["gate"], 1e-6) + penalty + app
            for r, c in self._hungarian(
                    cost, self.fallback_max_cost + self.appearance_weight):
                tid = left_t[r]
                pairs[tid] = left_d[c]
                used_t.add(tid)
                used_d.add(left_d[c])

        objects = []
        for tid, di in pairs.items():
            tr, det = self.tracks[tid], dets[di]
            prev = tr["meas"]
            cur = det["centroid"]
            dt = now - tr["meas_t"]
            if dt <= 1e-3:
                dt = 1e-3
            vx = (cur[0] - prev[0]) / dt
            vy = (cur[1] - prev[1]) / dt
            if dt > self.vel_reset:
                tr["vel"] = (vx, vy)
            else:
                s = self.vel_smooth
                tr["vel"] = (s * vx + (1 - s) * tr["vel"][0],
                             s * vy + (1 - s) * tr["vel"][1])
            tr["cen"] = tr["meas"] = cur
            tr["last_t"] = tr["meas_t"] = now
            tr["age"] += 1
            tr["missed"] = 0
            tr["det"] = det
            tr["hist"] = self._blend(tr.get("hist"), det.get("hist"))
            objects.append(TrackInfo(tid, det["bbox"], det["label"],
                                     det["class_id"], det["confidence"],
                                     cur, prev, tr["age"]))

        for tid, tr in self.tracks.items():
            if tid in used_t:
                continue
            dt = max(0.0, now - tr["last_t"])
            tr["cen"] = self._advance(tr, dt)
            tr["last_t"] = now
            tr["age"] += 1
            tr["missed"] += 1
            objects.append(TrackInfo(tid, tr["det"]["bbox"], tr["det"]["label"],
                                     tr["det"]["class_id"], tr["det"]["confidence"],
                                     tr["cen"], None, tr["age"]))

        for di, det in enumerate(dets):
            if di in used_d:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"cen": det["centroid"], "meas": det["centroid"],
                                "vel": (0.0, 0.0), "age": 0, "missed": 0,
                                "det": det, "last_t": now, "meas_t": now,
                                "hist": det.get("hist")}
            objects.append(TrackInfo(tid, det["bbox"], det["label"],
                                     det["class_id"], det["confidence"],
                                     det["centroid"], None, 0))

        for tid in [t for t, tr in self.tracks.items()
                    if tr["missed"] > self.max_age]:
            del self.tracks[tid]
        return objects
