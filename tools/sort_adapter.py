"""Classic SORT baseline for the tracker bake-off.

Uses the Kalman box tracker vendored with OC-SORT, but association is plain
IoU + Hungarian, without OC-SORT's observation-centric re-update or direction
cost. It is a useful control: if SORT and OC-SORT fail the same way, the issue
is not the OC-SORT-specific machinery.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import TrackInfo, box_iou  # noqa: E402
from ocsort.association import associate_detections_to_trackers  # noqa: E402
from ocsort.ocsort import KalmanBoxTracker  # noqa: E402


class SortAdapter:
    def __init__(self, max_age=40, iou_threshold=0.3, min_hits=1):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.prev = {}
        self.meta = {}
        self.age = {}
        KalmanBoxTracker.count = 0

    @staticmethod
    def _centroid(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _best_det(self, bbox, dets, used):
        best_i, best_iou = None, -1.0
        for i, d in enumerate(dets):
            if i in used:
                continue
            iou = box_iou(bbox, d["bbox"])
            if iou > best_iou:
                best_i, best_iou = i, iou
        return best_i, best_iou

    def update(self, dets, w, h, now=None):
        self.frame_count += 1
        if dets:
            arr = np.array([[float(d["bbox"][0]), float(d["bbox"][1]),
                             float(d["bbox"][2]), float(d["bbox"][3])]
                            for d in dets], dtype=np.float32)
        else:
            arr = np.empty((0, 4), dtype=np.float32)

        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        for i, trk in enumerate(self.trackers):
            trks[i] = trk.predict()[0][:4]

        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(arr, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(
                np.concatenate((arr[m[0]], [1.0])))
        for i in unmatched_trks:
            self.trackers[int(i)].update(None)
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(
                np.concatenate((arr[int(i)], [1.0]))))

        measured, used = {}, set()
        for trk in self.trackers:
            if trk.time_since_update >= 1:
                continue
            bbox = tuple(float(x) for x in trk.last_observation[:4])
            tid = int(trk.id) + 1
            di, iou = self._best_det(bbox, dets, used)
            if di is not None and iou >= 0.3:
                used.add(di)
                d = dets[di]
                meta = (int(d["class_id"]), d["label"], float(d["confidence"]))
            else:
                meta = self.meta.get(tid, (7, "truck", 0.0))
            self.meta[tid] = meta
            prev = self.prev.get(tid)
            cen = self._centroid(bbox)
            self.prev[tid] = cen
            age = self.age.get(tid, 0) + 1
            self.age[tid] = age
            measured[tid] = TrackInfo(tid, bbox, meta[1], meta[0], meta[2],
                                      cen, prev, age)

        objects = list(measured.values())
        for trk in self.trackers:
            tid = int(trk.id) + 1
            if tid in measured:
                continue
            state = trk.get_state()[0]
            if np.any(~np.isfinite(state)):
                continue
            bbox = tuple(float(x) for x in state[:4])
            meta = self.meta.get(tid, (7, "truck", 0.0))
            objects.append(TrackInfo(tid, bbox, meta[1], meta[0], meta[2],
                                     self._centroid(bbox), None,
                                     int(getattr(trk, "age", 0))))

        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]
        live = {int(t.id) + 1 for t in self.trackers}
        for d in (self.prev, self.meta, self.age):
            for tid in list(d):
                if tid not in live:
                    del d[tid]
        return objects
