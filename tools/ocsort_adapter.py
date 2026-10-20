"""OC-SORT adapter for the production tracker interface.

The upstream OC-SORT core is vendored under ``ocsort/`` (MIT). It expects
``update()`` once per camera frame, including frames with no detections; that is
the native mode this adapter exposes.

``run.py``'s ``TrackInfo`` is used so the benchmark can pass these objects to the
same ``CrossingGate`` as the production centroid/ByteTrack backends.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import TrackInfo, box_iou  # noqa: E402
from ocsort.ocsort import OCSort  # noqa: E402


class OCSortAdapter:
    def __init__(self, det_thresh=0.3, max_age=40, min_hits=1,
                 iou_threshold=0.3, delta_t=3, inertia=0.2, debug=False):
        self.oc = OCSort(det_thresh=det_thresh, max_age=max_age,
                         min_hits=min_hits, iou_threshold=iou_threshold,
                         delta_t=delta_t, asso_func="iou", inertia=inertia,
                         use_byte=False)
        self.prev = {}       # tid -> last measured centroid
        self.meta = {}       # tid -> (class_id, label, confidence)
        self.age = {}        # tid -> measured age
        self.debug = debug

    @staticmethod
    def _bbox_centroid(bbox):
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
        if dets:
            arr = np.array([[float(d["bbox"][0]), float(d["bbox"][1]),
                             float(d["bbox"][2]), float(d["bbox"][3]),
                             float(d["confidence"])] for d in dets],
                           dtype=np.float32)
        else:
            arr = np.empty((0, 5), dtype=np.float32)

        # img_info/img_size are the frame size, so OC-SORT's scale is 1.0.
        out = self.oc.update(arr, (h, w), (h, w))

        measured, used = {}, set()
        for row in out:
            bbox = tuple(float(x) for x in row[:4])
            tid = int(row[4])
            di, iou = self._best_det(bbox, dets, used)
            if di is not None and iou >= 0.3:
                used.add(di)
                d = dets[di]
                meta = (int(d["class_id"]), d["label"], float(d["confidence"]))
            else:
                meta = self.meta.get(tid, (7, "truck", 0.0))
            self.meta[tid] = meta
            prev = self.prev.get(tid)
            cen = self._bbox_centroid(bbox)
            self.prev[tid] = cen
            age = self.age.get(tid, 0) + 1
            self.age[tid] = age
            measured[tid] = TrackInfo(tid, bbox, meta[1], meta[0], meta[2],
                                      cen, prev, age)

        objects = list(measured.values())
        # Emit lost/unconfirmed tracks as predictions so the live overlay and the
        # gate's per-id state do not blink out between detector results.
        for trk in self.oc.trackers:
            tid = int(trk.id) + 1
            if tid in measured:
                continue
            state = trk.get_state()[0]
            if np.any(~np.isfinite(state)):
                continue
            bbox = tuple(float(x) for x in state[:4])
            meta = self.meta.get(tid, (7, "truck", 0.0))
            objects.append(TrackInfo(tid, bbox, meta[1], meta[0], meta[2],
                                     self._bbox_centroid(bbox), None,
                                     int(getattr(trk, "age", 0))))

        live = {int(t.id) + 1 for t in self.oc.trackers}
        for d in (self.prev, self.meta, self.age):
            for tid in list(d):
                if tid not in live:
                    del d[tid]
        return objects
