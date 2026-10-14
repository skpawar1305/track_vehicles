#!/usr/bin/env python3
"""Regression: the production tracker must be detector-cadence independent.

The Pi reads the substream at 25+ fps but runs detection at its own (variable,
slower) rate, so the gate must see a usable probe between detections. The
``centroid`` backend tracks in px/s with variable dt and coasts/re-associates at
the reader rate, so it must return exactly one OUT across a wide sweep of
detection cadences. ``bytetrack`` is frame-based and is expected to be fragile
here; this test pins the centroid property that production relies on.

    python tools/test_tracker_cadence.py
"""
import os
import sys

os.environ.setdefault("TRACKER_RELAY_REEXEC", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import (  # noqa: E402
    ByteTrackAdapter, CentroidTracker, CrossingGate, CROSSING_IN, CROSSING_OUT,
)


W, H = 640, 360
FPS = 25.0
LINE = (500, 0, 500, H)
# Dumper passes right-to-left across a vertical line and keeps going well past
# it (as in the real clips), so a coarse detector still has post-line
# observations with which to confirm the transit.
T_START, T_END = 0.2, 3.0
X0, X1 = 620, 180
BOX_W, BOX_H = 90, 80
BOX_Y = 150


def det_at(t):
    if t < T_START or t > T_END:
        return None
    frac = (t - T_START) / (T_END - T_START)
    x = X0 + (X1 - X0) * frac
    box = (x, BOX_Y, x + BOX_W, BOX_Y + BOX_H)
    return {"bbox": box, "centroid": (x + BOX_W / 2, BOX_Y + BOX_H / 2),
            "confidence": 0.9, "class_id": 7, "label": "truck"}


def replay(tracker, detect_every):
    gate = CrossingGate(min_age=1, margin_frac=0.06, rearm_frac=0.2,
                        confirm_frac=0.2, probe_edge="left")
    gate_ids, events = set(), []
    n = int((T_END + 0.4) * FPS)
    objects = []
    for frame in range(n):
        t = frame / FPS
        detect_now = (frame % detect_every == 0)
        d = det_at(t)
        raw = [d] if (detect_now and d) else []
        # CentroidTracker is designed to be advanced at reader rate (coasts with
        # empty dets); ByteTrack is only advanced when a detection arrives.
        if isinstance(tracker, ByteTrackAdapter):
            if not detect_now:
                continue
            objects = tracker.update(raw, W, H, now=t)
        else:
            objects = tracker.update(raw, W, H, now=t)
        present = {o.track_id for o in objects}
        gate.drop(gate_ids - present)
        gate_ids = present
        for obj in objects:
            if obj.prev_centroid is None or obj.age < 1:
                continue
            c = gate.update(LINE, obj, flip=True, frame_h=H, now=t, frame_w=W)
            if c == CROSSING_OUT:
                events.append("OUT")
            elif c == CROSSING_IN:
                events.append("IN")
    return events


def main():
    cadences = [2, 3, 4, 5, 6, 7]
    failures = []
    print("detect_every   centroid          bytetrack")
    for every in cadences:
        cen = replay(
            CentroidTracker(max_age=40, iou_gate=0.2, assoc_frac=0.2,
                            gap_frac=0.5, gap_max_frac=1.0, vel_reset=0.4,
                            pred_cap=0.5),
            every)
        bt = replay(
            ByteTrackAdapter(track_thresh=0.3, track_buffer=40,
                             match_thresh=0.8, frame_rate=30),
            every)
        print(f"{every:>10}   {str(cen):<16}  {bt}")
        if cen != ["OUT"]:
            failures.append((every, cen))
    assert not failures, f"centroid lost the crossing at cadences {failures}"
    print("PASS: centroid returns exactly one OUT at every detection cadence")


if __name__ == "__main__":
    main()
