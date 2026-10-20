#!/usr/bin/env python3
"""Replay the three diagnostic clips through the production detector and gate,
comparing the centroid tracker with the vendored OC-SORT.

Detector output is cached once per clip, then replayed at several detection
cadences. This isolates tracking/crossing behaviour from detector runtime and
lets every tracker see exactly the same boxes.

    python tools/bench_tracker_clips.py
    python tools/bench_tracker_clips.py --clips failed/ins/2.mp4 --limit 300
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import (  # noqa: E402
    PicodetNcnn, CentroidTracker, ByteTrackAdapter, CrossingGate,
    CROSSING_IN, CROSSING_OUT, line_geometry,
)
from ocsort_adapter import OCSortAdapter  # noqa: E402
from sort_adapter import SortAdapter  # noqa: E402
from motion_adapter import MotionTracker  # noqa: E402

# Live server config (2026-10-20), normalized to the clip dimensions.
LINE_NORM = [[0.5965770171149144, 0.015213257267046998],
             [0.5427872860635696, 0.658516707416463]]
ROI_NORM = [[0.1234718826405868, 0.030426514534093996],
            [0.13691931540342298, 0.5976636783482749],
            [0.9865525672371638, 0.6172235805487639],
            [0.988997555012225, 0.017386579733767997]]
FLIP = True
PROBE = "left"

CLIPS = [
    "failed/outs/1.mp4",
    "failed/outs/2.mp4",
    "failed/ins/2.mp4",
]
EXPECTED = {
    "failed/outs/1.mp4": {"IN": 0, "OUT": 1},
    "failed/outs/2.mp4": {"IN": 0, "OUT": 1},
    "failed/ins/2.mp4": {"IN": 1, "OUT": 1},
}
EXPECTED_IDS = {
    "failed/outs/1.mp4": 1,
    "failed/outs/2.mp4": 1,
    "failed/ins/2.mp4": 2,
}


def line_px(line_norm, w, h):
    a, b = line_norm
    return (int(a[0] * w), int(a[1] * h), int(b[0] * w), int(b[1] * h))


def starts_on_expected_side(direction, first_centroid, line):
    """An IN track must first be seen left of the line; OUT right of it."""
    x1, y1, x2, y2 = line
    px, py = first_centroid
    lx = x1 if abs(y2 - y1) < 1e-6 else x1 + (x2 - x1) * (py - y1) / (y2 - y1)
    left = px < lx
    return left if direction == "IN" else not left


def roi_px(roi_norm, w, h):
    return [[int(x * w), int(y * h)] for x, y in roi_norm]


def hsv_hist(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    total = float(h.sum())
    return None if total <= 0 else (h / total).astype(np.float32)


def load_dets(path, detector, classes, limit=None):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rid = roi_px(ROI_NORM, w, h)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect_roi(frame, rid, enabled_classes=classes)
        for d in dets:
            d["hist"] = hsv_hist(frame, d["bbox"])
        out.append(dets)
        if limit and len(out) >= limit:
            break
    cap.release()
    return fps, w, h, out


def suppress_contained(dets, thresh):
    """Drop a lower-confidence box mostly inside a kept box (NanoDet-style).

    PicoDet can emit a tight cab/front box nested in the full-truck box. Their
    IoU is low, so IoU-NMS keeps both; the extra box can steal a track's
    association exactly at the line.
    """
    if thresh <= 0 or len(dets) < 2:
        return dets
    keep = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        x1, y1, x2, y2 = d["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        redundant = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k["bbox"]
            karea = max(0, kx2 - kx1) * max(0, ky2 - ky1)
            iw = max(0, min(x2, kx2) - max(x1, kx1))
            ih = max(0, min(y2, ky2) - max(y1, ky1))
            small = min(area, karea)
            if small > 0 and (iw * ih) / small > thresh:
                redundant = True
                break
        if not redundant:
            keep.append(d)
    return keep


def make_tracker(name, max_age=40, fps=30.0, appearance_weight=0.5):
    if name == "centroid":
        return CentroidTracker(max_age=max_age, iou_gate=0.2, assoc_frac=0.2,
                               gap_frac=0.5, gap_max_frac=1.0,
                               vel_reset=0.4, pred_cap=0.5)
    if name == "ocsort":
        return OCSortAdapter(det_thresh=0.3, max_age=max_age, min_hits=1,
                             iou_threshold=0.3, delta_t=3, inertia=0.2)
    if name == "ocsort-detonly":
        return OCSortAdapter(det_thresh=0.3, max_age=max_age, min_hits=1,
                             iou_threshold=0.3, delta_t=3, inertia=0.2)
    if name == "bytetrack":
        return ByteTrackAdapter(track_thresh=0.3, track_buffer=max_age,
                                match_thresh=0.8,
                                frame_rate=int(round(fps)))
    if name == "sort":
        return SortAdapter(max_age=max_age, iou_threshold=0.3, min_hits=1)
    if name == "motion":
        return MotionTracker(max_age=max_age, iou_gate=0.2, assoc_frac=0.2,
                             gap_frac=0.5, gap_max_frac=1.0, vel_reset=0.4,
                             pred_cap=0.5)
    if name == "appearance":
        return MotionTracker(max_age=max_age, iou_gate=0.2, assoc_frac=0.2,
                             gap_frac=0.5, gap_max_frac=1.0, vel_reset=0.4,
                             pred_cap=0.5, appearance_weight=appearance_weight)
    raise ValueError(name)


def replay(tracker_name, dets_frames, fps, w, h, cadence, contain_iou=0.0,
           max_age=40, appearance_weight=0.5, confirm_frames=1):
    tracker = make_tracker(tracker_name, max_age=max_age, fps=fps,
                           appearance_weight=appearance_weight)
    gate = CrossingGate(min_age=1, margin_frac=0.06, rearm_frac=0.2,
                        confirm_frac=0.2, probe_edge=PROBE)
    line = line_px(LINE_NORM, w, h)
    gate_ids, events, objects = set(), [], []
    first_seen = {}
    stable_side, stable_run = {}, {}

    for i, frame_dets in enumerate(dets_frames):
        t = i / fps
        detect_now = (i % cadence == 0)
        feed = suppress_contained(frame_dets, contain_iou) if detect_now else []
        if tracker_name in ("ocsort-detonly", "bytetrack"):
            if detect_now:
                objects = tracker.update(feed, w, h, now=t)
        else:
            objects = tracker.update(feed, w, h, now=t)

        for obj in objects:
            first_seen.setdefault(obj.track_id, (t, obj.centroid))
        present = {o.track_id for o in objects}
        gate.drop(gate_ids - present)
        gate_ids = present

        # Detection-only ByteTrack-style mode: carry the last measured objects,
        # but do not re-apply the gate between detections.
        if tracker_name in ("ocsort-detonly", "bytetrack") and not detect_now:
            continue

        for obj in objects:
            if obj.prev_centroid is None or obj.age < 1:
                continue
            # Optional multi-frame confirmation: a track that teleports across
            # the line (an ID switch) is not allowed to commit on its first
            # measured frame on the new side.
            if confirm_frames > 1:
                pt = gate._probe(obj)
                dist, _ = line_geometry(line, pt)
                if dist is not None:
                    side = 1 if dist >= 0 else -1
                    margin = gate.margin_frac * h
                    if abs(dist) < margin:
                        stable_side[obj.track_id] = side
                        stable_run[obj.track_id] = 0
                    else:
                        if stable_side.get(obj.track_id) == side:
                            stable_run[obj.track_id] = stable_run.get(obj.track_id, 0) + 1
                        else:
                            stable_side[obj.track_id] = side
                            stable_run[obj.track_id] = 1
                    prev_side = gate.side.get(obj.track_id)
                    confirm = (gate.confirm_frac * h if gate.confirm_frac > 0
                               else margin)
                    if (prev_side is not None and side != prev_side
                            and abs(dist) >= confirm
                            and stable_run.get(obj.track_id, 0) < confirm_frames):
                        continue
            crossing = gate.update(line, obj, flip=FLIP, frame_h=h,
                                   now=t, frame_w=w)
            if crossing == CROSSING_IN:
                events.append(("IN", round(t, 2), obj.track_id,
                               tuple(round(float(v), 1) for v in obj.centroid)))
            elif crossing == CROSSING_OUT:
                events.append(("OUT", round(t, 2), obj.track_id,
                               tuple(round(float(v), 1) for v in obj.centroid)))

    return events, first_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    ap.add_argument("--cadences", default="1,2,3,4,5,7")
    ap.add_argument("--trackers", default="centroid,ocsort,sort,bytetrack")
    ap.add_argument("--limit", type=int, default=0,
                    help="only read the first N frames (smoke test)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--contain-iou", type=float, default=0.0,
                    help="drop lower-confidence boxes mostly inside a kept box "
                         "(intersection/min-area; 0 disables)")
    ap.add_argument("--max-age", type=int, default=40,
                    help="coasted reader frames before a track is retired")
    ap.add_argument("--appearance-weight", type=float, default=0.5,
                    help="HSV-histogram cost weight for the appearance tracker")
    ap.add_argument("--confirm-frames", type=int, default=1,
                    help="measured frames required on the new side before a crossing")
    args = ap.parse_args()

    cadences = [int(x) for x in args.cadences.split(",") if x]
    trackers = [x for x in args.trackers.split(",") if x]

    model = os.path.join(ROOT, "models", "picodet_s_320.ncnn.param")
    detector = PicodetNcnn(model_param=model, conf_thresh=0.3,
                           nms_thresh=0.3, num_threads=args.threads,
                           input_size=320)
    classes = [2, 5, 7]

    rows = []
    for clip in args.clips:
        path = clip if os.path.isabs(clip) else os.path.join(ROOT, clip)
        print(f"[bench] detections: {clip}", flush=True)
        fps, w, h, dets_frames = load_dets(path, detector, classes, args.limit)
        print(f"[bench] {w}x{h} @ {fps:.1f} fps, {len(dets_frames)} frames",
              flush=True)
        for cadence in cadences:
            for name in trackers:
                events, first = replay(name, dets_frames, fps, w, h, cadence,
                                       contain_iou=args.contain_iou,
                                       max_age=args.max_age,
                                       appearance_weight=args.appearance_weight,
                                       confirm_frames=args.confirm_frames)
                counts = {"IN": 0, "OUT": 0}
                for d, _t, _tid, _c in events:
                    counts[d] += 1
                label = name
                if args.contain_iou > 0:
                    label = f"{name}+cont{args.contain_iou:g}"
                ids = tuple(sorted({e[2] for e in events}))
                lpx = line_px(LINE_NORM, w, h)
                id_ok = all(
                    tid in first and starts_on_expected_side(d, first[tid][1], lpx)
                    for d, _t, tid, _c in events)
                rows.append((clip, cadence, label, counts, events, ids, id_ok))

    print("\nclip                         cad  tracker          IN OUT  ids     idok  events")
    print("-" * 112)
    for clip, cadence, name, counts, events, ids, id_ok in rows:
        print(f"{clip:<28} {cadence:>3}  {name:<15} "
              f"{counts['IN']:>2} {counts['OUT']:>3}  {str(ids):<7} {str(id_ok):<5} {events}")

    if args.limit:
        print("\n[bench] --limit set; expectations not evaluated")
        return

    bad = []
    for clip, cadence, name, counts, events, ids, id_ok in rows:
        exp = EXPECTED.get(clip)
        exp_ids = EXPECTED_IDS.get(clip)
        if exp and (counts["IN"] != exp["IN"] or counts["OUT"] != exp["OUT"]
                    or (exp_ids is not None and len(ids) != exp_ids)
                    or not id_ok):
            bad.append((clip, cadence, name, counts, exp, events, ids, id_ok))

    print("\n[bench] mismatches:")
    if not bad:
        print("  none")
    for clip, cadence, name, counts, exp, events, ids, id_ok in bad:
        print(f"  {clip} cadence={cadence} {name}: got {counts} ids={ids} "
              f"id_ok={id_ok}, expected {exp} with "
              f"{EXPECTED_IDS.get(clip)} distinct id(s), events={events}")

    algos = sorted({r[2].split("+")[0] for r in rows})
    print("\n[bench] per-algorithm summary:")
    for algo in algos:
        ab = [b for b in bad if b[2].split("+")[0] == algo]
        print(f"  {algo:<10} {'PASS' if not ab else f'FAIL ({len(ab)} case(s))'}")


if __name__ == "__main__":
    main()
