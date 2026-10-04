#!/usr/bin/env python3
"""Smoke-test the detector/tracker against an RTSP stream.

    python tools/test_stream.py --url "rtsp://admin:pass@host:554/unicast/c1/s0/live" --seconds 20

Forces RTSP-over-TCP, skips flat decoder warm-up frames, runs NanoDet + ByteTrack,
prints stats and writes the best annotated frame.
"""
import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import YoloNcnn, BYTETracker, _VEHICLE_NAMES

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def draw(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", default="2,3,5,7")
    ap.add_argument("--roi", help="ROI as JSON: '[[x,y],[x,y],[x,y],[x,y]]'")
    ap.add_argument("--out", default="/tmp/opencode/stream_sample.jpg")
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",") if c != ""]
    roi = json.loads(args.roi) if args.roi else None

    det = YoloNcnn(conf_thresh=args.conf)
    trk = BYTETracker(track_thresh=args.conf, track_buffer=50, match_thresh=0.7, frame_rate=25)
    print(f"[test] opening {args.url.split('@')[-1]}")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[test] FAILED to open stream"); sys.exit(1)
    print(f"[test] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # Warm up until the decoder emits a non-flat frame.
    warm = 0
    while warm < 150:
        ok, frame = cap.read()
        warm += 1
        if ok and float(frame.std()) > 15:
            break
    print(f"[test] warm-up frames={warm} flat={'yes' if not ok or float(frame.std()) <= 15 else 'no'}")

    t0 = time.time(); frames = 0; det_total = 0; hist = collections.Counter()
    best = (0, None, [])
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok or float(frame.std()) <= 5:
            continue
        frames += 1
        raw = det.detect_roi(frame, roi, classes) if roi else det.detect(frame, classes)
        det_total += len(raw)
        for d in raw:
            hist[d["label"]] += 1
        if raw:
            arr = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                             d["confidence"], d["class_id"]] for d in raw], dtype=np.float32)
            trk.update(arr, None)
            if len(raw) > len(best[2]):
                best = (frames, frame.copy(), raw)
    cap.release()

    dt = max(time.time() - t0, 1e-6)
    print(f"[test] frames={frames} time={dt:.1f}s read+detect={frames/dt:.2f} FPS")
    print(f"[test] detections={det_total} avg/frame={det_total/max(frames,1):.2f} classes={dict(hist)}")
    if best[1] is not None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        cv2.imwrite(args.out, draw(best[1], best[2]))
        print(f"[test] best frame @{best[0]} with {len(best[2])} dets -> {args.out}")


if __name__ == "__main__":
    main()
