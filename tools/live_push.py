#!/usr/bin/env python3
"""Push an annotated live feed from an RTSP source to the VPS dashboard.

Best-effort by design: a reader thread always keeps only the newest frame, the
detect loop runs at a fixed rate on that newest frame, and a background sender
thread drops stale frames if the uplink is slow. Nothing blocks the camera read,
so the feed stays live instead of slowly drifting behind real time.

Useful for testing before the Pi is deployed.

    python tools/live_push.py \
        --url "rtsp://admin:pass@host:554/unicast/c11/s0/live" \
        --live "https://tracker.drnanoinc.com/api/live" --token XXX
"""
import argparse
import collections
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from queue import Queue, Empty

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bytetracker import BYTETracker
from run import (YoloNcnn, MotionDetector, HybridDetector, TrackInfo, annotate_live,
                 annotate_frame, _VEHICLE_NAMES, MAX_DETECTIONS_PER_FRAME,
                 detect_crossing, segment_crosses_line, bbox_touches_line, CrossingGate,
                 CROSSING_NONE, CROSSING_IN, CROSSING_OUT)


class LatestFrame:
    """Thread-safe holder for exactly the most recent frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._at = 0.0
        self.seq = 0

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._at = time.time()
            self.seq += 1

    def get(self):
        with self._lock:
            return self._frame, self._at, self.seq


def reader(url, latest, stop):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[live] FAILED to open stream")
        stop.set()
        return
    print("[live] stream connected")
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            print("[live] stream lost, reconnecting…")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        latest.put(frame.copy())   # hand over an owned buffer; the decoder reuses memory
    cap.release()


def config_poller(url, headers, stop, cam, lock):
    s = requests.Session()
    while not stop.is_set():
        try:
            r = s.get(url, headers=headers, timeout=5)
            if r.ok:
                d = r.json()
                with lock:
                    cam["line"] = d.get("line")
                    cam["roi"] = d.get("roi")
                    cam["flip"] = bool(d.get("flip_sides", False))
                    cam["live_wanted"] = bool(d.get("live_wanted", False))
                    cam["counts"] = d.get("counts") or cam["counts"]
                    cam["updated_at"] = d.get("updated_at")
        except Exception:
            # Never let a transient/non-JSON response kill the poller; if this
            # thread dies, live_wanted freezes and the feed goes black.
            pass
        time.sleep(2)


def sender(url, headers, q, stop, stats):
    s = requests.Session()
    while not stop.is_set():
        try:
            jpeg = q.get(timeout=1.0)
        except Empty:
            continue
        try:
            r = s.post(url, data=jpeg, headers=headers, timeout=5)
            stats["sent"] += 1
            if stats["sent"] % 50 == 1:
                print(f"[live] sent={stats['sent']} bytes={len(jpeg)} -> {r.status_code}")
        except Exception as e:
            stats["failed"] += 1
            print("[live] post failed:", e)


def detect_worker(det, det_classes, latest, cam, cam_lock, shared, stop):
    """Run detection as fast as the CPU allows on the newest frame.

    Decoupled from tracking so line-crossing is evaluated at the camera's full
    sample rate, while YOLO only refreshes boxes/labels as often as it can.
    """
    last = -1
    while not stop.is_set():
        frame, _at, seq = latest.get()
        if frame is None or seq == last:
            time.sleep(0.005)
            continue
        last = seq
        if float(frame.std()) <= 5:
            continue
        with cam_lock:
            nr = cam["roi"]
        fh, fw = frame.shape[:2]
        roi_px = ([[int(p[0] * fw), int(p[1] * fh)] for p in nr]
                  if nr and len(nr) >= 3 else None)
        try:
            raw = (det.detect_roi(frame, roi_px, det_classes) if roi_px
                   else det.detect(frame, det_classes))
        except Exception as e:
            print("[live] detect error:", e)
            continue
        if len(raw) > MAX_DETECTIONS_PER_FRAME:
            print(f"[live] implausible detections ({len(raw)}) — glitch, ignoring")
            raw = []
        with shared["lock"]:
            shared["dets"] = raw
            shared["counter"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--fps", type=float, default=10.0, help="tracking loop rate")
    ap.add_argument("--live-fps", type=float, default=5.0, help="live upload rate")
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--quality", type=int, default=65)
    ap.add_argument("--config", default=None, help="config endpoint (default: derive from --live)")
    ap.add_argument("--detector", choices=["hybrid", "motion", "yolo"], default="hybrid")
    args = ap.parse_args()
    config_url = args.config or args.live.replace("/api/live", "/api/config")
    ingest_url = args.live.replace("/api/live", "/api/ingest")

    if args.detector == "motion":
        det = MotionDetector()
        det_classes = None
    elif args.detector == "yolo":
        det = YoloNcnn(conf_thresh=args.conf)
        det_classes = [2, 5, 7]
    else:
        det = HybridDetector(conf_thresh=args.conf)
        det_classes = [2, 5, 7]
    print(f"[live] detector={args.detector}")
    # frame_rate only sets the lost-track buffer size (buffer = frame_rate/30 * track_buffer);
    # keep frame_rate=30 so track_buffer is in processed frames (~8s at 5fps).
    trk = BYTETracker(track_thresh=args.conf, track_buffer=40, match_thresh=0.85, frame_rate=30)
    headers = {"Authorization": f"Bearer {args.token}"}
    track_interval = 1.0 / args.fps
    live_interval = 1.0 / args.live_fps

    latest = LatestFrame()
    stop = threading.Event()
    q = Queue(maxsize=1)
    stats = {"sent": 0, "failed": 0}
    cam = {"line": None, "roi": None, "flip": False, "live_wanted": False,
           "counts": {"in": 0, "out": 0}, "updated_at": None}
    cam_lock = threading.Lock()
    shared = {"dets": [], "counter": 0, "lock": threading.Lock()}

    threading.Thread(target=reader, args=(args.url, latest, stop), daemon=True).start()
    threading.Thread(target=sender, args=(args.live, headers, q, stop, stats), daemon=True).start()
    threading.Thread(target=config_poller,
                     args=(config_url, headers, stop, cam, cam_lock), daemon=True).start()
    threading.Thread(target=detect_worker,
                     args=(det, det_classes, latest, cam, cam_lock, shared, stop), daemon=True).start()

    prev, ages = {}, {}
    gate = CrossingGate(min_age=2, min_travel_frac=0.02)
    debug = os.environ.get("TRACKER_DEBUG") == "1"
    last_seq = -1
    last_track = 0.0
    last_live = 0.0
    used_det = -1
    print(f"[live] {args.url.split('@')[-1]} -> {args.live} @ track {args.fps}fps / live {args.live_fps}fps")

    try:
        while True:
            frame, _at, seq = latest.get()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            now = time.time()
            if now - last_track < track_interval:
                time.sleep(0.005)
                continue
            last_track = now
            last_seq = seq

            if float(frame.std()) <= 5:      # HEVC warm-up / dead frames
                continue

            with cam_lock:
                nl, nr, wanted, flip = cam["line"], cam["roi"], cam["live_wanted"], cam["flip"]
                cnts = dict(cam["counts"])
            fh, fw = frame.shape[:2]
            line_px = None
            if nl and len(nl) >= 2:
                line_px = (int(nl[0][0] * fw), int(nl[0][1] * fh),
                           int(nl[1][0] * fw), int(nl[1][1] * fh))
            roi_px = ([[int(p[0] * fw), int(p[1] * fh)] for p in nr]
                      if nr and len(nr) >= 3 else None)

            # Consume fresh YOLO detections if the worker produced any; otherwise
            # feed empty detections so ByteTrack coasts (Kalman) between updates.
            with shared["lock"]:
                dcount = shared["counter"]
                raw = list(shared["dets"])
            if dcount == used_det:
                raw = []
            else:
                used_det = dcount
            if raw:
                arr = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                                 d["confidence"], d["class_id"]] for d in raw], dtype=np.float32)
            else:
                arr = np.empty((0, 6), dtype=np.float32)
            tracked = trk.update(arr, None)

            objects = []
            for t in tracked:
                x1, y1, x2, y2, tid, cls_id, score = t
                x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
                cen = ((x1 + x2) // 2, (y1 + y2) // 2)
                objects.append(TrackInfo(tid, (x1, y1, x2, y2),
                                         _VEHICLE_NAMES.get(int(cls_id), f"cls_{int(cls_id)}"),
                                         int(cls_id), float(score), cen, prev.get(tid), ages.get(tid, 0)))
                prev[tid] = cen
                ages[tid] = ages.get(tid, 0) + 1
                gate.update_first(tid, cen)

            if debug and objects:
                print("[dbg] " + ", ".join(
                    f"{o.label}#{o.track_id} age={o.age} c={o.centroid} prev={o.prev_centroid}"
                    for o in objects[:8]))

            # Line crossing -> capture + upload event (same logic as run.py)
            for obj in objects:
                if not (line_px and obj.age >= 2):
                    continue
                crossing = gate.crossing(line_px, obj, flip=flip, margin=0.06 * fh)
                if debug:
                    print(f"[dbg] #{obj.track_id} {obj.label} cross={crossing} "
                          f"age={obj.age} side={gate.side.get(obj.track_id)}")
                if crossing == CROSSING_NONE:
                    continue
                if not gate.allow(obj, now, fh):
                    continue
                direction = "in" if crossing == CROSSING_IN else "out"
                # Raw frame (no overlay) as training data
                ok, cj = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    continue
                th = cv2.resize(frame, (640, max(1, int(frame.shape[0] * 640 / frame.shape[1]))),
                                interpolation=cv2.INTER_AREA)
                _, tj = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 80])
                eid = f"test_{int(now * 1000)}_{obj.track_id}"
                fields = {
                    "id": eid, "track_id": obj.track_id, "class_id": obj.class_id,
                    "label": obj.label, "confidence": obj.confidence, "direction": direction,
                    "crossed_at": datetime.now(timezone.utc).isoformat(),
                    "bbox": json.dumps(list(obj.bbox)), "line": json.dumps(list(line_px)),
                }
                try:
                    requests.post(ingest_url, data=fields,
                                  files={"image": (eid + ".jpg", cj.tobytes(), "image/jpeg"),
                                         "thumb": (eid + ".jpg", tj.tobytes(), "image/jpeg")},
                                  headers=headers, timeout=8)
                    print(f"[live] EVENT {direction.upper()} {obj.label} #{obj.track_id}")
                except requests.RequestException as e:
                    print("[live] ingest failed:", e)

            if not wanted or (now - last_live) < live_interval:
                continue
            last_live = now
            ann = annotate_live(frame, line_px, objects, cnts,
                                target_h=args.height, roi=roi_px, flip=flip)
            ok, jpeg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue
            try:
                q.put_nowait(jpeg.tobytes())
            except Exception:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
