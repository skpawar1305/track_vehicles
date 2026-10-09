#!/usr/bin/env python3
"""Run the production detector + centroid tracker + crossing logic over a video
clip (file or stream) and report IN/OUT events.

Useful for tuning recall against known footage without touching the live edge.
Optionally pushes the annotated feed + events to a VPS dashboard so the test is
visible there (temporarily overriding the Pi's live feed).

    python tools/test_clip.py --file /tmp/opencode/traffic.mp4 --seconds 90 \
        --line 0.05,0.62,0.95,0.62 --out /tmp/opencode/traffic_ann.mp4

    # push to the dashboard (needs a bearer token):
    python tools/test_clip.py --file /tmp/opencode/traffic.mp4 --seconds 90 --realtime \
        --live-url https://tracker.drnanoinc.com/api/live --token XXX
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from queue import Queue, Empty

import cv2
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import YoloNcnn, CentroidTracker, CrossingGate, CROSSING_NONE, CROSSING_IN


def parse_line(s, w, h):
    """'x1,y1,x2,y2' in normalized 0..1 (or absolute if >1)."""
    x1, y1, x2, y2 = (float(v) for v in s.split(","))
    f = lambda v, d: int(v * d) if v <= 1.0 else int(v)
    return (f(x1, w), f(y1, h), f(x2, w), f(y2, h))


def sender(q, url, headers, stop):
    s = requests.Session()
    while not stop.is_set():
        try:
            jpeg = q.get(timeout=1.0)
        except Empty:
            continue
        try:
            s.post(url, data=jpeg, headers=headers, timeout=5)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="video file or stream URL")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--line", default="0.05,0.62,0.95,0.62")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--classes", default="2,5,7")
    ap.add_argument("--min-age", type=int, default=1)
    ap.add_argument("--min-travel-frac", type=float, default=0.02)
    ap.add_argument("--cross-margin-frac", type=float, default=0.015)
    ap.add_argument("--assoc-frac", type=float, default=0.2)
    ap.add_argument("--out", default="")
    ap.add_argument("--realtime", action="store_true", help="pace playback to video fps")
    ap.add_argument("--live-url", default="", help="VPS /api/live to push annotated frames")
    ap.add_argument("--token", default="", help="bearer token for the VPS")
    ap.add_argument("--push-fps", type=float, default=5.0)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",") if c != ""]
    det = YoloNcnn(conf_thresh=args.conf)
    tracker = CentroidTracker(assoc_frac=args.assoc_frac)
    gate = CrossingGate(min_age=args.min_age, min_travel_frac=args.min_travel_frac)

    cap = cv2.VideoCapture(args.file, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[clip] FAILED to open", args.file)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line = parse_line(args.line, w, h)
    print(f"[clip] {w}x{h} @ {fps:.1f} fps  line(px)={line}")

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    ingest_url = args.live_url.replace("/api/live", "/api/ingest") if args.live_url else ""
    stop = threading.Event()
    q = Queue(maxsize=1)
    if args.live_url:
        threading.Thread(target=sender, args=(q, args.live_url, headers, stop), daemon=True).start()
        print(f"[clip] pushing live -> {args.live_url} @ {args.push_fps} fps")

    last_cross_info, gate_ids = {}, set()
    events, c_in, c_out = [], 0, 0
    total, t0, last = 0, time.time(), -1.0
    play_t0 = time.time()
    push_interval = 1.0 / args.push_fps if args.push_fps > 0 else 10.0
    last_push = 0.0

    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        now = total / fps
        if args.realtime or args.live_url:
            target = play_t0 + total / fps
            delay = target - time.time()
            if delay > 0:
                time.sleep(delay)

        raw = det.detect(frame, classes)
        objects = tracker.update(raw, w, h)
        present = set()
        for o in objects:
            present.add(o.track_id)
            gate.update_first(o.track_id, o.centroid)
        gate.drop(gate_ids - present)
        gate_ids = present

        for obj in objects:
            if obj.age < args.min_age:
                continue
            crossing = gate.crossing(line, obj, margin=args.cross_margin_frac * h)
            if crossing == CROSSING_NONE:
                continue
            if not gate.allow(obj, now, h):
                continue
            info = last_cross_info.get(obj.track_id, {"frame": -60})
            if total - info["frame"] < 15:
                continue
            last_cross_info[obj.track_id] = {"frame": total}
            d = "IN" if crossing == CROSSING_IN else "OUT"
            events.append((round(now, 1), obj.track_id, obj.label, d, obj.centroid))
            if crossing == CROSSING_IN:
                c_in += 1
            else:
                c_out += 1
            if ingest_url:
                okj, cj = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                th = cv2.resize(frame, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
                _, tj = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 80])
                eid = f"test_{int(time.time() * 1000)}_{obj.track_id}"
                fields = {
                    "id": eid, "track_id": obj.track_id, "class_id": obj.class_id,
                    "label": obj.label, "confidence": obj.confidence, "direction": d.lower(),
                    "crossed_at": datetime.now(timezone.utc).isoformat(),
                    "bbox": json.dumps(list(obj.bbox)), "line": json.dumps(list(line)),
                }
                try:
                    requests.post(ingest_url, data=fields,
                                  files={"image": (eid + ".jpg", cj.tobytes(), "image/jpeg"),
                                         "thumb": (eid + ".jpg", tj.tobytes(), "image/jpeg")},
                                  headers=headers, timeout=8)
                except requests.RequestException:
                    pass

        # Annotate once: line + boxes + counts (used for both video out and live push)
        cv2.line(frame, line[:2], line[2:], (59, 130, 246), 2)
        for o in objects:
            x1, y1, x2, y2 = o.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cv2.putText(frame, f"{o.label}#{o.track_id} a{o.age}", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1)
        cv2.putText(frame, f"IN {c_in}  OUT {c_out}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2)

        if writer is not None:
            writer.write(frame)
        if args.live_url and time.time() - last_push >= push_interval:
            last_push = time.time()
            okj, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if okj:
                try:
                    q.put_nowait(jpeg.tobytes())
                except Exception:
                    pass

    stop.set()
    cap.release()
    if writer is not None:
        writer.release()
    print(f"[clip] frames={total}  crossings: in={c_in} out={c_out} total={len(events)}")
    for e in events:
        print("  cross", e)
    if args.out:
        print("[clip] annotated ->", args.out)


if __name__ == "__main__":
    main()
