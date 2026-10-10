#!/usr/bin/env python3
"""Synthetic check for the crossing gate's re-arm logic, with an annotated
video showing the hysteresis band and per-track re-arm state.

It feeds a fabricated track (no camera, no detector) through the *production*
``CrossingGate`` and asserts that a vehicle which only "goes a bit back" at the
line is counted once:

  wobble      cross IN, then bob back and forth across the line, then continue
              -> 1 event with re-arm on; many events with re-arm off (the bug)
  passthrough straight IN and off              -> 1 event
  true_return drive IN, then genuinely come back OUT -> 2 events

    python tools/test_crossing.py --out /tmp/crossing_demo.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("TRACKER_RELAY_REEXEC", "1")   # keep the import hermetic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import CrossingGate, CROSSING_NONE, CROSSING_IN, draw_gate_overlay

FPS = 25
CLOCK0 = 1000.0         # simulated wall clock (absolute, like time.time())
MARGIN_FRAC = 0.06
W, H = 640, 360
LINE = (int(0.05 * W), int(0.60 * H), int(0.95 * W), int(0.60 * H))
CX = W * 0.5


class Obj:
    __slots__ = ("track_id", "centroid", "age", "bbox", "label", "class_id", "confidence")

    def __init__(self, tid, cx, cy, w=64, h=48):
        self.track_id = tid
        self.centroid = (float(cx), float(cy))
        self.bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        self.age = 1
        self.label = "truck"
        self.class_id = 7
        self.confidence = 0.8


def linspace(a, b, n):
    n = max(2, n)
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def wobble_ys():
    """Cross the line, bob across it by less than the re-arm distance, continue."""
    ys = linspace(0.34 * H, 0.665 * H, 25)
    for _ in range(4):
        ys += linspace(0.665 * H, 0.535 * H, 65)
        ys += linspace(0.535 * H, 0.665 * H, 65)
    ys += linspace(0.665 * H, 0.92 * H, 20)
    return ys


def passthrough_ys():
    return linspace(0.34 * H, 0.92 * H, 45)


def true_return_ys():
    return linspace(0.34 * H, 0.92 * H, 45) + linspace(0.92 * H, 0.30 * H, 45)


def nudge_return_ys():
    """Cross the line, overshoot by only ~0.15*H, drift back over the line, then
    continue. A slow/loitering vehicle (e.g. the 12:54 dumper) must count once:
    its small overshoot must not satisfy the re-arm distance."""
    return (linspace(0.34 * H, 0.665 * H, 20)
            + linspace(0.665 * H, 0.50 * H, 30)
            + linspace(0.50 * H, 0.90 * H, 25))


def simulate(rearm_frac, ys):
    """Run one track through the gate (mirrors run.py's per-frame checks)."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac)
    events = []
    for i, y in enumerate(ys):
        frame = i + 1
        obj = Obj(1, CX, y)
        obj.age = frame
        cr = gate.crossing(LINE, obj, frame_h=H)
        if cr != CROSSING_NONE and gate.allow(obj, H):
            events.append((round(frame / FPS, 2), "IN" if cr == CROSSING_IN else "OUT", frame))
    return events


def render(path, rearm_frac, ys):
    """Same run, drawn frame-by-frame with the real gate (band + anchor + state)."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac)
    margin = MARGIN_FRAC * H
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    c_in = c_out = 0
    for i, y in enumerate(ys):
        frame = i + 1
        obj = Obj(1, CX, y)
        obj.age = frame
        cr = gate.crossing(LINE, obj, frame_h=H)
        if cr != CROSSING_NONE and gate.allow(obj, H):
            if cr == CROSSING_IN:
                c_in += 1
            else:
                c_out += 1

        img = np.full((H, W, 3), 22, np.uint8)
        cv2.line(img, LINE[:2], LINE[2:], (59, 130, 246), 2)
        cv2.putText(img, "IN", (LINE[0], LINE[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (34, 197, 94), 2, cv2.LINE_AA)
        cv2.putText(img, "OUT", (LINE[2] - 64, LINE[3] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (239, 68, 68), 2, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), 2)
        draw_gate_overlay(img, LINE, [obj], gate, H, margin, 1.0)
        cv2.putText(img, f"IN {c_in}  OUT {c_out}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (59, 130, 246), 2, cv2.LINE_AA)
        cv2.putText(img, f"re-arm frac {rearm_frac:.2f}   frame {frame}/{len(ys)}",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"[demo] annotated -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tracker_crossing_demo.mp4")
    ap.add_argument("--rearm-frac", type=float, default=0.2)
    args = ap.parse_args()

    dirs = lambda evs: [d for _, d, _ in evs]
    wobble = simulate(args.rearm_frac, wobble_ys())
    wobble_off = simulate(0.0, wobble_ys())
    pass_ = simulate(args.rearm_frac, passthrough_ys())
    ret = simulate(args.rearm_frac, true_return_ys())
    nudge = simulate(args.rearm_frac, nudge_return_ys())
    nudge_old = simulate(0.15, nudge_return_ys())

    print(f"wobble   (rearm={args.rearm_frac}): {len(wobble):2d} events {dirs(wobble)}")
    print(f"wobble   (rearm=0.00): {len(wobble_off):2d} events {dirs(wobble_off)}")
    print(f"passthru (rearm={args.rearm_frac}): {len(pass_):2d} events {dirs(pass_)}")
    print(f"return   (rearm={args.rearm_frac}): {len(ret):2d} events {dirs(ret)}")
    print(f"nudge    (rearm={args.rearm_frac}): {len(nudge):2d} events {dirs(nudge)}")
    print(f"nudge    (rearm=0.15): {len(nudge_old):2d} events {dirs(nudge_old)}  (the double-count bug)")

    ok = (len(wobble) == 1 and len(wobble_off) > 1 and len(pass_) == 1
          and len(ret) == 2 and len(nudge) == 1 and len(nudge_old) > 1)
    if args.out:
        render(args.out, args.rearm_frac, wobble_ys())
    if not ok:
        print("FAIL: re-arm behaviour unexpected")
        sys.exit(1)
    print(f"OK: wobble counted once (was {len(wobble_off)} without re-arm); real passes/returns intact")


if __name__ == "__main__":
    main()
