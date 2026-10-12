#!/usr/bin/env python3
"""Synthetic check for the crossing gate, with an annotated video showing the
hysteresis band and per-track re-arm state.

It feeds fabricated tracks (no camera, no detector) through the *production*
``CrossingGate`` and asserts that only genuine pass-throughs count:

  wobble      cross IN, then bob back and forth across the line, then continue
              -> 1 event (many with re-arm off — the old bug)
  passthrough straight IN and off              -> 1 event
  true_return drive IN, then genuinely come back OUT -> 2 events
  nudge       cross, overshoot only slightly, drift back, then continue -> 1
  rapid       poke just past the line, return, and stop -> 0 (confirm distance)
  queue       two tracks passing one after another -> 2 events

The *confirmed transit* (TRACKER_CONFIRM_FRAC) is what makes the rapid/nudge
cases zero: clearing the small hysteresis band is no longer enough — the
centroid must reach ``confirm_frac`` of the frame height past the line before a
crossing commits, and a return before that cancels it.

    python tools/test_crossing.py --out /tmp/crossing_demo.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("TRACKER_RELAY_REEXEC", "1")   # keep the import hermetic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import CrossingGate, CROSSING_IN, CROSSING_OUT, draw_gate_overlay

FPS = 25
CLOCK0 = 1000.0         # simulated wall clock (absolute, like time.time())
MARGIN_FRAC = 0.06
CONFIRM_FRAC = 0.2
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


def rapid_reversal_ys():
    """A working machine: poke just past the hysteresis band (but well short of
    the confirm distance), then return and stop. Must count 0 — before the
    confirmed transit this committed a spurious OUT/IN pair."""
    return linspace(0.60 * H, 0.68 * H, 8) + linspace(0.68 * H, 0.45 * H, 10)


def simulate(ys, rearm_frac=0.2, confirm_frac=CONFIRM_FRAC, cx=CX):
    """Run one track through the gate (mirrors run.py's per-frame checks)."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac, confirm_frac=confirm_frac)
    events = []
    for i, y in enumerate(ys):
        frame = i + 1
        obj = Obj(1, cx, y)
        obj.age = frame
        cr = gate.update(LINE, obj, frame_h=H)
        if cr in (CROSSING_IN, CROSSING_OUT):
            events.append((round(frame / FPS, 2), "IN" if cr == CROSSING_IN else "OUT", frame))
    return events


def simulate_queue(rearm_frac=0.2, confirm_frac=CONFIRM_FRAC):
    """Two tracks crossing one after another (a queue). Both must count, so the
    follower is not collapsed into the leader."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac, confirm_frac=confirm_frac)
    ys = {1: passthrough_ys(), 2: [0.34 * H] * 20 + passthrough_ys()}
    events = []
    for i in range(max(len(v) for v in ys.values())):
        frame = i + 1
        for tid, track in ys.items():
            if i >= len(track):
                continue
            obj = Obj(tid, CX, track[i])
            obj.age = max(1, frame - (20 if tid == 2 else 0))
            cr = gate.update(LINE, obj, frame_h=H)
            if cr in (CROSSING_IN, CROSSING_OUT):
                events.append((tid, round(frame / FPS, 2),
                               "IN" if cr == CROSSING_IN else "OUT"))
    return events


def render(path, rearm_frac, ys):
    """Same run, drawn frame-by-frame with the real gate (band + anchor + state)."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac, confirm_frac=CONFIRM_FRAC)
    margin = MARGIN_FRAC * H
    confirm = CONFIRM_FRAC * H
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    c_in = c_out = 0
    for i, y in enumerate(ys):
        frame = i + 1
        obj = Obj(1, CX, y)
        obj.age = frame
        cr = gate.update(LINE, obj, frame_h=H)
        if cr == CROSSING_IN:
            c_in += 1
        elif cr == CROSSING_OUT:
            c_out += 1

        img = np.full((H, W, 3), 22, np.uint8)
        # Confirm-distance band (dashed) outside the hysteresis band.
        for off in (confirm, -confirm):
            yy = int(LINE[1] + off)
            for x in range(0, W, 8):
                cv2.line(img, (x, yy), (min(W, x + 4), yy), (90, 90, 90), 1)
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
        cv2.putText(img, f"confirm frac {CONFIRM_FRAC:.2f}   frame {frame}/{len(ys)}",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"[demo] annotated -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tracker_crossing_demo.mp4")
    ap.add_argument("--rearm-frac", type=float, default=0.2)
    ap.add_argument("--confirm-frac", type=float, default=CONFIRM_FRAC)
    args = ap.parse_args()

    dirs = lambda evs: [d for _, d, _ in evs]
    wobble = simulate(wobble_ys(), args.rearm_frac, args.confirm_frac)
    wobble_off = simulate(wobble_ys(), 0.0, 0.0)
    pass_ = simulate(passthrough_ys(), args.rearm_frac, args.confirm_frac)
    ret = simulate(true_return_ys(), args.rearm_frac, args.confirm_frac)
    nudge = simulate(nudge_return_ys(), args.rearm_frac, args.confirm_frac)
    nudge_old = simulate(nudge_return_ys(), 0.15, 0.0)
    rapid = simulate(rapid_reversal_ys(), args.rearm_frac, args.confirm_frac)
    queue = simulate_queue(args.rearm_frac, args.confirm_frac)

    print(f"wobble   (confirm={args.confirm_frac}): {len(wobble):2d} events {dirs(wobble)}")
    print(f"wobble   (legacy)     : {len(wobble_off):2d} events {dirs(wobble_off)}")
    print(f"passthru (confirm={args.confirm_frac}): {len(pass_):2d} events {dirs(pass_)}")
    print(f"return   (confirm={args.confirm_frac}): {len(ret):2d} events {dirs(ret)}")
    print(f"nudge    (confirm={args.confirm_frac}): {len(nudge):2d} events {dirs(nudge)}")
    print(f"nudge    (legacy)     : {len(nudge_old):2d} events {dirs(nudge_old)}  (the double-count bug)")
    print(f"rapid    (confirm={args.confirm_frac}): {len(rapid):2d} events {dirs(rapid)}")
    print(f"queue    (confirm={args.confirm_frac}): {len(queue)} events "
          f"{[(t, d) for t, _, d in queue]}")

    ok = (len(wobble) == 1 and len(wobble_off) > 1 and len(pass_) == 1
          and len(ret) == 2 and len(nudge) == 1 and len(nudge_old) > 1
          and len(rapid) == 0 and len(queue) == 2)
    if args.out:
        render(args.out, args.rearm_frac, wobble_ys())
    if not ok:
        print("FAIL: crossing behaviour unexpected")
        sys.exit(1)
    print("OK: confirmed transit counts real passes, ignores wobble/rapid reversals, "
          "keeps a queued pair")


if __name__ == "__main__":
    main()
