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

    @classmethod
    def from_span(cls, tid, cx, top, bottom, w=64):
        """A box with an explicit top/bottom edge (so the top can flap — a
        machine's arm/load — while the ground-contact bottom stays put)."""
        obj = cls(tid, cx, (top + bottom) / 2, w, bottom - top)
        obj.bbox = (cx - w / 2, float(top), cx + w / 2, float(bottom))
        return obj


# The point fed to the production gate: the centroid (current prod) or the
# box's bottom edge (the ground-contact "tail", the Qengineering-style probe).
PROBE_CENTROID, PROBE_BOTTOM = "centroid", "bottom"


class _Probe:
    """Presents a chosen probe point to CrossingGate as the 'centroid'."""
    __slots__ = ("track_id", "centroid", "age", "bbox")

    def __init__(self, obj, point):
        self.track_id = obj.track_id
        self.centroid = point
        self.age = obj.age
        self.bbox = obj.bbox


def probe_obj(obj, probe):
    if probe == PROBE_BOTTOM:
        return _Probe(obj, (obj.centroid[0], obj.bbox[3]))
    return obj


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


def simulate(ys, rearm_frac=0.2, confirm_frac=CONFIRM_FRAC, cx=CX, probe=PROBE_CENTROID):
    """Run one track through the gate (mirrors run.py's per-frame checks)."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac, confirm_frac=confirm_frac)
    events = []
    for i, y in enumerate(ys):
        frame = i + 1
        obj = Obj(1, cx, y)
        obj.age = frame
        cr = gate.update(LINE, probe_obj(obj, probe), frame_h=H)
        if cr in (CROSSING_IN, CROSSING_OUT):
            events.append((round(frame / FPS, 2), "IN" if cr == CROSSING_IN else "OUT", frame))
    return events


def simulate_queue(rearm_frac=0.2, confirm_frac=CONFIRM_FRAC, probe=PROBE_CENTROID):
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
            cr = gate.update(LINE, probe_obj(obj, probe), frame_h=H)
            if cr in (CROSSING_IN, CROSSING_OUT):
                events.append((tid, round(frame / FPS, 2),
                               "IN" if cr == CROSSING_IN else "OUT"))
    return events


def refit_boxes():
    """A machine parked straddling the line: the ground-contact bottom edge is
    fixed at 0.83*H (below the line) while its arm/load flaps the top edge from
    well above the line to just under the bottom. The centroid is dragged right
    across the line (well past the confirm distance) and back, with no real
    transit — the durable false-count the keep-out was meant to stop. The bottom
    edge never moves."""
    bottom = 0.83 * H
    tops = linspace(0.22 * H, 0.80 * H, 30) + linspace(0.80 * H, 0.22 * H, 30)
    tops += linspace(0.22 * H, 0.80 * H, 30) + linspace(0.80 * H, 0.22 * H, 30)
    return [(top, bottom) for top in tops]


def shadow_boxes():
    """The bottom probe's own failure mode: the box's *lower* edge is
    unreliable — a reflection/shadow puddle extends it to the floor, or a
    detector contracts it up toward the roof — while the roof stays put. The
    centroid only moves half as far and stays inside the confirm distance; the
    bottom edge alone is dragged across the line."""
    top = 0.42 * H
    bottoms = (linspace(0.44 * H, 1.00 * H, 30)
               + linspace(1.00 * H, 0.44 * H, 30)
               + linspace(0.44 * H, 1.00 * H, 30))
    return [(top, bottom) for bottom in bottoms]


def simulate_boxes(boxes, rearm_frac=0.2, confirm_frac=CONFIRM_FRAC, cx=CX,
                   probe=PROBE_CENTROID):
    """Feed explicit (top, bottom) boxes through the gate."""
    gate = CrossingGate(min_age=1, rearm_frac=rearm_frac, confirm_frac=confirm_frac)
    events = []
    for i, (top, bottom) in enumerate(boxes):
        frame = i + 1
        obj = Obj.from_span(1, cx, top, bottom)
        obj.age = frame
        cr = gate.update(LINE, probe_obj(obj, probe), frame_h=H)
        if cr in (CROSSING_IN, CROSSING_OUT):
            events.append((round(frame / FPS, 2), "IN" if cr == CROSSING_IN else "OUT", frame))
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

    # ── Probe comparison: centroid (prod) vs the box's bottom / ground edge ──
    # On pure translations the two probes only differ by a constant offset, so
    # every genuine case must keep the same count. The refit case is where they
    # diverge: a machine parked on the line drags the *centroid* across (arm/
    # load refits the top edge) while the bottom edge never moves.
    def counts(evs):
        return [d for _, d, _ in evs]

    probe_cases = {
        "wobble": wobble_ys, "passthru": passthrough_ys, "return": true_return_ys,
        "nudge": nudge_return_ys, "rapid": rapid_reversal_ys,
    }
    same = True
    for name, fn in probe_cases.items():
        cc = counts(simulate(fn(), args.rearm_frac, args.confirm_frac, probe=PROBE_CENTROID))
        cb = counts(simulate(fn(), args.rearm_frac, args.confirm_frac, probe=PROBE_BOTTOM))
        same = same and cc == cb
        print(f"[probe] {name:9s} centroid={cc}  bottom={cb}")

    q_cent = sorted((t, d) for t, _, d in simulate_queue(args.rearm_frac, args.confirm_frac,
                                                         probe=PROBE_CENTROID))
    q_bot = sorted((t, d) for t, _, d in simulate_queue(args.rearm_frac, args.confirm_frac,
                                                        probe=PROBE_BOTTOM))
    same = same and q_cent == q_bot
    print(f"[probe] queue     centroid={q_cent}  bottom={q_bot}")

    boxes = refit_boxes()
    refit_c = simulate_boxes(boxes, args.rearm_frac, args.confirm_frac, probe=PROBE_CENTROID)
    refit_b = simulate_boxes(boxes, args.rearm_frac, args.confirm_frac, probe=PROBE_BOTTOM)
    print(f"refit    centroid ({args.confirm_frac}): {len(refit_c):2d} events {dirs(refit_c)}"
          f"   <- machine parked on the line must be 0")
    print(f"refit    bottom   ({args.confirm_frac}): {len(refit_b):2d} events {dirs(refit_b)}")

    sb = shadow_boxes()
    shadow_c = simulate_boxes(sb, args.rearm_frac, args.confirm_frac, probe=PROBE_CENTROID)
    shadow_b = simulate_boxes(sb, args.rearm_frac, args.confirm_frac, probe=PROBE_BOTTOM)
    print(f"shadow   centroid ({args.confirm_frac}): {len(shadow_c):2d} events {dirs(shadow_c)}")
    print(f"shadow   bottom   ({args.confirm_frac}): {len(shadow_b):2d} events {dirs(shadow_b)}"
          f"   <- bottom edge's own failure mode")

    ok = (len(wobble) == 1 and len(wobble_off) > 1 and len(pass_) == 1
          and len(ret) == 2 and len(nudge) == 1 and len(nudge_old) > 1
          and len(rapid) == 0 and len(queue) == 2
          and same and len(refit_c) >= 1 and len(refit_b) == 0
          and len(shadow_c) == 0 and len(shadow_b) >= 1)
    if args.out:
        render(args.out, args.rearm_frac, wobble_ys())
    if not ok:
        print("FAIL: crossing behaviour unexpected")
        sys.exit(1)
    print("OK: confirmed transit counts real passes, ignores wobble/rapid reversals, "
          "keeps a queued pair")


if __name__ == "__main__":
    main()
