#!/usr/bin/env python3
"""Synthetic check for ZoneGate (zone-to-zone directional counting).

Two rectangles are drawn on a 640x360 frame: an ENTRY zone (right) and an EXIT
zone (left). A track that moves from one committed zone to the other counts once;
wobble, a stop in the gap, a first sight already inside a zone, and anything
inside the keep-out count nothing.

    python tools/test_zones.py
"""
import os
import sys

os.environ.setdefault("TRACKER_RELAY_REEXEC", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import ZoneGate, CROSSING_IN, CROSSING_OUT

W, H = 640, 360


def rect(x0, y0, x1, y1):
    return [[int(x0 * W), int(y0 * H)], [int(x1 * W), int(y0 * H)],
            [int(x1 * W), int(y1 * H)], [int(x0 * W), int(y1 * H)]]


ENTRY = rect(0.55, 0.20, 0.97, 0.80)   # right box  (counted IN)
EXIT = rect(0.03, 0.20, 0.45, 0.80)    # left box   (counted OUT)
KEEP = rect(0.50, 0.10, 1.00, 0.90)    # covers ENTRY


class Obj:
    __slots__ = ("track_id", "centroid", "age", "bbox")

    def __init__(self, tid, cx, cy):
        self.track_id = tid
        self.centroid = (float(cx), float(cy))
        self.bbox = (cx - 30, cy - 20, cx + 30, cy + 20)
        self.age = 1


def run(path, keepout=None, tid=1):
    gate = ZoneGate(min_age=1, margin_frac=0.06, rearm_frac=0.2,
                    confirm_frac=0.2, keepout=keepout)
    events = []
    for i, x in enumerate(path):
        obj = Obj(tid, x, 0.5 * H)
        obj.age = i + 1
        cr = gate.update(ENTRY, EXIT, obj, frame_h=H)
        if cr in (CROSSING_IN, CROSSING_OUT):
            events.append("IN" if cr == CROSSING_IN else "OUT")
    return events


def lin(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def main():
    tx = lambda x: x * W
    cases = {
        "entry->exit (OUT)": (lin(tx(0.75), tx(0.25), 40), None, ["OUT"]),
        "exit->entry (IN)":  (lin(tx(0.25), tx(0.75), 40), None, ["IN"]),
        "wobble in entry":   (lin(tx(0.75), tx(0.80), 20) + lin(tx(0.80), tx(0.70), 20), None, []),
        "stop in gap":       (lin(tx(0.75), tx(0.50), 25) + lin(tx(0.50), tx(0.75), 20), None, []),
        "born in entry":     (lin(tx(0.70), tx(0.90), 20), None, []),
        "round trip":        (lin(tx(0.75), tx(0.25), 40) + lin(tx(0.25), tx(0.75), 40), None, ["OUT", "IN"]),
        "keepout freezes":   (lin(tx(0.25), tx(0.75), 40), KEEP, []),
    }
    ok = True
    for name, (path, keep, want) in cases.items():
        got = run(path, keep)
        good = got == want
        ok = ok and good
        print(f"{'ok ' if good else 'FAIL'} {name:20s} got={got} want={want}")
    if not ok:
        sys.exit(1)
    print("OK: zone transitions count once, gaps/wobble/keepout count nothing")


if __name__ == "__main__":
    main()
