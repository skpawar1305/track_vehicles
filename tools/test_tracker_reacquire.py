#!/usr/bin/env python3
"""Regression: a track must re-acquire after an occlusion gap.

The detector can stop returning a box for a vehicle for a second or more while
it passes behind another vehicle. The tracker coasts, but its velocity is often
stale by the time the box reappears, so the new detection can be far from the
predicted position. The fallback gate must widen with the *real* measurement
gap; otherwise the detection is treated as a new track — and on this site a new
track born past the counting line has no opposite-side history, so the crossing
is silently lost.

Run: python tools/test_tracker_reacquire.py
"""
import os
import sys

os.environ.setdefault("TRACKER_RELAY_REEXEC", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import CentroidTracker  # noqa: E402

W, H = 640, 360
FPS = 25.0
BOX_W, BOX_H, BOX_Y = 90, 80, 150
X0, SPEED, MOVING_S = 300, 20.0, 1.0
OCCLUDED_S = 1.6
X_REAPPEAR = 500


def det(x):
    return {
        "bbox": (x, BOX_Y, x + BOX_W, BOX_Y + BOX_H),
        "centroid": (x + BOX_W / 2, BOX_Y + BOX_H / 2),
        "confidence": 0.9,
        "class_id": 7,
        "label": "truck",
    }


def main():
    tr = CentroidTracker(max_age=40, iou_gate=0.2, assoc_frac=0.2,
                        gap_frac=0.5, gap_max_frac=1.0, vel_reset=0.4,
                        pred_cap=0.5)
    first_id = None
    ids = set()

    moving_frames = int(MOVING_S * FPS)
    for frame in range(moving_frames):
        t = frame / FPS
        objs = tr.update([det(int(X0 + SPEED * t))], W, H, now=t)
        present = {o.track_id for o in objs}
        ids |= present
        if first_id is None:
            first_id = objs[0].track_id

    last_seen_t = (moving_frames - 1) / FPS
    occluded_frames = int(OCCLUDED_S * FPS)
    for n in range(1, occluded_frames + 1):
        now = last_seen_t + n / FPS
        objs = tr.update([], W, H, now=now)
        ids |= {o.track_id for o in objs}

    now = last_seen_t + (occluded_frames + 1) / FPS
    reacquired = tr.update([det(X_REAPPEAR)], W, H, now=now)
    got = {o.track_id for o in reacquired}

    assert first_id is not None, "no initial track"
    assert got == {first_id}, (
        f"track was not re-acquired: initial id {first_id}, "
        f"ids seen {sorted(ids)}, re-appeared as {sorted(got)}")
    assert len(ids) == 1, f"occlusion minted extra ids: {sorted(ids)}"
    print(f"PASS: track #{first_id} re-acquired after {OCCLUDED_S:.1f}s occlusion")


if __name__ == "__main__":
    main()
