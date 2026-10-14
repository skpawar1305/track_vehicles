#!/usr/bin/env python3
"""Regression for running ByteTrack slower than the camera frame rate.

ByteTrack advances once per detector observation, not once per reader frame.
This reproduces the old reader-rate/empty-update failure and checks the fixed
detector-result cadence at about 5 fps.

    python tools/test_bytetrack_cadence.py
"""
import os
import sys

os.environ.setdefault("TRACKER_RELAY_REEXEC", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import (  # noqa: E402
    ByteTrackAdapter, CrossingGate, CROSSING_IN, CROSSING_OUT,
)
from bytetracker.byte_tracker import STrack  # noqa: E402


W, H = 640, 360
LINE = (500, 0, 500, H)
DETECT_EVERY = 5


def detection(left):
    box = (left, 120, left + 120, 220)
    return {
        "bbox": box,
        "centroid": (left + 60, 170),
        "confidence": 0.9,
        "class_id": 7,
        "label": "truck",
    }


def replay(update_every_reader_frame, old_tentative_activation=False):
    # Recreate the old ByteTrack behavior when comparing the broken cadence.
    original_activate = STrack.activate
    if old_tentative_activation:
        def tentative_activate(track, kalman_filter, frame_id):
            original_activate(track, kalman_filter, frame_id)
            track.is_activated = frame_id == 1
        STrack.activate = tentative_activate
    tracker = ByteTrackAdapter(track_thresh=0.3, track_buffer=40,
                               match_thresh=0.8, frame_rate=30)
    gate = CrossingGate(min_age=1, margin_frac=0.06, rearm_frac=0.2,
                        confirm_frac=0.2, probe_edge="left")
    gate_ids = set()
    events = []
    # The first detection arrives after ByteTrack has already advanced several
    # empty reader frames. Then the dumper crosses right-to-left over successive
    # detector results.
    positions = {10: 560, 15: 530, 20: 500, 25: 470, 30: 440, 35: 410}
    result = []
    try:
        for frame in range(40):
            has_detection = frame in positions
            if has_detection or update_every_reader_frame:
                dets = [detection(positions[frame])] if has_detection else []
                result = tracker.update(dets, W, H, now=frame / 25.0)
            if not has_detection:
                continue
            present = {obj.track_id for obj in result}
            gate.drop(gate_ids - present)
            gate_ids = present
            for obj in result:
                if obj.prev_centroid is None:
                    continue
                crossing = gate.update(LINE, obj, flip=True, frame_h=H,
                                       now=frame / 25.0, frame_w=W)
                if crossing == CROSSING_OUT:
                    events.append("OUT")
                elif crossing == CROSSING_IN:
                    events.append("IN")
        return events
    finally:
        STrack.activate = original_activate


def main():
    broken = replay(update_every_reader_frame=True, old_tentative_activation=True)
    fixed = replay(update_every_reader_frame=False)
    print(f"old reader-frame updates: {broken}")
    print(f"detector-result updates: {fixed}")
    assert broken == [], f"Expected old cadence to lose the crossing, got {broken}"
    assert fixed == ["OUT"], f"Expected one right-to-left OUT, got {fixed}"
    print("PASS: detector-cadence fix preserves the crossing")


if __name__ == "__main__":
    main()
