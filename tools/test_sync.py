#!/usr/bin/env python3
"""Offline check for the sync worker's self-healing / no-silent-drop rules.

No network: ``post_event`` is stubbed so the assertions isolate the store and
pass logic. Run:

    python tools/test_sync.py

Covers:
  * ``Store.requeue_present_images`` re-queues a row marked ``synced=1`` whose
    image is *still on disk* (the "marked done while the VPS was unreachable"
    outage bug) and leaves a genuinely-delivered row (no image) alone.
  * a successful upload deletes the image and marks the row synced;
  * a pending row whose image is missing is retried, not silently dropped,
    and only given up once it is older than the grace window;
  * a network failure keeps the row and its image (never loses a capture).
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sync  # noqa: E402
from store import Store  # noqa: E402


def iso(hours_ago=0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def make_img(cap, name):
    with open(os.path.join(cap, name), "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")


def main():
    tmp = tempfile.mkdtemp()
    try:
        cap = os.path.join(tmp, "captures")
        os.makedirs(os.path.join(cap, "thumb"), exist_ok=True)
        store = Store(os.path.join(tmp, "events.db"))

        # a) synced row whose image is still on disk -> re-queued
        make_img(cap, "a.jpg")
        store.add({"id": "a", "crossed_at": iso(1), "image_path": "a.jpg",
                   "thumb_path": "thumb/a.jpg", "direction": "in"})
        store.mark_synced(["a"])
        # b) synced row with no image -> left alone
        store.add({"id": "b", "crossed_at": iso(1), "image_path": "gone.jpg",
                   "direction": "in"})
        store.mark_synced(["b"])
        n = store.requeue_present_images(cap)
        assert n == 1, f"requeue expected 1 got {n}"
        assert any(e["id"] == "a" for e in store.pending()), "a should be re-queued"
        assert all(e["id"] != "b" for e in store.pending()), "b should stay synced"
        print("ok: requeue heals only rows whose image is still on disk")

        # c) normal pending + image -> uploads (stubbed), deletes, marks synced
        make_img(cap, "c.jpg")
        store.add({"id": "c", "crossed_at": iso(0), "image_path": "c.jpg",
                   "direction": "in"})
        orig = sync.post_event
        sync.post_event = lambda u, t, d, e: (True, "ok", True, False)
        assert sync.pass_once("http://x", "t", cap, store) >= 1
        assert not os.path.exists(os.path.join(cap, "c.jpg")), "c image not deleted"
        print("ok: successful upload deletes the image and marks synced")

        # d) pending, fresh, image missing -> kept (retried), never dropped
        store.add({"id": "d", "crossed_at": iso(0), "image_path": "missing.jpg",
                   "direction": "in"})
        sync.post_event = orig  # real missing-image path
        sync.pass_once("http://x", "t", cap, store, giveup_hours=24)
        row = next(e for e in store.pending() if e["id"] == "d")
        assert row["sync_attempts"] >= 1 and row["last_error"] == "image missing"
        print("ok: fresh missing-image row is retried, not dropped")

        # e) pending, old, image missing -> given up after the grace window
        store.add({"id": "e", "crossed_at": iso(48), "image_path": "missing.jpg",
                   "direction": "in"})
        sync.pass_once("http://x", "t", cap, store, giveup_hours=24)
        assert all(e["id"] != "e" for e in store.pending()), "e should be given up"
        print("ok: orphan older than the grace window is marked done")

        # f) network failure keeps the row and the image
        make_img(cap, "f.jpg")
        store.add({"id": "f", "crossed_at": iso(0), "image_path": "f.jpg",
                   "direction": "in"})
        sync.post_event = lambda u, t, d, e: (False, "connection refused", True, False)
        sync.pass_once("http://x", "t", cap, store, giveup_hours=24)
        assert any(e["id"] == "f" for e in store.pending()), "f must stay pending"
        assert os.path.exists(os.path.join(cap, "f.jpg")), "f image must stay"
        print("ok: network failure keeps the row and the image")

        print("\nALL SYNC TESTS PASSED")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
