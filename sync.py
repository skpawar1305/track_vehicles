#!/usr/bin/env python3
"""Upload pending captures from the Pi to the VPS over HTTP.

Offline-first: events live in the local SQLite store until the server
acknowledges them. Images are deleted only after a 2xx response.

Env:
  TRACKER_API_URL        e.g. https://tracker.drnanoinc.com/api/ingest
  TRACKER_TOKEN          bearer token shared with the server
  TRACKER_DB_PATH        default events.db
  TRACKER_CAPTURE_DIR    default captures
  TRACKER_SYNC_INTERVAL  seconds between passes (default 20)
  TRACKER_SYNC_RESYNC_INTERVAL  seconds between self-heal sweeps (default 300)
  TRACKER_SYNC_GIVEUP_HOURS     drop a pending row whose image never appears
                                (default 24)
"""
import os
import sys
import time
from datetime import datetime, timezone

import requests

from store import Store

EVENT_FIELDS = ("id", "track_id", "class_id", "label", "confidence",
                "direction", "crossed_at", "bbox", "line")


def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        print(f"[sync] FATAL: {name} is not set", file=sys.stderr)
        sys.exit(2)
    return val


def post_event(api_url, token, capture_dir, ev):
    img = os.path.join(capture_dir, ev["image_path"])
    if not os.path.isfile(img):
        return False, "image missing", False, False
    data = {k: ("" if ev[k] is None else str(ev[k])) for k in EVENT_FIELDS}
    thumb = None
    if ev.get("thumb_path"):
        tp = os.path.join(capture_dir, ev["thumb_path"])
        if os.path.isfile(tp):
            thumb = tp
    sub = None
    if ev.get("sub_path"):
        sp = os.path.join(capture_dir, ev["sub_path"])
        if os.path.isfile(sp):
            sub = sp
    try:
        with open(img, "rb") as fh:
            files = {"image": (ev["image_path"], fh, "image/jpeg")}
            if thumb:
                th = open(thumb, "rb")
                files["thumb"] = (ev["thumb_path"], th, "image/jpeg")
            if sub:
                sf = open(sub, "rb")
                files["sub"] = (ev["sub_path"], sf, "image/jpeg")
            try:
                r = requests.post(api_url, data=data, files=files,
                                  headers={"Authorization": f"Bearer {token}"},
                                  timeout=(5, 60))
            finally:
                if thumb:
                    th.close()
                if sub:
                    sf.close()
        if r.ok:
            return True, r.text[:200], True, False
        # 4xx is permanent until an operator fixes it (bad token, malformed
        # event); 5xx/network are transient. Neither should drop the capture.
        permanent = 400 <= r.status_code < 500
        return False, f"HTTP {r.status_code}: {r.text[:200]}", False, permanent
    except requests.RequestException as e:
        return False, str(e), False, False


def delete_local(capture_dir, ev):
    for rel in (ev.get("image_path"), ev.get("thumb_path"), ev.get("sub_path")):
        if not rel:
            continue
        p = os.path.join(capture_dir, rel)
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def _age_hours(ev, now=None):
    """Age of an event in hours (0 if the timestamp is missing/unparseable)."""
    ts = ev.get("crossed_at") or ev.get("synced_at")
    if not ts:
        return 0.0
    try:
        t = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return max(0.0, ((now or datetime.now(timezone.utc)) - t).total_seconds() / 3600.0)


def pass_once(api_url, token, capture_dir, store, giveup_hours=24.0):
    sent = 0
    for ev in store.pending(limit=100):
        ok, msg, has_image, permanent = post_event(api_url, token, capture_dir, ev)
        if ok:
            store.mark_synced([ev["id"]])
            delete_local(capture_dir, ev)
            sent += 1
        elif not has_image:
            # The local image is not on disk (yet): do NOT silently drop the
            # capture. The row may have been queued a moment before its JPEG
            # landed, or a bogus "synced" mark from an outage is being healed.
            # Keep retrying; only give up once the event is old enough that the
            # pixels can no longer appear.
            if _age_hours(ev) >= giveup_hours:
                store.mark_synced([ev["id"]])
                print(f"[sync] {ev['id']}: {msg}; older than {giveup_hours:.0f}h, marked done")
            else:
                store.record_failure(ev["id"], msg)
                print(f"[sync] {ev['id']}: {msg}; retrying")
            continue  # local issue: try the rest of the batch, don't hammer
        else:
            # Keep the row: offline is expected, and a permanent 4xx must be
            # fixed by an operator (we never silently discard a capture).
            store.record_failure(ev["id"], msg)
            tag = "PERMANENT" if permanent else "offline"
            print(f"[sync] {ev['id']}: {msg} [{tag}]; keeping row")
            break  # don't hammer the server; retry next pass
    return sent


def main():
    once = "--once" in sys.argv
    api_url = env("TRACKER_API_URL", required=True)
    token = env("TRACKER_TOKEN", required=True)
    capture_dir = env("TRACKER_CAPTURE_DIR", "captures")
    db_path = env("TRACKER_DB_PATH", "events.db")
    interval = float(env("TRACKER_SYNC_INTERVAL", "20"))
    resync_interval = float(env("TRACKER_SYNC_RESYNC_INTERVAL", "300"))
    giveup_hours = float(env("TRACKER_SYNC_GIVEUP_HOURS", "24"))

    store = Store(db_path)
    print(f"[sync] API {api_url} -> {capture_dir} (db {db_path})")
    last_resync = 0.0
    while True:
        try:
            # Self-heal rows falsely marked synced: a successful upload deletes
            # the local image, so an image still on disk means the mark never
            # reached the server (e.g. it was set while the VPS was unreachable).
            # Re-queueing them lets the pass below re-upload; idempotent because
            # the server de-dupes by id, and cheap because it is throttled.
            if time.time() - last_resync >= resync_interval:
                last_resync = time.time()
                n = store.requeue_present_images(capture_dir)
                if n:
                    print(f"[sync] re-queued {n} event(s) marked synced but still on disk")
            sent = pass_once(api_url, token, capture_dir, store, giveup_hours)
        except Exception as e:
            print(f"[sync] pass error: {e}", file=sys.stderr)
            sent = 0
        if once:
            print(f"[sync] done, {sent} sent")
            return
        if sent == 0:
            time.sleep(interval)
        # if we sent a full batch, loop immediately to drain the backlog


if __name__ == "__main__":
    main()
