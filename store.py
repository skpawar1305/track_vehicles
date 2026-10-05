"""Offline-first event store for the Pi.

Crossings are written to a local SQLite DB plus a JPEG on disk. The sync
worker uploads pending rows to the VPS over HTTP and only deletes the image
after the server acknowledges it, so nothing is lost while offline.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone, timedelta


class Store:
    def __init__(self, db_path="events.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                track_id    INTEGER,
                class_id    INTEGER,
                label       TEXT,
                confidence  REAL,
                direction   TEXT,
                crossed_at  TEXT,
                bbox        TEXT,
                line        TEXT,
                image_path  TEXT,
                thumb_path  TEXT,
                synced      INTEGER DEFAULT 0,
                synced_at   TEXT
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def _json_safe(v):
        """Coerce numpy scalars (np.int32/float32) to plain Python types.

        A numpy value that reaches json.dumps() raises and — because the edge
        persists inline in its main loop — takes the whole process down, losing
        the crossing. Never let that happen.
        """
        def norm(o):
            if isinstance(o, (list, tuple)):
                return [norm(x) for x in o]
            return o.item() if hasattr(o, "item") else o
        return norm(v)

    def add(self, ev):
        bbox = ev.get("bbox")
        line = ev.get("line")
        with self.lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO events
                   (id, track_id, class_id, label, confidence, direction,
                    crossed_at, bbox, line, image_path, thumb_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ev["id"], ev.get("track_id"), ev.get("class_id"),
                    ev.get("label"), ev.get("confidence"), ev.get("direction"),
                    ev.get("crossed_at") or datetime.now(timezone.utc).isoformat(),
                    json.dumps(self._json_safe(bbox)) if bbox is not None else None,
                    json.dumps(self._json_safe(line)) if line is not None else None,
                    ev.get("image_path"), ev.get("thumb_path"),
                ),
            )
            self.conn.commit()

    def pending(self, limit=50):
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM events WHERE synced=0 ORDER BY crossed_at ASC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def mark_synced(self, ids):
        if not ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.conn.executemany(
                "UPDATE events SET synced=1, synced_at=? WHERE id=?",
                [(now, i) for i in ids],
            )
            self.conn.commit()

    def total_counts(self):
        with self.lock:
            cur = self.conn.execute(
                "SELECT direction, COUNT(*) FROM events GROUP BY direction"
            )
            return {d: n for d, n in cur.fetchall()}

    def prune(self, retention_days=30):
        """Delete events older than retention_days; return their file paths."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%d")
        with self.lock:
            cur = self.conn.execute(
                "SELECT id, image_path, thumb_path FROM events "
                "WHERE date(COALESCE(crossed_at, synced_at)) < ?",
                (cutoff,),
            )
            rows = cur.fetchall()
            paths = [p for _, ip, tp in rows for p in (ip, tp) if p]
            self.conn.executemany("DELETE FROM events WHERE id=?", [(r[0],) for r in rows])
            self.conn.commit()
            return paths
