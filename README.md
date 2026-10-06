# Vehicle Line Counter

RTSP vehicle counting with line-crossing detection. Inference is **YOLO26n INT8** via
**ncnn** (ARM-optimised), tracking is a **low-fps centroid tracker**, the edge store is
**offline-first SQLite**, and the dashboard is **Bun + Drizzle + SQLite** on a VPS.

```
┌──────────────── Pi 5 (DietPi, behind NAT) ─────────────────────────┐
│ NVR substream 640x360 H.264                                        │
│   reader thread ─▶ newest frame                                    │
│   detect thread ─▶ motion latch + YOLO26n INT8 (ncnn)             │
│   tracking loop ─▶ centroid tracker @ camera rate + crossing gate │
│   captures → JPEG + SQLite (offline) ; sync worker → HTTPS POST   │
│   terminal agent ─▶ outbound WSS → dashboard /terminal (PTY shell)│
└────────────────────────────────────────────────────────────────────┘
                                   │ https://tracker.drnanoinc.com
                                   ▼
┌──────────────── VPS (Bun + Drizzle + SQLite) ─────────────────────┐
│ POST /api/ingest (bearer)  → images + events                      │
│ POST /api/live   (bearer)  → live frame relay                     │
│ WS   /api/term   (bearer)  → reverse terminal relay (browser↔Pi)  │
│ GET  /                     → live + calendar/history + setup + term│
└────────────────────────────────────────────────────────────────────┘
```

Data leaves the Pi over **HTTPS only** (no SSH). Captures are stored locally and deleted
only after the VPS acknowledges them, so outages lose nothing. Live frames are relayed the
same way (outbound POST), and the dashboard is behind HTTP Basic auth.

The terminal is the same idea in reverse: `term_agent.py` keeps an **outbound** WebSocket to
the VPS (`/api/term?role=agent`) and bridges it to a local PTY, so a browser can get a shell
through `https://tracker.drnanoinc.com/terminal` without the Pi ever being reachable inbound.

## Detection pipeline

- **Input is the NVR sub-stream** (`.../unicast/c11/s1/live`, 640x360 H.264 @10 fps).
  It is cheaper to decode than the 1080p main stream and — because it is already 640 wide —
  gives better small-object recall than downscaling 1080p.
- **Motion latch.** A cheap MOG2 + frame-difference gate *arms* detection: the moment
  motion is seen, YOLO runs on every frame for `TRACKER_MOTION_HOLD` seconds (default 60).
  A **heartbeat** (`TRACKER_MOTION_HEARTBEAT`, default 3 s) still runs YOLO periodically
  when idle, so a crossing can never be missed entirely.
- **Decoupled tracking.** A dedicated thread runs detection as fast as the CPU allows while
  the main loop tracks at the camera rate (10 fps). Between fresh detections the tracker is
  fed an empty set and **coasts** the tracks forward (constant velocity), so the crossing
  test samples at 10 fps even when YOLO runs at only 3-6 fps.
- **Centroid tracking, not IoU.** At 10 fps a fast vehicle can move farther than its own box
  between samples, so IoU association (ByteTrack) loses it — often returning nothing after
  the first frame, which made fast bikes uncountable. Association is instead by *predicted
  centroid distance* (`TRACKER_ASSOC_FRAC`), so the prev→cur centroid segment can still be
  tested against the line. This suits sparse traffic; it is not built for dense crowds.
- **Crossing test (hysteresis).** Each track commits to a side of the line; a crossing
  counts only when the centroid emerges ≥ `TRACKER_CROSS_MARGIN_FRAC` of the frame height
  past the line on the other side, with its projection inside the drawn segment. This
  rejects low-confidence boxes that jitter across the line while still catching fast
  movers. A track must also be ≥ `TRACKER_MIN_TRACK_AGE` frames old, have travelled
  ≥ `TRACKER_MIN_TRAVEL_FRAC` of the frame height, and respect a per-track cooldown
  (`CrossingGate`).

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | Headless edge loop: ncnn detector + centroid tracker + crossing + capture + SQLite |
| `store.py` | Offline-first SQLite event store (+ retention prune) |
| `sync.py` | HTTP upload worker (retries; deletes after ACK) |
| `term_agent.py` | Reverse terminal agent: outbound WSS → local PTY shell (for `/terminal`) |
| `tools/export_model.py` | Export a YOLO `.pt` to ncnn at any input size |
| `tools/make_calib.py` | Build an INT8 calibration set from ROI crops |
| `tools/quantize_int8.py` | INT8-quantize an ncnn model (`ncnn2table`/`ncnn2int8`) |
| `tools/live_push.py` | Dev harness that mimics `run.py` from a laptop (can push to the VPS) |
| `tools/test_clip.py` | Run the production tracker/crossing over a video clip and report IN/OUT |
| `web/` | Bun dashboard + ingest API (Drizzle + SQLite) |
| `deploy/` | systemd units + env examples for the Pi |
| `models/yolo26n_ncnn_320x320/` | Production model (fp32 + INT8), ROI-matched input |

## Pi quick start (DietPi 64-bit)

A pre-imaged card is prepared via DietPi first-boot automation:
- installs `python3-venv`, `python3-pip`, creates a `tracker` user
- creates the venv and installs dependencies
- enables `tracker.service`, `sync.service` and `term-agent.service`

On first boot DietPi runs `/boot/Automation_Custom_Script.sh`; logs land in
`/var/log/tracker-install.log`.

### Manual install

```bash
sudo apt-get install -y python3-venv python3-pip
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
# ncnn pulls the GUI OpenCV; replace it with the headless build on a server:
./venv/bin/pip uninstall -y opencv-python opencv-python-headless
./venv/bin/pip install --no-cache-dir opencv-python-headless
```

Copy `config.example.json` to `config.json` and set the RTSP URL, then run:

```bash
./venv/bin/python run.py          # detector + tracker
./venv/bin/python sync.py         # uploader (loop), or --once
./venv/bin/python term_agent.py   # reverse terminal agent (optional)
```

The line and scan-area (ROI) are drawn in the dashboard and pulled by the Pi over HTTPS;
they are stored normalized, so they are resolution-independent.

### Environment (`/etc/tracker/tracker.env`)

See `deploy/tracker.env.example`. Key knobs:

| Var | Default | Meaning |
|-----|---------|---------|
| `TRACKER_MODEL` | `yolo26n` | Model family (`models/<name>_ncnn_<imgsz>/`) |
| `TRACKER_MODEL_IMGSZ` | `320x320` | Model input `HxW`, matched to the ROI aspect |
| `TRACKER_INT8` | `1` | Use `model_int8.ncnn.param` |
| `TRACKER_THREADS` | `4` | ncnn inference threads |
| `TRACKER_MOTION_HOLD` | `60` | Seconds motion keeps detection armed |
| `TRACKER_MOTION_HEARTBEAT` | `3` | Idle YOLO cadence (s) |
| `TRACKER_MOTION_SCALE` / `_MIN_AREA` | `0.5` / `12` | Motion-gate sensitivity |
| `TRACKER_MIN_TRACK_AGE` / `_MIN_TRAVEL_FRAC` | `1` / `0.02` | Crossing gate |
| `TRACKER_CROSS_MARGIN_FRAC` | `0.015` | Hysteresis past the line (fraction of frame height) |
| `TRACKER_ASSOC_FRAC` | `0.2` | Max centroid-association distance (fraction of frame width) |
| `TRACKER_TRACK_MAX_AGE` | `25` | Frames a track coasts with no detection before dropping |
| `TRACKER_RETENTION_DAYS` | `30` | Local capture/event retention |
| `TRACKER_VULKAN` | `0` | Optional GPU compute (use fp32 model) |
| `TRACKER_CAPTURE_PIPELINE` | – | GStreamer pipeline, e.g. Pi hardware `v4l2h264dec` |

The reverse terminal agent reads its own file, `/etc/tracker/term.env` (plus
`TRACKER_TOKEN` from `sync.env`):

| Var | Default | Meaning |
|-----|---------|---------|
| `TRACKER_TERM_URL` | `wss://tracker.drnanoinc.com/api/term` | Broker endpoint the agent dials |
| `TRACKER_TERM_SHELL` | `/bin/bash` | Shell exposed at `/terminal` |
| `TRACKER_TERM_COLS` / `_ROWS` | `80` / `24` | Initial PTY size |

It runs as the unprivileged `tracker` user; change `User=` in `term-agent.service`
to grant more privilege (not recommended).

## VPS dashboard

```bash
cd web
bun install
# env: DB_PATH, CAPTURES_DIR, TRACKER_TOKEN, DASH_USER, DASH_PASS, PORT=3010,
#      HOST=127.0.0.1, ENABLE_TERMINAL=1 (opt-in reverse shell at /terminal)
bun run server.ts
```

Exposed through the Cloudflare Tunnel as `tracker.drnanoinc.com` → `http://localhost:3010`.
Install with `web/deploy/tracker-web.service`.

Pages (each its own URL; `/` redirects to `/live`):

| Route | Contents |
|-------|----------|
| `/live` | MJPEG/snapshot live view + setup editor |
| `/calendar` | Month grid of per-day IN/OUT; tap a day to reveal its captures inline |
| `/history` | Capture list, optionally filtered by `?day=YYYY-MM-DD` |
| `/terminal` | Reverse web shell into the Pi (xterm.js); opt-in via `ENABLE_TERMINAL=1` |

- **Timezone:** timestamps are stored in UTC but displayed and day-bucketed in **IST**
  (UTC+5:30), so calendar days and "today" totals line up with local midnight.
- **Browser auth:** dashboard routes (`/live`, `/calendar`, `/history`, `/img/*`,
  `/thumb/*`, `/live.jpg`) use HTTP Basic.
- **Machine auth:** `/api/ingest`, `/api/live`, and `GET /api/config` accept the `TRACKER_TOKEN` bearer.
- **Live feed:** the Pi POSTs annotated frames to `/api/live` only while a viewer is present
  (`live_wanted`); the page polls `/live.jpg` snapshots (self-recovering after restarts).
- **Auto-refresh:** the calendar and history views poll `GET /api/events` every 8 s and
  re-render only when the event count changes; the header counts refresh from `GET /api/config`.

### Web terminal (`/terminal`)

A browser shell into the Pi, carried entirely over the existing outbound channel:

- Off unless `ENABLE_TERMINAL=1` in `/etc/tracker/web.env` — it grants an interactive
  shell, so it is opt-in. The page and its self-hosted `@xterm` assets sit behind the
  same HTTP Basic auth as the dashboard.
- The Pi's `term-agent.service` dials `wss://…/api/term?role=agent` with the
  `TRACKER_TOKEN` bearer and bridges the socket to a local PTY (`/bin/bash` as the
  `tracker` user). It retries every 3 s and respawns the shell if it exits.
- A browser fetches a short-lived one-time token from `GET /api/term-token`, then opens
  `wss://…/api/term?role=browser&t=…`; the Bun broker relays bytes between the single
  agent and all browser tabs. Keystrokes are text frames, PTY output is binary, and a
  leading `\x01` marks a JSON resize control frame.
- Needs no inbound port and no SSH: the Pi only needs outbound internet, so it works
  behind NAT on any network.

### Ingest contract

`POST /api/ingest` (`Authorization: Bearer <token>`, `multipart/form-data`):

| Field | Notes |
|-------|-------|
| `id` | unique, idempotency key |
| `track_id`, `class_id`, `label`, `confidence` | detection metadata |
| `direction` | `in` / `out` |
| `crossed_at` | ISO-8601 UTC |
| `bbox`, `line` | JSON arrays |
| `image`, `thumb` | JPEG files (raw frame, no overlay) |

## Models

Build the production model (and optional faster/other sizes):

```bash
# 1. export to ncnn at a %32 input size (pnnx; auto-falls back to Ultralytics for NMS-free models)
python tools/export_model.py --weights yolo26n.pt --imgsz 320x320

# 2. INT8-quantize, calibrated on the actual ROI crops (whole frames give poor accuracy)
python tools/make_calib.py --frames /path/to/frames --out calib --shape 320 320 \
    --roi x0,y0 x1,y1 x2,y2 x3,y3
python tools/quantize_int8.py --model models/yolo26n_ncnn_320x320 \
    --calib calib/list.txt --shape 320 320   # needs ncnn2table/ncnn2int8 on PATH

# 3. run
TRACKER_MODEL=yolo26n TRACKER_MODEL_IMGSZ=320x320 TRACKER_INT8=1 ./venv/bin/python run.py
```

Measured on x86 (ROI inference): YOLO26n `320x320` INT8 ≈ 30 fps; `288x288` ≈ 36;
`256x256` ≈ 46 (smaller = faster, lower confidence). `run.py` logs
`[stat] track=<fps> detect=<fps> tracks=<n>` every 10 s so the split is visible on-device.

## Gotchas

- **ncnn use-after-free:** keep the input `Mat` alive until after `extractor.extract()`.
  An inline temporary is GC'd while the extractor still references its memory — one bad
  frame then poisons the net into returning ~100 garbage boxes.
- Copy frames handed between threads; OpenCV may reuse the decode buffer.
- Ultralytics' ncnn export mis-compiles non-square graphs; use `pnnx` directly (handled by
  `tools/export_model.py`).
- The NVR has no MJPEG/JPEG endpoint, and browsers cannot play raw RTSP.

## License

MIT
