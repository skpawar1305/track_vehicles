# Vehicle Line Counter

RTSP vehicle counting with line-crossing detection. Inference is **YOLO26n INT8** via
**ncnn** (ARM-optimised), tracking is a **two-tier velocity tracker** (IoU primary,
centroid-distance fallback), the edge store is **offline-first SQLite**, and the dashboard
is **Bun + Drizzle + SQLite** on a VPS.

```
┌──────────────── Pi 5 (DietPi, behind NAT) ─────────────────────────┐
│ CCTV substream 640x360 H.264 (RtspRelay: SHA-256 Digest)           │
│   reader thread ─▶ newest frame                                    │
│   detect thread ─▶ motion latch + YOLO26n INT8 (ncnn)             │
│   tracking loop ─▶ IoU tracker @ camera rate + crossing gate       │
│   captures → JPEG + SQLite (offline) ; sync worker → HTTPS POST   │
│   terminal agent ─▶ outbound WSS → dashboard /terminal (PTY shell)│
└────────────────────────────────────────────────────────────────────┘
                                   │ https://tracker.drnanoinc.com
                                   ▼
┌──────────────── VPS (Bun + Drizzle + SQLite) ─────────────────────┐
│ POST /api/ingest (bearer)  → images + events                      │
│ POST /api/live   (bearer)  → live frame relay                     │
│ WS   /api/term   (bearer)  → reverse terminal relay (browser↔Pi)  │
│ GET  /                     → live + timeline + setup + term       │
└────────────────────────────────────────────────────────────────────┘
```

Data leaves the Pi over **HTTPS only** (no SSH). Captures are stored locally and deleted
only after the VPS acknowledges them, so outages lose nothing. Live frames are relayed the
same way (outbound POST), and the dashboard is behind HTTP Basic auth.

## Android/phone replacement (evaluated, not pursued)

A phone edge was prototyped to use the camera's 1080p main stream, which the Pi cannot afford
to decode. It was **rejected**: on the S21 FE the software YUV→RGB path capped 1080p decode at
~12 fps and the SoC throttled (inference ~45–51 ms), i.e. no faster than the Pi's inference, so
the phone offered no counting advantage while being a far weaker always-on host (OS service
kills, battery and thermal limits). The edge stays on the Pi with the substream. The camera and
stream constraints behind this decision are recorded in `AGENTS.md`.

The terminal is the same idea in reverse: `term_agent.py` keeps an **outbound** WebSocket to
the VPS (`/api/term?role=agent`) and bridges it to a local PTY, so a browser can get a shell
through `https://tracker.drnanoinc.com/terminal` without the Pi ever being reachable inbound.

## Detection pipeline

- **Input is a camera sub-stream** (ONVIF path `.../Streaming/Channels/102`, 640x360
  H.264). It is cheaper to decode than the 1080p main stream and — because it is already
  640 wide — gives better small-object recall than downscaling 1080p. Cameras that demand
  SHA-256 RTSP Digest are reached through `rtsp_relay.py` (see Gotchas).
- **Motion latch.** A cheap MOG2 + frame-difference gate *arms* detection: the moment
  motion is seen, YOLO runs on every frame for `TRACKER_MOTION_HOLD` seconds (default 60).
  A **heartbeat** (`TRACKER_MOTION_HEARTBEAT`, default 3 s) still runs YOLO periodically
  when idle, so a crossing can never be missed entirely.
- **Decoupled tracking.** A dedicated thread runs detection as fast as the CPU allows while
  the main loop tracks at the camera rate (a genuine ~25 fps on the current camera). Between
  fresh detections the tracker is fed an empty set and **coasts** the tracks forward
  (constant velocity), so the crossing test samples at frame rate (YOLO is ~27 ms/frame
  ≈ 37 fps on the Pi 5, so detection keeps up).
- **Two-tier association.** Detections are matched to tracks in two tiers: (1) **IoU**
  (`TRACKER_IOU_GATE`) against each track's *velocity-predicted* box — the last box
  translated by its px/second velocity over the elapsed time, so frame-to-frame at ~25 fps
  this is a standard IoU tracker; (2) a **centroid-distance fallback** whose gate widens
  with the gap (`TRACKER_ASSOC_FRAC`, `TRACKER_ASSOC_GAP_FRAC`, capped by
  `TRACKER_ASSOC_MAX_FRAC`), so a track re-acquires a vehicle that moved far during a
  multi-second stall even if the predicted box missed. IoU always wins over the fallback.
  No appearance model; suits sparse traffic, not dense crowds.
- **Crossing test (hysteresis + confirmed transit + re-arm).** Three rules, in `CrossingGate`:
  * *Hysteresis.* Each track commits to a side of the line; a flip is only a *candidate*
    when the centroid emerges ≥ `TRACKER_CROSS_MARGIN_FRAC` of the frame height past the
    line, with its projection inside the drawn segment. A parked/working vehicle on the
    line can't alternate IN/OUT.
  * *Confirmed transit.* The candidate only *counts* once the centroid reaches
    `TRACKER_CONFIRM_FRAC` (0.2) of the frame height past the line — *deeper* than the
    hysteresis margin, on the opposite side. Short of that it is only *pending*; a return
    toward the original side (or into the band) cancels it. This is what stops a machine
    oscillating on the line — or a bbox refit flicking the centroid across — from
    committing a spurious OUT/IN pair a second apart. Set it to `0` for the legacy
    "count as soon as the margin is cleared".
  * *Re-arm.* After a crossing commits, the track must move `TRACKER_CROSS_REARM_FRAC` of
    the frame height **away from the crossing point** before another crossing may commit —
    so a vehicle that only drifts back and forth at the line is counted once. Set it to `0`
    to disable.
  `TRACKER_MIN_TRACK_AGE` is the only other gate (the run.py pre-filter and `update()`). A
  **keep-out polygon** — drawn with the **Keep-out** tool in Camera setup, stored as
  `keepout` in the camera config (`norm_keepout` in `config.json`) — additionally *freezes*
  the gate while the centroid is inside it; place it over a working excavator/loader that
  swings across the line. `tools/test_crossing.py` exercises all of the above.
- **Zone-transition counting (instead of the line).** Draw an **Entry zone** and an **Exit
  zone** in Camera setup and counting switches to `ZoneGate`: a track is committed to a zone
  only while its centroid is `TRACKER_CROSS_MARGIN_FRAC` of the frame height *inside* it (the
  gap between the zones is the hysteresis a bare point-in-polygon test lacks), and a
  transition between the two committed zones counts once the centroid is `TRACKER_CONFIRM_FRAC`
  inside the destination zone. Entry = IN, Exit = OUT (`IN side`/`flip_sides` swaps). A vehicle
  must be seen in the other zone first, so a machine working inside one zone never counts and
  the "appears already past the line" undercount disappears. The keep-out still applies.
  `tools/test_zones.py` covers it; the line is the fallback when the zones aren't drawn.

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | Headless edge loop: ncnn detector + IoU tracker + crossing + capture + SQLite |
| `rtsp_relay.py` | SHA-256 RTSP Digest → local RTP/SDP shim for cameras FFmpeg can't authenticate |
| `store.py` | Offline-first SQLite event store (+ retention prune) |
| `sync.py` | HTTP upload worker (retries; deletes after ACK) |
| `term_agent.py` | Reverse terminal agent: outbound WSS → local PTY shell (for `/terminal`) |
| `tools/export_model.py` | Export a YOLO `.pt` to ncnn at any input size |
| `tools/make_calib.py` | Build an INT8 calibration set from ROI crops |
| `tools/quantize_int8.py` | INT8-quantize an ncnn model (`ncnn2table`/`ncnn2int8`) |
| `tools/live_push.py` | Dev harness that mimics `run.py` from a laptop (can push to the VPS) |
| `tools/test_clip.py` | Run the production tracker/crossing over a video clip and report IN/OUT |
| `tools/test_crossing.py` | Synthetic crossing-gate regression + annotated re-arm demo video |
| `tools/set_camera_time.py` | Read/set a camera clock over ONVIF (Manual or NTP) |
| `tools/onvif_media.py` | Inspect a camera's ONVIF encoder configs / profiles / RTSP URIs (diagnose a blurry substream) |
| `tools/set_encoder.py` | Read/set a camera's ONVIF encoder (resolution/bitrate/GOP/fps/quality); dry-run by default |
| `web/` | SvelteKit (static SPA) dashboard + Bun ingest/reverse-terminal API (Drizzle + SQLite) |
| `deploy/` | systemd units + env examples for the Pi |
| `models/yolo26n_ncnn_320x320/` | Production model (fp32 + INT8), ROI-matched input; INT8 calibrated on dumpers |
| `calib/dumper_trucks/` | INT8 calibration images (dump trucks / tippers / haulers) for the production model |

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
| `TRACKER_MODEL` | `yolo26n` | Model family (`models/<name>_ncnn_<imgsz>/`) — also `"model"` in `config.json` |
| `TRACKER_MODEL_IMGSZ` | `320x320` | Model input `HxW`, matched to the ROI aspect — also `"model_imgsz"` in `config.json` |
| `TRACKER_INT8` | `1` | Use `model_int8.ncnn.param` — also `"int8"` in `config.json` |
| `TRACKER_DETECTOR` | `hybrid` | `hybrid` (motion-gated YOLO) · `motion` · `yolo` · `nanodet` (ncnn) · `opencv` (`cv2.dnn` ONNX) — also `"detector"` in `config.json` |
| `TRACKER_DNN_MODEL` | – | Model path for the `nanodet`/`opencv` detectors — also `"dnn_model"` in `config.json` |
| `TRACKER_THREADS` | `4` | ncnn inference threads (also settable as `"threads"` in `config.json`) |
| `TRACKER_MOTION_HOLD` | `60` | Seconds motion keeps detection armed |
| `TRACKER_MOTION_HEARTBEAT` | `3` | Idle YOLO cadence (s) |
| `TRACKER_MOTION_SCALE` / `_MIN_AREA` | `0.5` / `12` | Motion-gate sensitivity |
| `TRACKER_MIN_TRACK_AGE` | `1` | Frames a track must live before the crossing gate will count it |
| `TRACKER_CROSS_MARGIN_FRAC` | `0.06` | Hysteresis past the line (fraction of frame height) |
| `TRACKER_CROSS_REARM_FRAC` | `0.2` | Distance a track must move from its last crossing before it can count again (`0` disables) |
| `TRACKER_CONFIRM_FRAC` | `0.2` | Extra depth past the line a crossing must reach before it counts (`0` = legacy, count at the margin) |
| `TRACKER_IOU_GATE` | `0.2` | Min IoU with the velocity-predicted box to match a detection |
| `TRACKER_ASSOC_FRAC` | `0.2` | Fallback centroid gate = fraction of frame width |
| `TRACKER_ASSOC_GAP_FRAC` | `0.5` | Extra fallback gate per second of stream gap |
| `TRACKER_ASSOC_MAX_FRAC` | `1.0` | Cap on the fallback gate (fraction of frame diagonal) |
| `TRACKER_TRACK_MAX_AGE` | `40` | Frames a track coasts with no detection before dropping |
| `TRACKER_ASSOC_PRED_CAP` | `0.5` | Cap (s) on the coasted-centroid extrapolation used for display |
| `TRACKER_RETENTION_DAYS` | `30` | Local capture/event retention |
| `TRACKER_VULKAN` | `0` | Optional GPU compute (use fp32 model) |
| `TRACKER_CAPTURE_PIPELINE` | – | GStreamer pipeline, e.g. Pi hardware `v4l2h264dec` |
| `TRACKER_RTSP_RELAY` | `0` | Route capture through `RtspRelay` for SHA-256 Digest cameras |
| `TRACKER_CAM2_URL` | – | Optional second camera substream shown raw on `/live` (no detection) |
| `TRACKER_CAPTURE_MAIN_STREAM` | `1` | Saved image = main-profile ONVIF snapshot; set `0` to save the substream frame instead |
| `TRACKER_CAPTURE_URL` | derived | Full-res crossing still URL; used when `TRACKER_CAPTURE_MAIN_STREAM=1` |
| `TRACKER_CAPTURE_MAX_AGE` | `1.5` | Queue age (s) after which the main-stream fetch is skipped |
| `TRACKER_DUP_IOU` | `0.6` | Same-vehicle IoU: detector cross-class NMS + crossing de-dupe |
| `TRACKER_DEDUP_CENTER_FRAC` | `0.03` | Same-vehicle centre tolerance (frame width) for the crossing de-dupe, so a queued follower isn't collapsed |
| `TRACKER_CROSS_DEDUP_S` | `1` | Window (s) for collapsing a duplicate crossing from a second track |

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

The UI is a **static SvelteKit SPA** (`web/`, `@sveltejs/adapter-static`); the Bun API server
(`web/server.ts`) owns `/api/*`, image serving, and the reverse-terminal WS, and serves the
SPA's assets (client routes fall back to `index.html`).

```bash
cd web
bun install
bun run build                 # static SPA -> web/build/
# deploy: copy web/build/ to WEB_ROOT (default <server.ts dir>/spa) and server.ts alongside
bun run server.ts
# env: DB_PATH, CAPTURES_DIR, TRACKER_TOKEN, DASH_USER, DASH_PASS, PORT=3010,
#      HOST=127.0.0.1, ENABLE_TERMINAL=1 (opt-in reverse shell at /terminal),
#      CAM2=1 (second raw camera on /live; Pi sets TRACKER_CAM2_URL),
#      WEB_ROOT (where the built SPA lives; default: <server.ts dir>/spa)
```

Exposed through the Cloudflare Tunnel as `tracker.drnanoinc.com` → `http://localhost:3010`.
Install with `web/deploy/tracker-web.service`.

Pages (each its own URL; `/` redirects to `/live`):

| Route | Contents |
|-------|----------|
| `/live` | MJPEG/snapshot live view + setup editor |
| `/timeline` | Capture timeline: a horizontally scrollable strip of day squares (date + IN/OUT counts), oldest→newest with today auto-scrolled into view; `←`/`→` step a day, never into the future; days with no captures are still shown; the selected day's captures load on demand (deep-link `?day=YYYY-MM-DD` or `?d=`; old `/calendar` links redirect here) |
| `/terminal` | Reverse web shell into the Pi (xterm.js); opt-in via `ENABLE_TERMINAL=1` |

- **Timezone:** timestamps are stored in UTC but displayed and day-bucketed in **IST**
  (UTC+5:30), so calendar days and "today" totals line up with local midnight.
- **Browser auth:** dashboard routes (`/live`, `/timeline`, `/img/*`,
  `/thumb/*`, `/live.jpg`) use HTTP Basic.
- **Machine auth:** `/api/ingest`, `/api/live`, and `GET /api/config` accept the `TRACKER_TOKEN` bearer.
- **Live feed:** the Pi POSTs annotated frames to `/api/live` only while a viewer is present
  (`live_wanted`); the page polls `/live.jpg` snapshots (self-recovering after restarts).
  A second camera can be shown raw alongside: with `CAM2=1` the Pi POSTs its substream to
  `/api/live2` (no detection) and the page polls `/live2.jpg` — again only while watched, so
  an idle edge streams nothing. Left/right arrows on `/live` switch between Main and Camera 2
  and **only the visible feed is polled** (the idle one's `live_wanted`/`live2_wanted` goes
  stale within ~10 s, so the Pi stops uploading it — half the live data); it always opens on
  Main. The first frame is instant while the pipeline is warm (a `/live.jpg` fetch in the last
  ~10 s) but ~3.5 s cold, while the Pi notices `live_wanted` on its 3 s config poll.
- **Timeline data:** the page loads only per-day IN/OUT counts (`GET /api/events`, polled
  every 8 s); a day's captures are fetched on demand from `GET /api/day?d=YYYY-MM-DD` and
  cached client-side (stale-while-revalidate), so switching days is instant and never flashes
  "Loading". The header counts refresh from `GET /api/config`. Each capture card shows the
  substream thumbnail at the line; tapping it opens the saved full image, and a small icon
  overlaid on the image opens the crossing thumbnail (`/thumb`) in a new tab. Each capture has a
  compact **Ignore** icon (`POST /api/ignore`): the card stays in the gallery (dimmed) but drops
  out of every IN/OUT total, and the toggle is optimistic (in place, no gallery rebuild, no
  scroll jump).

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
| `image`, `thumb`, `sub` | JPEG files (raw frame, no overlay; `sub` = detection substream) |

## Failure handling (best effort)

Nothing on the edge is allowed to fail silently: a lost frame is free, a lost
count or a dead worker is not. Invariants the code upholds:

- **Background workers never die quietly.** The reader and the capture worker
  each wrap their loop body in `try/except` and log + recover. A raise used to
  kill the thread (a daemon) while `run.py` kept "running" and counting.
- **A stalled stream reconnects itself.** If no genuinely-new frame arrives for
  `TRACKER_READ_TIMEOUT` (default 15 s) the reader forces a reopen; an advancing
  container timestamp counts as alive, so a legitimately static (noise-free)
  scene is not mistaken for a stall. `stimeout`/`rw_timeout` bound a blocked
  `cap.read()`.
- **In-memory counts and the DB never diverge.** If the capture queue is full,
  the crossing is persisted inline from the substream frame instead of dropped.
  A locked DB is retried (`busy_timeout` + a short retry) rather than fatal.
- **Offline is expected; unsynced captures are never pruned.** `store.prune`
  only deletes `synced=1` rows. The sync worker deletes local files only after a
  `2xx`.
- **A permanent upload error keeps the row.** `sync.py` classifies `4xx` as
  permanent (bad token/malformed) and logs `PERMANENT …; keeping row`, recording
  `sync_attempts`/`last_error` so the stuck event is visible. Only a genuine
  `2xx` marks it synced.
- **A "synced" mark is verified by deleting the image — and self-heals when it
  isn't.** A successful upload deletes the local JPEG, so a row marked
  `synced=1` whose image is *still on disk* means the mark never reached the
  server (e.g. it was set while the VPS was unreachable during an internet
  outage). `sync.py` sweeps those rows (`TRACKER_SYNC_RESYNC_INTERVAL`, default
  300 s) and re-uploads them; the VPS de-dupes by id, so it is idempotent. A
  missing image is *retried*, not silently dropped, until
  `TRACKER_SYNC_GIVEUP_HOURS` (default 24) before it is given up.
- **Ingest is atomic.** The VPS writes each uploaded image to a temp file and
  renames it into place, so a crash mid-write can never leave a truncated JPEG
  to be served.
- **No silent `except: pass`.** Recurring errors from the config poll, live
  relay, prune sweep and snapshot fetch are rate-limited (first hit, then once a
  minute) so a wrong URL or a bad token is diagnosable.

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

INT8 is **post-training only — no fine-tuning.** The calibration images never
train the model; they only pick the per-tensor activation ranges (`ncnn2table`)
that `ncnn2int8` bakes into `model_int8.*`. The production model is calibrated on
`calib/dumper_trucks/` (public dump-truck/tipper photos — the truck class the
counter cares about; see its `README.md` / `ATTRIBUTIONS.md`), which only rewrites
`model_int8.ncnn.bin` + `model_int8.table` while the fp32 graph stays identical:

```bash
python tools/quantize_int8.py --model models/yolo26n_ncnn_320x320 \
    --calib calib/dumper_trucks/list.txt --shape 320 320 \
    --tools <ncnn bin dir with ncnn2table/ncnn2int8>
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
- **Drop NanoDet "container" boxes by containment.** The `nanodet`/`opencv` detectors can
  return a loose box around a vehicle — or spanning two adjacent trucks — that IoU-NMS misses
  because its IoU with the tight box is low. Both paths drop any box mostly inside a
  higher-confidence one, so the tracker does not mint a phantom second track.
- **The NVR can be the RTSP source (`"relay": false`) once Stream Encryption is off.** The
  Prama NVR (`192.168.225.150`) uses **MD5** RTSP Digest, so FFmpeg/OpenCV authenticate
  directly and `rtsp_relay` is not needed. Channel map: ch1 = `192.168.225.70` (embankment),
  ch2 = `192.168.225.71` (dumper site); substreams are `640x360`
  (`/Streaming/Channels/202`). **`Platform Access ▸ Stream Encryption` must be OFF** — with it
  on the NVR advertises a valid H.264 SDP `sprop` but scrambles the RTP payload, so decoders
  emit pure green frames (in-band SPS `profile_idc=150`, slices referencing a nonexistent PPS).
  Its ONVIF is **HTTP-only**; evidence can come from
  `http://192.168.225.150/onvif/snapshot/201` (full-res).
- The NVR has no MJPEG/JPEG endpoint, and browsers cannot play raw RTSP.
- **SHA-256 RTSP Digest cameras:** FFmpeg (and therefore OpenCV's capture
  backend) negotiates RTSP Digest with MD5 only and returns `401` against a
  camera that challenges with `algorithm="SHA-256"` (e.g. the PT-NC120D3-WNM(D2)
  at `.../Streaming/Channels/102`). Set `"relay": true` in `config.json` (or
  `TRACKER_RTSP_RELAY=1`): `rtsp_relay.RtspRelay` does the SHA-256 handshake
  itself, asks the camera for RTP/UDP on a loopback port, writes an SDP, and
  OpenCV decodes that instead. It needs the camera and Pi on the same LAN
  (UDP return path). Use ONVIF `GetStreamUri` to discover the correct RTSP path
  when a new camera's URL is unknown.
- **Camera clock:** the camera ships with a *Manual* clock that is often wrong
  (the PT-NC120D3 arrived 5h30m behind — local time mistaken for UTC). Read or
  set it with `tools/set_camera_time.py`; prefer `--set-ntp pool.ntp.org` so it
  self-corrects. It uses ONVIF `Get/SetSystemDateAndTime` + `SetNTP` with
  WS-UsernameToken auth (pass the password via `TRACKER_CAM_PASSWORD`).
- **Sharpen the substream at the camera, not the Pi.** `tools/set_encoder.py`
  reads/sets the ONVIF encoder (resolution/bitrate/GOP/fps/quality) over
  WS-UsernameToken, dry-run by default. The PT-NC120D3 substream only offers
  `640x360` and `640x480` (there is no `640x320`); production runs
  **`640x360`, 2048 kbps, GOP 25** (was `640x480`/1024/50). The camera downscales
  the *same* field of view rather than cropping, so the normalized line/ROI stay
  valid across a resolution change — no re-draw needed. If blur persists after
  raising the bitrate and cutting the GOP, it is packet loss (UDP/Wi-Fi), not the
  encoder; a shorter GOP only clears artifacts faster, it does not reduce them.
- **Evidence saves the main-stream image, thumbnails come from the substream.**
  The saved `<id>.jpg` is the camera's main-profile ONVIF snapshot
  (`/onvif-http/snapshot?Profile_1`, camera-side JPEG, SHA-256 Digest), while the
  **thumbnail shown in the timeline is always the 640×360 substream frame from the
  crossing instant**, so it shows the vehicle at the line. Set
  `TRACKER_CAPTURE_MAIN_STREAM=0` to save the crossing substream frame as the full
  image too: requesting the main-profile snapshot
  can perturb the counting substream on the PT-NC120D3 (partial/blurry stills,
  occasional substream glitch), which is why that was once the default. The fetch
  runs on the capture worker
  (never the track loop), reuses a `requests.Session`, skips a stale still
  (`TRACKER_CAPTURE_MAX_AGE`, default 1.5 s), falls back to the substream frame,
  and stores **both** `<id>.jpg`
  (main stream) and `<id>_sub.jpg` (crossing frame, served at `/sub/<id>`) when it
  succeeds. URL/creds default to the stream URL's (`TRACKER_CAPTURE_URL`/`_USER`/`_PASSWORD`).
- **One vehicle can arrive as two class-boxes — de-duplicated.** The INT8 detector
  sometimes fires two near-identical boxes for one vehicle with different winning
  classes (car/truck), which would otherwise become two tracks and two counts. A
  class-agnostic NMS (`TRACKER_DUP_IOU`, default 0.6) merges them at the detector
  (the Ultralytics `agnostic_nms` equivalent), and a crossing-level de-dupe
  (`TRACKER_CROSS_DEDUP_S`, default 1 s) guarantees one event. Counting uses a
  two-tier tracker — IoU on the velocity-predicted box (`TRACKER_IOU_GATE`), with a
  gap-widened centroid-distance fallback — so the vehicle stays matchable across a
  multi-second stall.
- **A vehicle that backs up at the line is counted once.** The gate is three rules —
  hysteresis + confirmed transit + re-arm. Hysteresis alone isn't enough: a vehicle that
  crosses, then drifts back and forth over the line keeps re-committing a side flip, so a
  crossing must reach `TRACKER_CONFIRM_FRAC` (0.2) of the frame height past the line to
  count (a return before that cancels it), and `CrossingGate` re-anchors on each committed
  crossing and requires the track to move `TRACKER_CROSS_REARM_FRAC` (0.2) away from that
  point before another crossing can commit (0.15 was too small — a slow dumper that
  overshot ~55 px re-armed and re-counted). A genuine pass-through confirms and its later
  OUT still counts. The older `min_travel_frac`/cooldown/15-frame gates were removed: they
  were redundant with (or weaker than) these. `tools/test_crossing.py` asserts wobble=1,
  passthru=1, true-return=2, nudge=1, rapid-reversal=0 and a queued pair=2, and renders the
  band/anchor/state.
- **Thermals / CPU.** `detect_loop` runs ncnn unthrottled on `TRACKER_THREADS` cores
  (default 4) plus software H.264 decode, so the Pi 5 sits near the `temp_limit` in
  `/boot/firmware/config.txt`. Lower `threads` (via `config.json`) to cut heat — a
  320×320 model scales sublinearly, so 2 threads keeps plenty of inf/s. Read live
  rates/temp from `/tmp/tracker_stats.json` (`TRACKER_STATS_PATH`); `run.py` writes it
  every 10 s since the service journal isn't readable by the `tracker` user.
- **Offline-first, and unsynced rows are never pruned.** Events and images queue
  locally until `sync.py` gets a 2xx from the VPS, which is the only point a local
  file is deleted. `store.prune` only removes `synced=1` rows older than
  `TRACKER_RETENTION_DAYS`, so a VPS outage longer than the retention window
  cannot lose un-synced captures.

## License

MIT
