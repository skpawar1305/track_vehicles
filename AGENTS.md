# AGENTS

## Working style
- Always use best effort: act autonomously, make reasonable decisions, and take the
  task as far as possible without asking for hand-holding. Prefer a working result
  over questions when the path is clear.
- Follow the pareto principle: do the highest-value, lowest-effort work first.
- Be concise; this project is run from a terminal.

## Best-effort invariants
Failures are expected; silent failures are bugs. Keep these true everywhere:
- **Workers never die silently.** Every long-lived loop (`reader_loop`,
  `detect_loop`, `capture_loop`, `sync.py main`, `config_loop`, `live_loop`,
  `cam2_loop`, `prune_loop`) wraps its *whole* body in `try/except` and logs +
  recovers — not just the risky call. A raise in a daemon thread must not stop
  counting while the process still looks alive.
- **A wedged worker is caught, not just logged.** Each loop stamps `_beat(name)`
  every iteration; `supervisor_loop` forces a draining restart if a
  counting-path worker (`reader`/`detect`/`capture`) stops heartbeating past its
  budget (60 s). `WORKER_TIMEOUTS` must stay above each loop's longest idle wait
  or it false-positives. `cfg.running=False` + `stop.set()` exits through the
  main `finally`, which drains `capture_q` first, then systemd restarts us.
- **Counts == DB.** Never drop a crossing: if the capture queue is full, persist
  it inline from the substream frame (`persist_capture(..., jpeg=None)`). A locked
  DB is retried (`PRAGMA busy_timeout=5000` + `_safe_store_add`), not fatal.
- **Restarts drain the queue.** `SIGTERM`/`SIGINT` set `cfg.running=False` and
  `stop`, so `capture_loop` (`while cfg.running or not capture_q.empty()`) still
  persists everything already counted before exit. Skipping the network still
  during drain keeps it fast. Handle the signal, or `finally` never runs.
- **No unbounded per-id state.** Track ids are never reused, so anything keyed by
  `track_id` (`last_cross_info`, `CrossingGate.*`) must be dropped when the
  tracker retires the id (`dead = gate_ids - present`), or it leaks for months.
- **A stream stall self-heals.** `TRACKER_READ_TIMEOUT` (15 s) forces a reconnect
  when no fresh frame arrives; an advancing container timestamp counts as alive,
  so a static scene is not treated as a stall. `stimeout`/`rw_timeout` bound
  `cap.read()`.
- **Offline never loses data.** `store.prune` deletes only `synced=1` rows; the
  sync worker deletes local files only after a `2xx`.
- **Permanent upload errors keep the row.** `sync.py` treats `4xx` as permanent,
  logs it, and records `sync_attempts`/`last_error`; only `2xx` marks synced.
- **Atomic ingest.** The VPS writes uploads to a temp file then renames, so a
  crash can't leave a truncated JPEG at a served path.
- **No `except: pass`.** Use `_log_throttled` (first hit, then once a minute) so
  config/live/prune/snapshot failures are diagnosable.

## Project
- Edge vehicle line-counter: RTSP -> detector (YOLO26n INT8 via ncnn) -> centroid
  tracker -> crossing gate -> offline SQLite -> HTTPS sync to a Bun/VPS dashboard.
- Edge runs on a Raspberry Pi 5 (DietPi); input is a CCTV camera's H.264 substream
  (ONVIF path `.../Streaming/Channels/102`, 640x360). The camera's RTSP Digest is
  SHA-256-only, so capture goes through `rtsp_relay.RtspRelay` (`"relay": true`).
- `run.py` is the production edge loop. Dev harnesses: `tools/live_push.py` (mimics it
  from a laptop, can push to the VPS) and `tools/test_clip.py` (runs the production
  tracker/crossing over a video file and reports IN/OUT).

## Deploying to the edge (Pi)
- The Pi has **no git** and `/opt/tracker` is **not a checkout**; the dashboard repo is
  private, so `raw.githubusercontent.com`/the GitHub API 404 unauthenticated. Don't try
  `git pull` or a raw fetch from the Pi.
- Deploy over the reverse terminal (`https://tracker.drnanoinc.com/terminal`, HTTP Basic;
  the browser drives the Pi's PTY via `term_agent.py`). It runs as the unprivileged
  `tracker` user with `NoNewPrivileges`, so **`sudo` is blocked** — you can't edit
  `/etc/tracker/*.env` or `systemctl restart` from there.
- **Patch over the terminal**: build the change locally, then generate a patch script from
  the exact `git diff` old→new block, and *verify locally* that applying it to the previous
  blob reproduces the target file's `sha256sum`. Base64 the script, write it to the Pi in
  `printf` chunks, apply it to `/opt/tracker/run.py`, and confirm the file's sha256 matches
  the commit. Read runtime settings from `/opt/tracker/config.json` and
  `/etc/tracker/tracker.env`.
- **Restart**: `kill "$(pgrep -f '[r]un.py')"`; `tracker.service` is `Restart=always` and
  comes back in ~5 s with the new code (counts live in SQLite, so nothing is lost). Verify
  with `ps -o pid,lstart,cmd -p "$(pgrep -f '[r]un.py')"`. The service's journal isn't
  readable by `tracker` (not in `systemd-journal`); use the dashboard/DB instead.
- Event rows (incl. `confidence`, `bbox`) live on the Pi at `/var/lib/tracker/events.db`
  and on the VPS; the VPS API doesn't expose them, so query the Pi's SQLite for detail.

## Hard-won gotchas
- Never disable PyTorch warnings or lower ncnn log level to silence issues; fix them.
- ncnn: keep the input `Mat` alive until after `extractor.extract()` (an inline
  temporary gets GC'd -> use-after-free -> one bad frame poisons the net into
  returning ~100 garbage boxes forever). Copy frames handed between threads.
- Ultralytics' ncnn export breaks non-square models; use `pnnx` directly.
- **INT8 is post-training only — calibrate, never fine-tune.** `tools/quantize_int8.py`
  (`ncnn2table` + `ncnn2int8`) rewrites just `model_int8.ncnn.bin` + `model_int8.table`
  from the fp32 graph; the calibration images only set per-tensor activation ranges,
  they do not train anything. The production model is calibrated on `calib/dumper_trucks/`
  (public dump-truck/tipper photos — dumper is the class the counter cares about; site
  captures are deliberately **not** used). Calibration images must already be at the
  model input size (320x320, letterboxed like inference); do not re-export the fp32 graph
  when only re-calibrating.
- The NVR has no MJPEG/JPEG endpoint; browsers cannot play raw RTSP — do not assume.
- **RTSP Digest is MD5-only in FFmpeg — SHA-256 cameras need the relay.** Every
  FFmpeg-backed reader (OpenCV `CAP_FFMPEG`, PyAV, `imageio-ffmpeg`, GStreamer `rtspsrc`)
  negotiates RTSP Digest with MD5 and gets `401` from a camera that challenges
  `algorithm="SHA-256"` (the PT-NC120D3-WNM(D2) does; credentials are valid). `rtsp_relay.py`
  does the SHA-256 `OPTIONS/DESCRIBE/SETUP/PLAY` handshake itself, asks the camera for
  RTP/UDP on a loopback port, writes an SDP, and keepalives with `GET_PARAMETER`; OpenCV
  decodes the SDP. Enable with `"relay": true` in `config.json` / `TRACKER_RTSP_RELAY=1`.
  OpenCV snapshots `OPENCV_FFMPEG_CAPTURE_OPTIONS` at process start, so `run.py` re-execs
  once with `protocol_whitelist;file,udp,rtp,rtsp,tcp` (guarded by `TRACKER_RELAY_REEXEC`) —
  setting the env in-process alone does nothing. The UDP return path needs camera and Pi on
  the same LAN. Discover an unknown RTSP path via ONVIF `GetStreamUri` (WS-UsernameToken).
- **The camera clock is Manual by default and can be badly off** (the PT-NC120D3 shipped
  5h30m behind — local time mistaken for UTC). `tools/set_camera_time.py` reads/sets it over
  ONVIF (`SetSystemDateAndTime` + `SetNTP`, WS-UsernameToken; password via
  `TRACKER_CAM_PASSWORD`). Keep it on NTP so it self-corrects.
- **Evidence captures come from the substream, not the main stream.** Counting and
  evidence both use the crossing substream frame now: one `<id>.jpg` (the crossing
  frame) plus a thumbnail built from the same frame, no `_sub` copy (`sub_path`
  NULL; the dashboard/`sync.py` tolerate it). This replaced the old default of
  fetching the camera's main-profile ONVIF snapshot (`/onvif-http/snapshot?Profile_1`):
  asking the camera for a main-profile still perturbs the counting substream on the
  PT-NC120D3 (partial/blurry JPEGs, occasional substream glitch). The main-stream
  fetch is now **opt-in** via `TRACKER_CAPTURE_MAIN_STREAM=1`; when enabled it still
  runs on the capture worker (never the track loop), reuses a `requests.Session`
  (keeps TLS + the digest nonce alive), skips a stale still (`TRACKER_CAPTURE_MAX_AGE`,
  default 1.5 s in the queue), and writes both `<id>.jpg` (main stream) and
  `<id>_sub.jpg` (crossing substream frame). The thumbnail is **always** built from
  the crossing substream frame, so the timeline shows the vehicle at the line.
- **One vehicle often produces two class-boxes — dedupe at the detector.** The
  INT8 model frequently fires two near-identical boxes with different winning
  classes (car+truck, bus+truck; IoU 0.76–0.93) for one vehicle. Per-class NMS
  keeps both, the tracker then mints a second track, and both cross the line → a
  double count. `YoloNcnn._merge_cross_class` runs a **class-agnostic NMS** at
  `TRACKER_DUP_IOU` (default 0.6) after the per-class pass — this is the
  Ultralytics `agnostic_nms` equivalent. A crossing-level de-dupe safety net
  (same direction, IoU > `TRACKER_DUP_IOU`, within `TRACKER_CROSS_DEDUP_S`,
  default 1 s) guarantees one vehicle = one event. Do **not** try to fix this in
  the tracker: it is a detector problem, and any tracker would mint two tracks from
  two boxes.
- **Offline is safe, but never prune unsynced rows.** `store.prune` deletes only
  `synced=1` events past `TRACKER_RETENTION_DAYS`; unsynced captures are the only
  copy while the VPS is unreachable and must be kept until `sync.py` gets a 2xx.
  Syncing deletes local files only after the server acknowledges.
- **A second camera can be shown raw (no detection) with no idle cost.**
  `TRACKER_CAM2_URL` (substream) + `CAM2=1` on the VPS render a second `/live`
  panel (`/api/live2`, `/live2.jpg`). The Pi opens the RTSP/relay session only while
  `live2_wanted` is set and tears it down ~5 s after the last viewer, so an idle edge
  streams nothing. The main feed's `live_wanted` gate is the same idea. On `/live`, left/right
  arrows switch between Main and Camera 2 and **only the visible feed is polled** — the idle
  feed's `*_wanted` flag goes stale within ~10 s, so the Pi stops uploading it (half the live
  data). Always opens on Main; the header LIVE dot tracks the active feed.
- **The `/timeline` view is counts-first; captures load per day on demand.** `GET /api/events`
  ships only per-day IN/OUT counts (polled every 8 s) and `GET /api/day?d=YYYY-MM-DD`
  returns one day's events (cached client-side). Don't re-embed the whole event list in the
  initial HTML — that was the old `/history` page (removed) and it made first paint and
  month-hopping slow. The UI is a horizontal day-square strip (oldest→newest, today
  auto-scrolled in), `←`/`→` steps a day and never goes into the future; days with zero
  captures are still shown. `/calendar` 302-redirects to `/timeline`.
- **The substream is a genuine ~25 fps — verify it, don't assume.** Measured on the
  current camera: `CAP_PROP_FPS` 25, PTS delta 40 ms/frame, ~25–29 frames/s delivered,
  and **0% bit-identical consecutive frames** over 10 s. So the tracker advances on
  essentially every frame; a "track fps" around 25 is correct and expected. An earlier
  note here claimed ~9 genuinely-new fps on a bursty NVR — that was wrong (it inferred
  duplicates from pixel changes on an idle scene, which says nothing about the encoder).
  To tell a *duplicated* frame from a *static* one, use the container timestamp
  (`CAP_PROP_POS_MSEC`), not pixel diffs; the reader's `fresh` flag is for the stall case
  below, not normal operation.
- **Two-tier association: IoU primary, centroid-distance fallback.** Frame-to-frame at
  ~25 fps a plain IoU tracker works, so tier 1 matches by IoU against the track's last box
  *translated by px/second velocity over the elapsed time* (`_pred_box`) — the raw last box
  has 0 IoU with a vehicle that moved during a stall, the extrapolated one often still
  overlaps. `TRACKER_IOU_GATE` (0.2, == ByteTrack's `match_thresh` 0.8) is the minimum.
  Tier 2 (fallback) is a centroid-distance gate that widens with the gap
  (`TRACKER_ASSOC_FRAC` + `TRACKER_ASSOC_GAP_FRAC`, capped by `TRACKER_ASSOC_MAX_FRAC`): a
  tracked vehicle whose box changed shape/speed so IoU misses is still re-acquired by
  proximity. Sort key is `(tier, cost)`, so every IoU match beats any fallback and the best
  per-tier match is chosen (greedy; `bytetracker` is not used — only the dev harnesses).
- **Crossing needs hysteresis, not a bare side-change — and only commit on accept.** A
  low-confidence box near the line jitters across it and registers false IN/OUT.
  `CrossingGate.crossing` commits each track to a side and requires the centroid to emerge >=
  `TRACKER_CROSS_MARGIN_FRAC` of the frame height past the line, within the drawn segment. The
  flip is only a *candidate*: `crossing()` records it and `allow()` commits it, so a candidate
  the flicker gate rejects leaves the track on its old side. Flipping eagerly (the old bug)
  armed the next micro-crossing in the opposite direction, so a parked/working machine on the
  line alternated IN/OUT every cooldown.
- **The margin must exceed the detector's box-refit wobble.** A stationary vehicle parked
  *on* the line is the pathological case: YOLO re-fits the box as its parts move (an
  excavator's arm swings in/out of the `truck` box), which drags the centroid tens of px
  across the line with no real transit. `TRACKER_CROSS_MARGIN_FRAC` default is `0.06`
  (~22 px on the 360p substream). If a loitering machine still false-counts — a big arm can
  swing the centroid more than the margin — the durable fix is to move the counting line off
  the machine or shrink the ROI, not to raise the margin further (that starts missing real
  crossings). See the 2026-10-09 `id7 truck in` false count.
- **A stream freeze must not drop the track — or it silently loses a count.**
  If the stream blocks for seconds, the vehicle moves far more than one frame's worth by the
  time frames resume; a raw-box IoU match then fails, a *new* track id is minted already past
  the line, and its crossing has no opposite-side history to fire on. `CentroidTracker` keeps
  velocity in px/second and extrapolates the box by wall-clock (`now`) for association, so the
  track (and its crossing side history) survives the gap. Do not revert to matching the stale
  last box.
- **Cap the coasted display position; ignore repeated frames.** The centroid used for the
  live overlay is extrapolated by at most `TRACKER_ASSOC_PRED_CAP` (0.5 s) so it can't fly
  off-frame (the IoU box prediction uses the full gap, since a wrong prediction only costs an
  IoU miss). If the camera repeats its last frame while stalled, feeding that identical image
  while wall-clock advances would collapse velocity to zero; the reader flags it `fresh=False`,
  the tracker is not advanced on it (the live relay still is, so the dashboard doesn't
  10 s-timeout), and the real gap is measured on the next genuinely-new frame.
- **Never let numpy scalars reach `json.dumps`/`store.add`.** `detect_roi` used to take
  `max(0, pts[:,0].min())` straight from a numpy ROI, so every bbox/centroid carried
  `np.int32`. `store.add`'s `json.dumps(bbox)` then raised, and because persistence runs
  inline in the main loop the process died (systemd restarted it) *after* the capture
  was written but *before* the event row — silently losing the crossing. ROI offsets are
  cast with `int()`, and `Store._json_safe` coerces any numpy value as a backstop. On the
  Pi, orphaned `captures/*.jpg` with no matching `events` row are the fingerprint.
- `ncnn`'s wheel declares the GUI `opencv-python` (needs libxcb); first-boot.sh replaces
  it with `opencv-python-headless`.
- Counting classes come from the dashboard (`enabled_classes`); the detect thread reads
  them live, so no restart is needed. Dumpers = COCO class 7 (truck). Two-wheelers
  (COCO 1 bicycle, 3 motorcycle) are filtered out in `run.py` *and* on the VPS, so the
  UI offers only car/bus/truck (2/5/7).
- **`/terminal` is a reverse WebSocket, not inbound SSH.** `term_agent.py` dials the VPS
  (`/api/term?role=agent`, Bearer) and bridges the socket to a local PTY; the browser
  attaches with a short-lived token from the Basic-auth page. Keep it outbound-only —
  never add a port-forward or bind the Pi. It is gated by `ENABLE_TERMINAL=1`, runs as
  the unprivileged `tracker` user, and xterm.js is self-hosted from `node_modules`
  (no CDN). WebSocketApp's send is thread-safe (`enable_multithread`), so the PTY pump
  thread may write while `run_forever` reads.
