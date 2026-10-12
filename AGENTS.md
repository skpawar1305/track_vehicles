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
  `track_id` (`CrossingGate`, `recent_cross`) must be dropped when the tracker
  retires the id (`dead = gate_ids - present`), or it leaks for months.
- **A stream stall self-heals.** `TRACKER_READ_TIMEOUT` (15 s) forces a reconnect
  when no fresh frame arrives; an advancing container timestamp counts as alive,
  so a static scene is not treated as a stall. `stimeout`/`rw_timeout` bound
  `cap.read()`.
- **Offline never loses data.** `store.prune` deletes only `synced=1` rows; the
  sync worker deletes local files only after a `2xx`. A successful upload
  deletes the image, so `synced=1` with the image *still on disk* means the mark
  never reached the server (the outage bug); `sync.py` sweeps those rows
  (`Store.requeue_present_images`, throttled by `TRACKER_SYNC_RESYNC_INTERVAL`)
  and re-uploads them — idempotent, since the VPS de-dupes by id. A missing
  image is retried, never silently dropped, until `TRACKER_SYNC_GIVEUP_HOURS`.
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
- **A blurry/blocky stream is the substream encoder or UDP loss — not TCP, and check it with
  `tools/onvif_media.py`.** The relay already negotiates `RTP/AVP/UDP;unicast`
  (`rtsp_relay.py`), so there is no TCP→UDP switch; TCP would show as *freezes*, UDP loss as
  *blur*. `tools/onvif_media.py` (ONVIF `GetVideoEncoderConfigurations`/`GetProfiles`/
  `GetStreamUri`, WS-UsernameToken, creds from `config.json` `stream_url` or
  `TRACKER_CAM_PASSWORD`) prints each profile's resolution, `BitrateLimit`, `FrameRateLimit`
  and `GovLength`. Measured on the direct PT-NC120D3 (substream profile `Profile_2` /
  `Channels/102`, encoder token `VideoEncoderToken_2`): it offers only **`640x360` and
  `640x480`** (there is no `640x320`); production now runs **`640x360`, 2048 kbps, GOP=25**
  (was `640x480`/1024/50) to cut decode and compression blur. Set these with
  `tools/set_encoder.py` (`SetVideoEncoderConfiguration`, dry-run by default — it re-sends the
  camera's own XML with only the target tags substituted). Fixes for lingering blur: shorten
  the GOP (faster resync), raise the UDP receive buffer, or get the camera/Pi off Wi-Fi.
  **The camera downscales the same field of view rather than cropping**, so the normalized
  line/ROI stay valid across a resolution change — no re-draw needed. **When onboarding a new
  camera, run `tools/onvif_media.py` first** to record its substream bitrate/GOP.
- **The main stream can't rescue the Pi, and this camera offers no workaround.** Measured
  PT-NC120D3 capabilities (ONVIF): only two **H.264** encoders — main `VideoEncoderToken_1` /
  `Profile_1` / `Channels/101` (only `1280x720` or `1920x1080`) and sub `VideoEncoderToken_2` /
  `Profile_2` / `Channels/102` (`640x360`/`640x480`). **No MJPEG/JPEG** (common HTTP MJPEG paths
  return 404; ONVIF has no JPEG encoder — only `/onvif-http/snapshot` exists), **no PTZ**
  (`GetCapabilities` has no PTZ service), and the ONVIF **source `Bounds` is fixed**
  (`BoundsRange` `Width 1920..1920`, `XRange 0..0`), so it cannot crop/scale — every profile
  encodes the full field of view. The **Pi 5 has no hardware H.264 decoder** (no `/dev/video*`,
  no V4L2/ffmpeg/GStreamer), so the main stream is software-decoded: measured **720p main ≈
  14–15 fps at 73–76 °C** (sits on the 75 °C `step_wise` trip, decode-limited) vs the
  substream's **~19 fps at ~66 °C**, and changing the camera fps (`20` vs `25`) does **not** help
  — the Pi decodes every frame it receives, so it is decode-limited not source-limited.
  **Decode is whole-frame**: H.264 inter-frame prediction references pixels outside the ROI and
  entropy coding is sequential, so the ROI can only be cropped *after* decode; a crop saves
  inference input, never decode cost. Net: keep the Pi on the substream — there is no crop /
  zoom / MJPEG trick. (A phone edge with a hardware decoder was tried and rejected too: its
  1080p software YUV→RGB path capped at ~12 fps and throttled, with inference no faster than
  the Pi's.) The Prama/Hikvision-OEM NVR cannot offload *decode* (it only relays the camera's
  H.264), but it **can** serve the counting substream directly once Stream Encryption is off
  — see the NVR bullet below.
- **The NVR can be the edge's RTSP source — turn Stream Encryption off first.** The Prama NVR
  (`PT-NRAS2A04-K1` @ `192.168.225.150`) challenges RTSP Digest with **MD5**, so FFmpeg/OpenCV
  authenticate directly and `rtsp_relay` is not needed (unlike the SHA-256-only cameras). Channel
  map: **ch1 = `192.168.225.70` (embankment), ch2 = `192.168.225.71` (dumper site)**; substreams
  are `640x360` (`/Streaming/Channels/202`). **`Platform Access ▸ Stream Encryption` must be
  OFF**: with it on the NVR advertises a valid H.264 SDP `sprop` but scrambles the RTP payload
  (in-band SPS `profile_idc=150`, slices referencing a nonexistent PPS), so decoders get pure
  green/near-black frames — it is *not* an auth or codec problem. The NVR's ONVIF is **HTTP-only**
  (`http://192.168.225.150/onvif/device_service`); `tools/onvif_media.py` defaults to `https://`,
  so query it over HTTP. Full-res evidence: `http://192.168.225.150/onvif/snapshot/201`.
- **Evidence saves the main-stream image, thumbnails from the substream.** The saved
  `<id>.jpg` is the camera's full-resolution main-profile ONVIF snapshot
  (`/onvif-http/snapshot?Profile_1`); the thumbnail is **always** built from the
  crossing substream frame, so the timeline shows the vehicle at the line. Set
  `TRACKER_CAPTURE_MAIN_STREAM=0` to save the substream frame as the full image too:
  on the PT-NC120D3 asking the camera for a main-profile still can perturb the
  counting substream (partial/blurry JPEGs, occasional substream glitch), which is
  why that was once the default. The fetch runs on the capture worker (never the
  track loop), reuses a `requests.Session` (keeps TLS + the digest nonce alive),
  skips a stale still (`TRACKER_CAPTURE_MAX_AGE`, default 1.5 s in the queue), falls
  back to the substream frame, and writes `<id>.jpg` (main stream) plus `<id>_sub.jpg`
  (crossing substream frame) when it succeeds.
- **One vehicle often produces two class-boxes — dedupe at the detector.** The
  INT8 model frequently fires two near-identical boxes with different winning
  classes (car+truck, bus+truck; IoU 0.76–0.93) for one vehicle. Per-class NMS
  keeps both, the tracker then mints a second track, and both cross the line → a
  double count. `YoloNcnn._merge_cross_class` runs a **class-agnostic NMS** at
  `TRACKER_DUP_IOU` (default 0.6) after the per-class pass — this is the
  Ultralytics `agnostic_nms` equivalent. A crossing-level de-dupe safety net
  (same direction, IoU > `TRACKER_DUP_IOU`, **and `same_vehicle_box`** — centres
  within `TRACKER_DEDUP_CENTER_FRAC` and areas within 2x, so a queued follower is
  not collapsed — within `TRACKER_CROSS_DEDUP_S`, default 1 s) guarantees one
  vehicle = one event. Do **not** try to fix this in the tracker: it is a detector
  problem, and any tracker would mint two tracks from two boxes.
- **NanoDet can merge a vehicle into a lingering "container" box — drop by containment.**
  The `nanodet`/`opencv` detectors (`config.json` `"detector"`) sometimes emit a loose box
  around a vehicle — or spanning two adjacent trucks — whose IoU with the tight box is low,
  so plain IoU-NMS keeps both and the tracker mints a phantom track. Both paths drop any box
  mostly inside a higher-confidence one (intersection-over-min-area > 0.6). The production
  `hybrid` path uses YOLO and relies on `_merge_cross_class` instead.
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
  data). Always opens on Main; the header LIVE dot tracks the active feed. Because `/live`
  polls a JPEG the server holds in memory, the first frame is **instant while warm** (any
  `/live.jpg` fetch in the last ~10 s) but ~3.5 s **cold** — the Pi only learns `live_wanted`
  on its 3 s config poll, then encodes+POSTs; it stops ~10 s after the last viewer.
- **The dashboard is a static SvelteKit SPA served by `server.ts`, not server-rendered HTML.**
  `web/` is a SvelteKit app (`@sveltejs/adapter-static`, `ssr=false`, `/` falls back to
  `index.html`) built to `web/build/`; the Bun API server `web/server.ts` serves those assets
  (`WEB_ROOT`, default `<server.ts dir>/spa`) and keeps `/api/*`, `/img|thumb|sub`,
  `/live*.jpg`, and the reverse-terminal WS unchanged. Build with `bun run build`, copy
  `build/` to `WEB_ROOT` (+ `server.ts`), restart `tracker-web.service`. Do **not** add
  `+server.ts` / `+*server.ts` files to the SvelteKit app — adapter-static forbids server
  routes; the API lives only in `server.ts`.
- **The `/timeline` view is counts-first; captures load per day on demand.** The SPA fetches
  `GET /api/events` (per-day IN/OUT counts, polled every 8 s) and `GET /api/day?d=YYYY-MM-DD`
  for the open day (cached client-side, stale-while-revalidate, so day-hopping never flashes
  "Loading"). Don't re-embed the whole event list in the initial page — that was the old
  `/history` page (removed). The UI is a horizontal day-square strip (oldest→newest, today
  auto-scrolled in), `←`/`→` steps a day and never into the future; days with zero captures
  are still shown. Each capture has an **Ignore** toggle (`POST /api/ignore`) that keeps the
  card in the gallery (dimmed) but drops it from every IN/OUT total (day badge, tabs, header,
  `/api/config`); the toggle is optimistic (in-place, no gallery rebuild, no scroll jump).
  `/calendar` 302-redirects to `/timeline`.
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
- **The crossing gate is three rules: hysteresis + confirmed transit + re-arm.** Earlier
  versions stacked min-age + min-travel-from-birth + cooldown + a 15-frame refractory + this
  same re-arm; the extra gates were either redundant or wrong (net-travel from *birth* never
  constrains a reversal), so they were removed. What remains, in `CrossingGate.update()`:
  * **hysteresis:** a track commits to a side the first time it is seen and only *candidates*
    a flip when the centroid emerges >= `TRACKER_CROSS_MARGIN_FRAC` of the frame height past
    the line, within the drawn segment. A rejected candidate leaves the track on its old side.
  * **confirmed transit:** the candidate only *counts* once the centroid reaches
    `TRACKER_CONFIRM_FRAC` (0.2) of the frame height past the line — *deeper* than the
    hysteresis margin, on the opposite side. Short of that it is only *pending*; a return
    toward the original side (or into the band) cancels it. This is what kills the ~1 s
    opposite-direction double-counts: clearing the tiny 0.06 band is no longer enough to
    commit, so a working machine's swing (or a bbox refit that flicks the centroid across)
    never produces an OUT/IN pair. `0` restores the legacy "count at the margin".
  * **re-arm:** after a committed crossing the track must move `TRACKER_CROSS_REARM_FRAC`
    (0.2) of the frame height *away from the crossing point* (`last_pt` + `rearm_remaining`)
    before another flip may commit. `0` disables it.
  `min_age` is the only other gate (the run.py pre-filter and inside `update()`); it is not a
  travel/cooldown heuristic. `tools/test_crossing.py` asserts wobble=1, passthru=1,
  true-return=2, nudge=1, rapid-reversal=0, and a queued pair=2; don't regress.
- **Don't confuse the re-arm distance with the margin.** `0.15` was too small on this site: a
  slow dumper that overshot the line by ~55px re-armed and re-counted (`id6 out/in`, 12:54).
  `0.2` (72px on 360p) covers that swing without missing real returns.
- **The excavator working on the line is the durable false-count source; use the keep-out.**
  The 2026-10-12 14:12 `id275 out` 14:12:55 → `id275 in` 14:12:56 pair was a JCB backhoe
  loader swinging its arm across the line (both evidence frames show it, boom extended): one
  machine oscillation committed two opposite crossings a second apart (net zero, but each
  bumps the totals). The keep-out polygon (`norm_keepout`, drawn with the **Keep-out** tool in
  Camera setup and pulled from the server like the line; `config.json` is the fallback) is a
  polygon where the gate **freezes** — no side change commits while the centroid is inside it.
  Place it over the machine's work area. `0.2` confirm distance plus the keep-out together
  neutralise it; the deeper fix is still to move the line off the machine.
- **Zone-transition counting.** Drawing both an **Entry zone** and an **Exit zone** in Camera
  setup (`entry_zone`/`exit_zone` in the config, normalized) switches counting from
  `CrossingGate` to `ZoneGate`: a track is committed to a zone only while its centroid is
  `TRACKER_CROSS_MARGIN_FRAC` of the frame height *inside* it (the gap between the two zones is
  the hysteresis), and a committed move from one zone to the other counts once the centroid is
  `TRACKER_CONFIRM_FRAC` inside the destination. Entry = IN, Exit = OUT; `flip_sides`/`IN side`
  swaps. This is the robust fix for a working machine (it stays in one zone and never
  transitions) and for a vehicle that appears already past the line (it must be seen in the
  other zone first). First sight inside a zone commits without counting. The line remains the
  fallback when the zones aren't drawn; keep-out still applies. `tools/test_zones.py` covers it.
- **Overlap alone can suppress a queued follower — hence the centre/size check.** The
  same-direction crossing de-dupe (`TRACKER_CROSS_DEDUP_S` + `TRACKER_DUP_IOU`) collapses the
  two near-identical class-boxes of one vehicle, but two genuinely queued trucks can overlap
  too. `same_vehicle_box()` therefore also requires the centres within
  `TRACKER_DEDUP_CENTER_FRAC` (0.03) of the frame width AND areas within a 2x ratio, so a
  follower (different centre) still counts. `TRACKER_TRACK_MAX_AGE` was raised 25 → 40 so a
  briefly occluded truck keeps its id (and gate history) instead of being reminted already
  past the line (a guaranteed under-count).
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
- **Pi 5 runs hot; the knob you can actually turn is `threads`.** `detect_loop`
  runs ncnn unthrottled on `TRACKER_THREADS` cores (default 4) *plus* software H.264
  decode, so the SoC parks at the `temp_limit` in `/boot/firmware/config.txt` (75 °C
  here; it's a kernel `step_wise` trip). The 320×320 model scales sublinearly, so
  `threads: 2` cuts heat/CPU with no meaningful loss of inf/s (measured ~24 inf/s at
  4 threads with production also running). `tracker.env` is root-owned, so read
  `threads` (and `model`/`model_imgsz`/`int8`) from `config.json` (tracker-writable)
  instead. `run.py` also writes
  `/tmp/tracker_stats.json` (`TRACKER_STATS_PATH`) every 10 s with track/detect fps,
  tracks, temp and session in/out — the service journal is **not** readable by the
  `tracker` user, so this file (or the DB) is how you see rates on-device.
- **640×640 is ~9 fps on the Pi 5, not 25; pick the model via `config.json`.** The
  detector input scales linearly with pixels, so 640 is 4× the 320 work: measured
  `detect_fps` ~9 (vs ~24 at 320) at 2 threads, ~9 at 4 threads too — 4 threads just
  runs hotter (75.5 °C). The tracker still runs at the frame rate the detector feeds
  it (it coasts between detections), so ~9 fps is fine for this slow-dumper site but
  cuts the live preview to ~9 fps. Select it without root via
  `config.json` `{"model":"yolo26n","model_imgsz":"640x640","int8":true}` (the model
  dir must exist); revert `model_imgsz` to `320x320` to go back.
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
