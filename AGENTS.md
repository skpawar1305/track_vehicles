# AGENTS

## Working style
- Always use best effort: act autonomously, make reasonable decisions, and take the
  task as far as possible without asking for hand-holding. Prefer a working result
  over questions when the path is clear.
- Follow the pareto principle: do the highest-value, lowest-effort work first.
- Be concise; this project is run from a terminal.

## Project
- Edge vehicle line-counter: RTSP -> detector (YOLO26n INT8 via ncnn) -> centroid
  tracker -> crossing gate -> offline SQLite -> HTTPS sync to a Bun/VPS dashboard.
- Edge runs on a Raspberry Pi 5 (DietPi); input is a CCTV camera's H.264 substream
  (ONVIF path `.../Streaming/Channels/102`, 640x360). The camera's RTSP Digest is
  SHA-256-only, so capture goes through `rtsp_relay.RtspRelay` (`"relay": true`).
- `run.py` is the production edge loop. Dev harnesses: `tools/live_push.py` (mimics it
  from a laptop, can push to the VPS) and `tools/test_clip.py` (runs the production
  tracker/crossing over a video file and reports IN/OUT).

## Hard-won gotchas
- Never disable PyTorch warnings or lower ncnn log level to silence issues; fix them.
- ncnn: keep the input `Mat` alive until after `extractor.extract()` (an inline
  temporary gets GC'd -> use-after-free -> one bad frame poisons the net into
  returning ~100 garbage boxes forever). Copy frames handed between threads.
- Ultralytics' ncnn export breaks non-square models; use `pnnx` directly.
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
- **Evidence captures use the main stream, counting uses the substream.** At each
  crossing a worker thread fetches a full-res still from the camera's ONVIF
  snapshot (`/onvif-http/snapshot?Profile_1`, camera-side JPEG, SHA-256 Digest) —
  so no extra Pi decode — and falls back to the substream frame if the fetch
  fails. Keep the fetch off the main loop (a network call would stall tracking).
- **A second camera can be shown raw (no detection) with no idle cost.**
  `TRACKER_CAM2_URL` (substream) + `CAM2=1` on the VPS render a second `/live`
  panel (`/api/live2`, `/live2.jpg`). The Pi opens the RTSP/relay session only while
  `live2_wanted` is set and tears it down ~5 s after the last viewer, so an idle edge
  streams nothing. The main feed's `live_wanted` gate is the same idea.
- **Tracking at low fps: do not use IoU association.** At 10-20 fps a vehicle can move
  farther than its own box between samples, so ByteTrack (IoU) returns nothing after the
  first frame and fast movers are never counted. `run.py` uses a velocity-aware
  *centroid* tracker (`CentroidTracker`) instead. The edge no longer needs `bytetracker`
  (only the dev harnesses do).
- **Crossing needs hysteresis, not a bare side-change.** A low-confidence box near
  the line jitters across it and registers false IN/OUT. `CrossingGate.crossing` commits each
  track to a side and requires the centroid to emerge >= `TRACKER_CROSS_MARGIN_FRAC` of the
  frame height past the line, within the drawn segment.
- **A stream freeze must not drop the track — or it silently loses a count.**
  If the stream blocks for seconds, the vehicle has moved far more than one frame's
  worth by the time frames resume; a per-frame centroid prediction then misses the
  detection, a *new* track id is minted already past the line, and its crossing has no
  opposite-side history to fire on. `CentroidTracker` therefore keeps velocity in
  px/second and extrapolates by wall-clock (`now`), widening the association gate with
  the gap (`TRACKER_ASSOC_GAP_FRAC`, capped by `TRACKER_ASSOC_MAX_FRAC`). Do not revert
  to per-frame velocity/prediction.
- **…but cap the extrapolation and ignore repeated frames.** Linear extrapolation over
  a multi-second gap can fling the predicted centroid off-frame, past the widened gate,
  so the detection is still missed: `TRACKER_ASSOC_PRED_CAP` (default 0.5 s) bounds how
  far a track advances per update (the widened gate does the re-acquisition, so the cap
  can be short). And if the NVR repeats its last frame while stalled,
  feeding that identical image while wall-clock advances collapses velocity to zero;
  the reader flags it `fresh=False`, the tracker is not advanced on it (the live relay
  still is, so the dashboard doesn't 10 s-timeout), and the real gap is measured on the
  next genuinely-new frame.
- **Never let numpy scalars reach `json.dumps`/`store.add`.** `detect_roi` used to take
  `max(0, pts[:,0].min())` straight from a numpy ROI, so every bbox/centroid carried
  `np.int32`. `store.add`'s `json.dumps(bbox)` then raised, and because persistence runs
  inline in the main loop the process died (systemd restarted it) *after* the capture
  was written but *before* the event row — silently losing the crossing. ROI offsets are
  cast with `int()`, and `Store._json_safe` coerces any numpy value as a backstop. On the
  Pi, orphaned `captures/*.jpg` with no matching `events` row are the fingerprint.
- The NVR substream is **bursty**: it repeats bit-identical frames (median changed-pixels
  = 0) and only emits ~9 genuinely-new frames/s even when 20 fps is reported. This is why
  the freeze/repeat handling matters — and why "track fps" (distinct frames processed) is
  legitimately lower than the stream fps.
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
