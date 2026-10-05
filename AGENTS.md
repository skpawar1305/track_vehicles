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
- Edge runs on a Raspberry Pi 5 (DietPi); input is the NVR's H.264 substream
  (`.../unicast/c11/s1/live`, currently 640x360 @20fps).
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
  them live, so no restart is needed. Dumpers = COCO class 7 (truck).
