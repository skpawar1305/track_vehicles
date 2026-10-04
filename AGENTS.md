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
- **Crossing needs hysteresis, not a bare side-change.** A low-confidence box near the
  line jitters across it and registers false IN/OUT. `CrossingGate.crossing` commits each
  track to a side and requires the centroid to emerge >= `TRACKER_CROSS_MARGIN_FRAC` of
  the frame height past the line, within the drawn segment.
- `ncnn`'s wheel declares the GUI `opencv-python` (needs libxcb); first-boot.sh replaces
  it with `opencv-python-headless`.
- Counting classes come from the dashboard (`enabled_classes`); the detect thread reads
  them live, so no restart is needed. Dumpers = COCO class 7 (truck).
