# AGENTS

## Working style
- Always use best effort: act autonomously, make reasonable decisions, and take the
  task as far as possible without asking for hand-holding. Prefer a working result
  over questions when the path is clear.
- Follow the pareto principle: do the highest-value, lowest-effort work first.
- Be concise; this project is run from a terminal.

## Project
- Edge vehicle line-counter: RTSP -> detector (ncnn YOLO11n) -> ByteTrack -> crossing
  gate -> offline SQLite -> HTTPS sync to a Bun/VPS dashboard.
- Edge runs on a Raspberry Pi 5 (DietPi); input is the NVR's 640x360 H.264 substream
  (`.../unicast/c11/s1/live`).
- `run.py` is the production edge loop; `tools/live_push.py` is a dev harness that
  mimics it from a laptop.

## Hard-won gotchas
- Never disable PyTorch warnings or lower ncnn log level to silence issues; fix them.
- ncnn: keep the input `Mat` alive until after `extractor.extract()` (an inline
  temporary gets GC'd -> use-after-free -> one bad frame poisons the net into
  returning ~100 garbage boxes forever). Copy frames handed between threads.
- Ultralytics' ncnn export breaks non-square models; use `pnnx` directly.
- The NVR has no MJPEG/JPEG endpoint; browsers cannot play raw RTSP — do not assume.
