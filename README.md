# Vehicle Line Counter

RTSP-based vehicle counting with line-crossing detection, YOLO26n inference via ncnn, and a web UI for live viewing and configuration. Designed for edge devices (Raspberry Pi Zero 2W).

## Features

- **RTSP stream input** — connect to any IP camera or NVR
- **YOLO26n detection** — lightweight, optimized via ncnn (or motion-only without ncnn)
- **ByteTrack** — IoU-based multi-object tracking
- **Motion gate** — only runs inference when motion is detected near the counting line (saves CPU)
- **Virtual counting line** — draw any angled line on the video; vehicles crossing it are counted
- **Bidirectional counting** — IN / OUT totals displayed live
- **Vehicle type filtering** — enable/disable car, motorcycle, bus, truck per detection
- **Web UI** — live MJPEG stream, canvas-based line drawing, live counters, capture gallery
- **Daily analytics** — `/analytics` page groups captures by day with expandable rows
- **Capture gallery** — `/captures` page with filterable grid, select & delete
- **Persistent config** — line position, stream URL, counters all saved to `config.json`
- **Rust port** — lower memory footprint, suitable for constrained devices (eg. 512 MB RAM)

---

## Python Version (Original)

### Quick Start

```bash
pip install -r requirements.txt
# edit config.json with your RTSP URL
python run.py
```

Open `http://<device-ip>:5000` in a browser.

---

## Rust Port (`rust_port/`)

Lower-memory alternative using OpenCV for RTSP capture. Object detection currently runs via motion-gating only (ncnn stub) — line counting, tracking, and the full web UI work without YOLO.

### Dependencies

| Toolchain | Requirement |
|-----------|-------------|
| Rust | `rustup target add aarch64-unknown-linux-gnu` |
| Cross-compiler | `gcc-aarch64-linux-gnu`, `g++-aarch64-linux-gnu` |
| OpenCV ARM64 libs | Extract from `ghcr.io/hybridgroup/opencv:4.13.0` (see `setup_opencv.sh` in `face_door_unlock` project) |
| libclang | `LIBCLANG_PATH` must point to a directory containing `libclang.so` |

### Cross-compile for Pi

```bash
cd rust_port
bash build_pi.sh
```

Produces `vehicle_counter_pi/` containing the binary + bundled OpenCV `.so` files.

### Deploy

```bash
# Copy to device
scp -r vehicle_counter_pi dietpi@<device-ip>:~/track_vehicles/rust_port/

# On device, run
cd ~/track_vehicles/rust_port/vehicle_counter_pi
./run.sh
```

### Native build (on device)

```bash
cd rust_port
cargo build --release
./target/release/vehicle_counter
```

### Test with a YouTube video

```bash
cd test
./run_test.sh
# or with a specific video:
./run_test.sh "https://www.youtube.com/watch?v=MNn9qKG2UFI"
```

The script uses `yt-dlp` to extract a direct stream URL, writes a config, and starts the server. Open `http://localhost:5000`, draw a counting line, and watch it count.

### Config

The same `config.json` from the Python version is used. Copy it alongside the binary:

```bash
cp config.json vehicle_counter_pi/
```

---

## Web UI

| Page | Path | Description |
|------|------|-------------|
| Live view | `/` | MJPEG stream with canvas line drawing, live counters, config |
| Captures | `/captures` | Gallery of all crossing snapshots (filterable, select & delete) |
| Analytics | `/analytics` | Daily totals with expandable rows |

### Drawing the counting line

1. Click **Draw Line**
2. Click two points on the video to define the line
3. Click **Save** to persist
4. Use **⇄ Flip IN/OUT Sides** to swap direction labels

### Vehicle types

Check/uncheck vehicle classes in the sidebar to filter which types are tracked.

---

## Configuration

All settings are stored in `config.json` and can be edited via the web UI:

| Key | Default | Description |
|-----|---------|-------------|
| `stream_url` | — | RTSP URL |
| `target_size` | `320` | YOLO input size (224/320/640) |
| `conf_thresh` | `0.5` | Detection confidence threshold |
| `motion_thresh` | `500` | Foreground pixel count to trigger motion |
| `flip_sides` | `false` | Swap IN/OUT direction |
| `enabled_classes` | `[2,3,5,7]` | Active vehicle COCO class IDs |

---

## Architecture (Rust port)

```
RTSP → opencv::VideoCapture → motion gate → [ncnn stub] → ByteTrack → line-crossing check → capture save
                                                                                                    ↓
                                                               MJPEG stream ← annotate frame ← count captures
```

- **Motion gate**: Background subtraction on a narrow zone around the counting line. YOLO only runs when pixel change exceeds threshold.
- **Tracking**: ByteTrack with IoU-based matching + velocity prediction + greedy Hungarian assignment.
- **Double-count prevention**: Minimum 15-frame gap between crossing events per tracked object.

---

## COCO Vehicle Classes

| ID | Class | Tracked by default |
|----|-------|--------------------|
| 2 | Car | ✓ |
| 3 | Motorcycle | ✓ |
| 5 | Bus | ✓ |
| 7 | Truck | ✓ |
| 1 | Bicycle | ✗ (can enable via UI) |

## Testing

Run unit tests:
```bash
cd rust_port
LIBCLANG_PATH=/home/skpawar1305/robostack/.pixi/envs/humble/lib cargo test
```

Run integration test with a YouTube video:
```bash
cd test
./run_test.sh "https://www.youtube.com/watch?v=MNn9qKG2UFI"
```

Then test API endpoints:
```bash
curl http://localhost:5000/api/counts     # {"in":0,"out":0}
curl http://localhost:5000/api/config     # current configuration
curl http://localhost:5000/api/line       # {"line": null} until you draw one
```

## License

MIT
