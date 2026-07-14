# Vehicle Line Counter

RTSP-based vehicle counting with line-crossing detection, YOLO26n inference via ncnn, and a web UI for live viewing and configuration. Designed for edge devices (Raspberry Pi Zero 2W).

## Features

- **RTSP stream input** — connect to any IP camera or NVR
- **YOLO26n detection** — lightweight, optimized via ncnn
- **ByteTrack** — IoU-based multi-object tracking
- **Motion gate** — only runs inference when motion is detected near the counting line (saves CPU)
- **Virtual counting line** — draw any angled line on the video; vehicles crossing it are counted
- **Bidirectional counting** — IN / OUT totals displayed live
- **Vehicle type filtering** — enable/disable car, motorcycle, bus, truck per detection
- **Web UI** — live MJPEG stream, canvas-based line drawing, live counters, capture gallery
- **Daily analytics** — `/analytics` page groups captures by day with expandable rows
- **Capture gallery** — `/captures` page with filterable grid, select & delete
- **Persistent config** — line position, stream URL, counters all saved to `config.json`

## Quick Start

### 1. Export the model

On a machine with GPU and `ultralytics` installed:

```bash
cd models
./convert_yolo26n.sh
# or manually:
pip install ultralytics
yolo export model=yolo26n.pt format=ncnn imgsz=320
```

Copy the `models/yolo26n_ncnn_model/` directory to the target device.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

Edit `config.json`:

```json
{
  "stream_url": "rtsp://admin:password@192.168.0.229:554/unicast/c1/s1/live",
  "line": null,
  "target_size": 320,
  "conf_thresh": 0.5,
  "enabled_classes": [2, 3, 5, 7]
}
```

Or configure via the web UI after starting.

### 4. Run

```bash
python run.py
```

Open `http://<device-ip>:5000` in a browser.

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

## Architecture

```
RTSP → cap.read() → MOG2 motion gate → YOLO26n (ncnn) → ByteTrack → line-crossing check → capture save
                                                                                                    ↓
                                                               MJPEG stream ← annotate frame ← count captures
```

- **Motion gate**: MOG2 background subtraction on a narrow zone around the counting line (320px wide). YOLO only runs when pixel change exceeds threshold.
- **Tracking**: ByteTrack with IoU-based matching + velocity prediction + greedy Hungarian assignment.
- **Double-count prevention**: Minimum 15-frame gap between crossing events per tracked object.

## Performance (Raspberry Pi Zero 2W)

| Input size | Inference time | Est. FPS |
|------------|---------------|----------|
| 640×640 | ~350ms | ~3 FPS |
| 320×320 | ~120ms | ~8 FPS |
| 224×224 | ~75ms | ~13 FPS |

Motion gating means YOLO only runs when something moves near the line, so effective throughput depends on traffic density.

## COCO Vehicle Classes

| ID | Class | Tracked by default |
|----|-------|--------------------|
| 2 | Car | ✓ |
| 3 | Motorcycle | ✓ |
| 5 | Bus | ✓ |
| 7 | Truck | ✓ |
| 1 | Bicycle | ✗ (can enable via UI) |

## License

MIT
