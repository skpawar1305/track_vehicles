# Vehicle Line Counter

RTSP-based vehicle counting with line-crossing detection, NanoDet-Plus-1.5x ONNX inference, ByteTrack (Kalman filter + Hungarian matching), and a web UI for live viewing and configuration.

## Features

- **RTSP / YouTube stream input** — connect to any IP camera, NVR, or video URL
- **NanoDet-Plus-m-1.5x detection** — 416×416 input, GhostPAN + SimSPPF neck, ShuffleNetV2 1.5x backbone
- **ByteTrack** — multi-object tracking via Kalman filter + Hungarian algorithm
- **Bbox + centroid double-gate** — crossing counted only when both centroid crosses the line AND bbox touches it
- **Bidirectional counting** — IN / OUT totals displayed live and persisted
- **Vehicle type filtering** — enable/disable car, motorcycle, bus, truck per COCO class
- **Web UI** — live MJPEG stream, canvas-based line drawing, live counters, capture gallery
- **Daily analytics** — `/analytics` page groups captures by day with expandable rows
- **Capture gallery** — `/captures` page with filterable grid, select & delete
- **Persistent config** — line position, stream URL, counters all saved to `config.json`

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
| `target_size` | `416` | NanoDet input size |
| `conf_thresh` | `0.5` | Detection confidence threshold |
| `motion_thresh` | `500` | Foreground pixel count to trigger motion |
| `flip_sides` | `false` | Swap IN/OUT direction |
| `enabled_classes` | `[2,3,5,7]` | Active vehicle COCO class IDs |

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

Test API endpoints with a running server:
```bash
curl http://localhost:5000/api/counts     # {"in":0,"out":0}
curl http://localhost:5000/api/config     # current configuration
curl http://localhost:5000/api/line       # {"line": null} until you draw one
```

## License

MIT
