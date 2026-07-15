#!/bin/bash
set -e

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$TEST_DIR")"
RUST_DIR="$PROJECT_DIR/rust_port"
BINARY="$RUST_DIR/target/release/vehicle_counter"
CAPTURE_DIR="test_captures"

echo "===================================="
echo "Vehicle Counter Test"
echo "===================================="

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo "Building binary..."
    cd "$RUST_DIR"
    cargo build --release
fi

# Get YouTube URL
YT_URL="${1:-}"
if [ -z "$YT_URL" ]; then
    echo ""
    echo "Enter a YouTube video URL (or press Enter for default traffic cam):"
    read -r input_url
    if [ -z "$input_url" ]; then
        YT_URL="https://www.youtube.com/watch?v=Z9YHINfOhtA"
    else
        YT_URL="$input_url"
    fi
fi

echo ""
echo "Getting stream URL for: $YT_URL"
STREAM_URL=$(yt-dlp -g --format "best[height<=480]" "$YT_URL" 2>/dev/null | head -1)
if [ -z "$STREAM_URL" ]; then
    echo "ERROR: Could not extract stream URL from $YT_URL"
    exit 1
fi
echo "Stream URL: ${STREAM_URL:0:80}..."

# Write config to rust_port/ so the binary finds it
cat > "$RUST_DIR/config.json" <<CONFIG
{
  "stream_url": "$STREAM_URL",
  "line": null,
  "counts": { "in": 0, "out": 0 },
  "conf_thresh": 0.5,
  "flip_sides": false,
  "motion_thresh": 500,
  "target_size": 320,
  "capture_dir": "$TEST_DIR/$CAPTURE_DIR",
  "max_captures": 100,
  "model_path": "../models/yolo26n_ncnn_model",
  "enabled_classes": [2, 3, 5, 7]
}
CONFIG

# Clean up old captures
rm -rf "$TEST_DIR/$CAPTURE_DIR"

echo ""
echo "Starting vehicle_counter..."
echo "  Config: $RUST_DIR/config.json"
echo "  Web UI: http://localhost:5000"
echo ""
echo "1. Open http://localhost:5000 in a browser"
echo "2. Click 'Draw Line' and place two points on the video"
echo "3. Watch vehicles cross and counters increment"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$RUST_DIR"
exec "$BINARY"
