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
    echo "Building binary (this may take a while)..."
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
        YT_URL="https://www.youtube.com/watch?v=MNn9qKG2UFI"
    else
        YT_URL="$input_url"
    fi
fi

echo ""
echo "Getting video stream URL for: $YT_URL"
STREAM_URL=$(yt-dlp --extractor-args "youtube:player_client=android" \
    -g --format "best[height<=480]" "$YT_URL" 2>/dev/null | tail -1)

if [ -z "$STREAM_URL" ]; then
    echo "WARNING: Could not extract stream URL."
    echo "The server will start but won't have video input."
    echo "You can still test the API endpoints."
    STREAM_URL=""
fi
echo "Stream URL: ${STREAM_URL:0:60}..."

# Write config using Python to handle special chars in URL
cd "$RUST_DIR"
python3 -c "
import json, os
config = json.load(open('config.json')) if os.path.exists('config.json') else {}
config['stream_url'] = '$STREAM_URL'
config['line'] = None
config['capture_dir'] = '$TEST_DIR/$CAPTURE_DIR'
config['model_path'] = '../models/yolo26n_ncnn_model'
config.setdefault('counts', {'in': 0, 'out': 0})
config.setdefault('conf_thresh', 0.5)
config.setdefault('flip_sides', False)
config.setdefault('motion_thresh', 500)
config.setdefault('target_size', 320)
config.setdefault('max_captures', 100)
config.setdefault('enabled_classes', [2, 3, 5, 7])
json.dump(config, open('config.json', 'w'), indent=2)
print('Config written to config.json')
"

# Clean up old captures
rm -rf "$TEST_DIR/$CAPTURE_DIR"

echo ""
echo "Starting vehicle_counter on http://localhost:5000..."
echo ""
echo "  To draw a counting line:"
echo "    1. Open http://localhost:5000"
echo "    2. Click 'Draw Line'"
echo "    3. Click two points on the video"
echo "    4. Click 'Save'"
echo ""
echo "  API endpoints:"
echo "    http://localhost:5000/api/counts"
echo "    http://localhost:5000/api/config"
echo "    http://localhost:5000/api/line"
echo "    http://localhost:5000/api/captures"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec "$BINARY"
