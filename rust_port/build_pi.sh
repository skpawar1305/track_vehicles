#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCV_DIR="/home/skpawar1305/face_door_unlock/face_door_unlock_rust/opencv_arm64"
TARGET="aarch64-unknown-linux-gnu"
APP_DIR="$SCRIPT_DIR/vehicle_counter_pi"

echo "============================================"
echo "Cross-compile vehicle_counter for Pi Zero 2W"
echo "============================================"

# Pre-flight checks
if ! command -v aarch64-linux-gnu-gcc &> /dev/null; then
    echo "ERROR: aarch64-linux-gnu-gcc not found!"
    exit 1
fi
if [ ! -d "$OPENCV_DIR/lib" ]; then
    echo "ERROR: OpenCV ARM64 libs not found at $OPENCV_DIR"
    exit 1
fi

echo "OpenCV ARM64: $OPENCV_DIR"
echo "Target: $TARGET"
echo ""

# Create pkg-config for cross-compilation
PKG_CONFIG_DIR=$(mktemp -d)
cat > "$PKG_CONFIG_DIR/opencv4.pc" << PKGCONFIG
prefix=$OPENCV_DIR
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include/opencv4
Name: OpenCV
Description: Open Source Computer Vision Library
Version: 4.13.0
Libs: -L\${libdir} -Wl,-rpath-link,\${libdir} -ltbb -lopencv_gapi -lopencv_face -lopencv_dnn -lopencv_objdetect -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_video -lopencv_flann -lopencv_features2d -lopencv_calib3d -lopencv_photo -lopencv_text -lopencv_plot
Libs.private: -ldl -lm -lpthread -lrt
Cflags: -I\${includedir}
PKGCONFIG

# Environment
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH="$PKG_CONFIG_DIR"
export PKG_CONFIG_LIBDIR="$PKG_CONFIG_DIR"
unset PKG_CONFIG_SYSROOT_DIR

export OPENCV_INCLUDE_DIR="$OPENCV_DIR/include/opencv4"
export OPENCV_LIB_DIR="$OPENCV_DIR/lib"
export OPENCV_LINK_LIBS="tbb,opencv_gapi,opencv_face,opencv_dnn,opencv_objdetect,opencv_imgcodecs,opencv_imgproc,opencv_core,opencv_videoio,opencv_highgui,opencv_video,opencv_flann,opencv_features2d,opencv_calib3d,opencv_photo,opencv_text,opencv_plot"
export RUSTFLAGS="-C link-args=-L$OPENCV_DIR/lib"
export LIBCLANG_PATH="/home/skpawar1305/robostack/.pixi/envs/humble/lib"
export LLVM_CONFIG_PATH="/usr/bin/llvm-config-21"

# Build
echo "[1/2] Building binary..."
cargo build --target "$TARGET" --release 2>&1

# Package
echo "[2/2] Packaging..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/lib" "$APP_DIR/captures"

cp "target/$TARGET/release/vehicle_counter" "$APP_DIR/"

# OpenCV libs (only needed modules)
for lib in \
    libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so \
    libopencv_highgui.so libopencv_videoio.so libopencv_dnn.so \
    libopencv_objdetect.so libopencv_face.so libopencv_flann.so \
    libopencv_features2d.so libopencv_calib3d.so libopencv_gapi.so \
    libopencv_video.so libopencv_photo.so libopencv_text.so \
    libopencv_plot.so libtbb.so; do
    if [ -f "$OPENCV_DIR/lib/$lib" ]; then
        cp -L "$OPENCV_DIR/lib/$lib" "$APP_DIR/lib/$lib"
        ln -sf "$lib" "$APP_DIR/lib/${lib}.413" 2>/dev/null
    fi
done

# Run script
cat > "$APP_DIR/run.sh" << 'RUN'
#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib"
exec "$DIR/vehicle_counter" "$@"
RUN
chmod +x "$APP_DIR/run.sh"

echo ""
echo "Done! $APP_DIR/"
echo "  $(du -sh $APP_DIR | cut -f1) total"
echo ""
echo "Deploy:"
echo "  scp -r $APP_DIR dietpi@192.168.0.232:~/track_vehicles/rust_port/"
echo ""
echo "Run on Pi:"
echo "  cd ~/track_vehicles/rust_port/vehicle_counter_pi && ./run.sh"
