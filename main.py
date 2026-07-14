import cv2
import numpy as np
import threading
import time
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|allowed_media_types;video"

from config import Config
from motion_detector import MotionDetector
from detector import YOLO26nDetector
from tracker import ByteTrack
from line_counter import detect_crossing, CROSSING_NONE, CROSSING_IN, CROSSING_OUT
from capture_manager import CaptureManager
from web_server import run_server

cfg = Config()


def annotate_frame(frame, line, objects, counts, fps, det_count, flip_sides=False):
    h, w = frame.shape[:2]

    if line:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (59, 130, 246), 3)

        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy * 30, ux * 30
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            in_pos = (mx + int(px), my + int(py))
            out_pos = (mx - int(px), my - int(py))
            if flip_sides:
                in_pos, out_pos = out_pos, in_pos
            cv2.putText(frame, "IN", (in_pos[0] - 10, in_pos[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)
            cv2.putText(frame, "OUT", (out_pos[0] - 20, out_pos[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2)

            zone_ext = 40
            zone_w = 30
            zpts = np.array([
                [int(x1 - ux * zone_ext - px * zone_w / 30), int(y1 - uy * zone_ext - py * zone_w / 30)],
                [int(x1 - ux * zone_ext + px * zone_w / 30), int(y1 - uy * zone_ext + py * zone_w / 30)],
                [int(x2 + ux * zone_ext + px * zone_w / 30), int(y2 + uy * zone_ext + py * zone_w / 30)],
                [int(x2 + ux * zone_ext - px * zone_w / 30), int(y2 + uy * zone_ext - py * zone_w / 30)],
            ])
            cv2.polylines(frame, [zpts], True, (136, 136, 136), 1)

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
        label = f"{obj.label} #{obj.track_id}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 2)
        cv2.circle(frame, obj.centroid, 4, (251, 191, 36), -1)

    cv2.putText(frame, f"IN: {counts['in']}", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2)
    cv2.putText(frame, f"OUT: {counts['out']}", (w - 150, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239, 68, 68), 2)
    cv2.putText(frame, f"{fps:.1f} FPS  Det: {det_count}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136, 136, 136), 1)

    return frame


def main():
    print("[main] Starting Vehicle Line Counter")
    print(f"[main] Stream: {cfg.stream_url}")

    detector = None
    model_path = cfg.model_path

    def _find_model(path):
        if os.path.isdir(path):
            has_param = any(f.endswith('.param') for f in os.listdir(path))
            has_bin = any(f.endswith('.bin') for f in os.listdir(path))
            return has_param and has_bin
        return os.path.exists(f"{path}.param") and os.path.exists(f"{path}.bin")

    if _find_model(model_path):
        try:
            detector = YOLO26nDetector(model_path, conf_thresh=cfg.conf_thresh,
                                       target_size=cfg.target_size)
            print(f"[main] Detector loaded from {model_path} (target_size={cfg.target_size})")
        except Exception as e:
            print(f"[main] Failed to load detector: {e}")
    else:
        print(f"[main] Model files not found at {model_path}")
        print("[main] Run models/convert_yolo26n.sh to generate them")

    tracker = ByteTrack(conf_thresh=cfg.conf_thresh)
    motion_detector = MotionDetector(threshold=cfg.motion_thresh)
    capture_mgr = CaptureManager(cfg.capture_dir, cfg.max_captures)

    server_thread = threading.Thread(target=run_server, args=(cfg,), daemon=True)
    server_thread.start()
    print(f"[main] Web UI at http://0.0.0.0:5000")

    if cfg.line:
        motion_detector.update_line(cfg.line)

    frame_count = 0
    fps_timer = time.time()
    current_fps = 0.0
    det_count = 0
    periodic_interval = 30
    cap = None

    try:
        while cfg.running:
            if cap is not None and cfg.check_reconnect():
                print("[main] Stream URL changed, reconnecting...")
                cap.release()
                cap = None

            if cap is None:
                cap = cv2.VideoCapture(cfg.stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print(f"[main] Failed to open stream, retrying in 5s...")
                    cap.release()
                    cap = None
                    time.sleep(5)
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[main] Stream connected")

            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"[main] Read error: {e}, reconnecting...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            if not ret:
                print("[main] Stream disconnected, reconnecting...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                current_fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            if cfg.line and motion_detector.line != cfg.line:
                motion_detector.update_line(cfg.line)

            objects = []
            det_count = 0
            run_detection = False

            if cfg.line is not None:
                motion_detector.detect(frame)
                run_detection = (
                    motion_detector.motion_state
                    or (detector and frame_count % periodic_interval == 0)
                )

            if run_detection and detector:
                detections = detector.detect(frame, enabled_classes=cfg.enabled_classes)
                det_count = len(detections)
                tracked = tracker.update(detections)
                objects = tracked

            cap_dir = cfg.capture_dir
            c_in = c_out = 0
            if os.path.isdir(cap_dir):
                for fn in os.listdir(cap_dir):
                    if fn.endswith('_in.jpg'):
                        c_in += 1
                    elif fn.endswith('_out.jpg'):
                        c_out += 1

            annotated = annotate_frame(
                frame.copy(), cfg.line, objects, {"in": c_in, "out": c_out},
                current_fps, det_count,
                flip_sides=cfg.flip_sides
            )

            if cfg.has_viewers() and frame_count % 2 == 0:
                ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    cfg.push_jpeg(jpeg.tobytes())

            if run_detection and detector:
                for obj in tracked:
                    if obj.prev_centroid and obj.age >= 3:
                        crossing = detect_crossing(
                            cfg.line, obj.prev_centroid, obj.centroid,
                            flip=cfg.flip_sides
                        )
                        if crossing != CROSSING_NONE:
                            if frame_count - obj.last_crossing_frame < 15:
                                continue
                            obj.last_crossing = 'in' if crossing == CROSSING_IN else 'out'
                            obj.last_crossing_frame = frame_count
                            direction = 'IN' if crossing == CROSSING_IN else 'OUT'
                            capture_mgr.save(annotated, obj.track_id, crossing, obj.bbox)
                            print(f"[cross] ID#{obj.track_id} {direction}")
            else:
                if cfg.line is not None:
                    tracker.update([])

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
    except Exception as e:
        print(f"[main] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cfg.running = False
        if cap is not None:
            cap.release()
        print("[main] Done")


if __name__ == "__main__":
    main()
