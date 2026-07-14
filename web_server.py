import cv2
import os
import time
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, stream_with_context

app = Flask(__name__)


def create_server(config):
    app.config['cfg'] = config
    return app


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/captures')
def captures_page():
    return render_template('captures.html')


@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')


@app.route('/video_feed')
def video_feed():
    cfg = app.config['cfg']
    cfg.viewer_connected()

    def generate():
        try:
            while cfg.running:
                jpeg_bytes = cfg.pop_jpeg()
                if jpeg_bytes is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           jpeg_bytes + b'\r\n')
                else:
                    time.sleep(0.02)
        finally:
            cfg.viewer_disconnected()

    return Response(
        stream_with_context(generate()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
        }
    )


@app.route('/api/line', methods=['GET', 'POST'])
def api_line():
    cfg = app.config['cfg']
    if request.method == 'POST':
        data = request.get_json()
        if data and 'line' in data:
            line = data['line']
            if line is None or len(line) == 0:
                cfg.set_line(None)
                return jsonify({'status': 'ok', 'line': None})
            if len(line) == 4:
                cfg.set_line([int(v) for v in line])
                return jsonify({'status': 'ok', 'line': cfg.line})
        return jsonify({'status': 'error', 'message': 'Invalid line data'}), 400
    return jsonify({'line': cfg.line})


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    cfg = app.config['cfg']
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error'}), 400
        if 'stream_url' in data:
            cfg.set_stream_url(data['stream_url'])
            return jsonify({'status': 'ok', 'stream_url': cfg.stream_url})
        if 'flip_sides' in data:
            cfg.set_flip_sides(bool(data['flip_sides']))
            return jsonify({'status': 'ok', 'flip_sides': cfg.flip_sides})
        if 'enabled_classes' in data:
            cfg.set_enabled_classes(data['enabled_classes'])
            return jsonify({'status': 'ok', 'enabled_classes': cfg.enabled_classes})
        return jsonify({'status': 'error', 'message': 'Invalid config'}), 400
    return jsonify({
        'stream_url': cfg.stream_url,
        'line': cfg.line,
        'conf_thresh': cfg.conf_thresh,
        'target_size': cfg.target_size,
        'flip_sides': cfg.flip_sides,
        'enabled_classes': cfg.enabled_classes,
        'motion_thresh': cfg.motion_thresh,
    })


@app.route('/api/counts')
def api_counts():
    cfg = app.config['cfg']
    cap_dir = cfg.capture_dir
    c_in = c_out = 0
    if os.path.isdir(cap_dir):
        for fn in os.listdir(cap_dir):
            if fn.endswith('_in.jpg'):
                c_in += 1
            elif fn.endswith('_out.jpg'):
                c_out += 1
    return jsonify({"in": c_in, "out": c_out})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    cfg = app.config['cfg']
    cfg.reset_counts()
    return jsonify({'status': 'ok'})


@app.route('/api/captures')
def api_captures():
    cfg = app.config['cfg']
    base_url = request.host_url.rstrip('/')
    return jsonify([
        {**c, 'url': f"{base_url}/captures/{c['filename']}",
         'thumb_url': f"{base_url}/captures/{c['thumb']}"}
        for c in cfg.captures[:20]
    ])


@app.route('/api/captures/all')
def api_captures_all():
    cfg = app.config['cfg']
    base_url = request.host_url.rstrip('/')
    cap_dir = cfg.capture_dir
    results = []
    if os.path.isdir(cap_dir):
        for fn in sorted(os.listdir(cap_dir)):
            if not fn.endswith('.jpg') or fn.startswith('thumb'):
                continue
            thumb = f"thumb/{fn}"
            parts = fn.replace('.jpg', '').split('_')
            direction = parts[-1] if len(parts) > 2 else '?'
            timestamp = '_'.join(parts[:2]) if len(parts) > 2 else fn
            results.append({
                'filename': fn,
                'url': f"{base_url}/captures/{fn}",
                'thumb_url': f"{base_url}/captures/{thumb}",
                'direction': direction,
                'timestamp': timestamp,
            })
    return jsonify(results)


@app.route('/api/captures/delete', methods=['POST'])
def api_captures_delete():
    data = request.get_json()
    if not data or 'files' not in data:
        return jsonify({'status': 'error'}), 400
    cfg = app.config['cfg']
    cap_dir = cfg.capture_dir
    deleted = 0
    for fn in data['files']:
        for p in [fn, f"thumb/{fn}"]:
            fp = os.path.join(cap_dir, p)
            if os.path.exists(fp):
                os.remove(fp)
                deleted += 1
    return jsonify({'status': 'ok', 'deleted': deleted})


@app.route('/captures/<path:filename>')
def serve_capture(filename):
    cfg = app.config['cfg']
    return send_from_directory(cfg.capture_dir, filename)


def run_server(config, host='0.0.0.0', port=5000):
    app = create_server(config)
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
