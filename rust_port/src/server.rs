use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::fs;
use std::io::{Read, Write};
use crate::config::AppState;

const INDEX_HTML: &str = include_str!("../../templates/index.html");
const CAPTURES_HTML: &str = include_str!("../../templates/captures.html");
const ANALYTICS_HTML: &str = include_str!("../../templates/analytics.html");

fn resp_ok(body: &[u8], content_type: &str) -> Vec<u8> {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        content_type, body.len()
    );
    let mut buf = header.into_bytes();
    buf.extend_from_slice(body);
    buf
}

fn resp_html(body: &str) -> Vec<u8> {
    resp_ok(body.as_bytes(), "text/html; charset=utf-8")
}

fn resp_json(body: &str) -> Vec<u8> {
    resp_ok(body.as_bytes(), "application/json")
}

fn resp_404() -> Vec<u8> {
    let body = b"Not Found";
    let header = format!(
        "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut buf = header.into_bytes();
    buf.extend_from_slice(body);
    buf
}

fn extract_body(req: &str) -> &str {
    // Find double \r\n\r\n separating headers from body
    if let Some(idx) = req.find("\r\n\r\n") {
        &req[idx + 4..]
    } else {
        ""
    }
}

fn handle_request(req: &str, state: &Arc<AppState>) -> Vec<u8> {
    let lines: Vec<&str> = req.lines().collect();
    if lines.is_empty() {
        return resp_404();
    }
    let first = lines[0];
    let parts: Vec<&str> = first.split_whitespace().collect();
    if parts.len() < 2 {
        return resp_404();
    }
    let method = parts[0];
    let path = parts[1];
    let body = extract_body(req);

    match (method, path) {
        ("GET", "/") => resp_html(INDEX_HTML),
        ("GET", "/captures") | ("GET", "/captures/") => resp_html(CAPTURES_HTML),
        ("GET", "/analytics") => resp_html(ANALYTICS_HTML),

        ("GET", "/api/counts") => {
            let c_in = state.count_in.load(Ordering::Relaxed);
            let c_out = state.count_out.load(Ordering::Relaxed);
            resp_json(&format!(r#"{{"in":{},"out":{}}}"#, c_in, c_out))
        }

        ("GET", "/api/config") => {
            let cfg = state.config.read().unwrap();
            let classes = serde_json::to_string(&cfg.enabled_classes).unwrap_or_default();
            let line_str = match cfg.line {
                Some(l) => format!("[{},{},{},{}]", l[0], l[1], l[2], l[3]),
                None => "null".to_string(),
            };
            resp_json(&format!(
                r#"{{"stream_url":"{}","line":{},"conf_thresh":{},"target_size":{},"flip_sides":{},"enabled_classes":{},"motion_thresh":{}}}"#,
                cfg.stream_url, line_str, cfg.conf_thresh, cfg.target_size,
                if cfg.flip_sides { "true" } else { "false" },
                classes, cfg.motion_thresh
            ))
        }

        ("GET", "/api/line") => {
            let line = state.config.read().unwrap().line;
            let json = match line {
                Some(l) => format!(r#"{{"line":[{},{},{},{}]}}"#, l[0], l[1], l[2], l[3]),
                None => r#"{"line":null}"#.to_string(),
            };
            resp_json(&json)
        }

        ("POST", "/api/line") => {
            let body_str = body.to_string();
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_str) {
                if let Some(line_val) = json.get("line") {
                    if line_val.is_null() {
                        state.config.write().unwrap().line = None;
                        state.config.write().unwrap().save(&state.config_path);
                        return resp_json(r#"{"status":"ok","line":null}"#);
                    }
                    if let Some(arr) = line_val.as_array() {
                        if arr.len() == 4 {
                            let l: [i32; 4] = [
                                arr[0].as_i64().unwrap_or(0) as i32,
                                arr[1].as_i64().unwrap_or(0) as i32,
                                arr[2].as_i64().unwrap_or(0) as i32,
                                arr[3].as_i64().unwrap_or(0) as i32,
                            ];
                            state.config.write().unwrap().line = Some(l);
                            state.config.write().unwrap().save(&state.config_path);
                            return resp_json(&format!(r#"{{"status":"ok","line":[{},{},{},{}]}}"#, l[0], l[1], l[2], l[3]));
                        }
                    }
                }
            }
            resp_json(r#"{"status":"error"}"#)
        }

        ("GET", path) if path.starts_with("/captures/") => {
            let cfg = state.config.read().unwrap();
            let filepath = format!("{}/{}", cfg.capture_dir, &path[10..]);
            match fs::read(&filepath) {
                Ok(data) => {
                    let mime = if filepath.ends_with(".jpg") { "image/jpeg" } else { "application/octet-stream" };
                    resp_ok(&data, mime)
                }
                Err(_) => resp_404(),
            }
        }

        ("GET", "/api/captures") | ("GET", "/api/captures/all") => {
            let cfg = state.config.read().unwrap();
            let cap_dir = cfg.capture_dir.clone();
            let all = path.ends_with("/all");
            let mut results = Vec::new();
            if let Ok(entries) = fs::read_dir(&cap_dir) {
                let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
                files.sort_by_key(|f| f.file_name());
                if !all { files.reverse(); }
                for entry in files.iter().take(if all { files.len() } else { 20 }) {
                    let fn_ = entry.file_name().to_string_lossy().to_string();
                    if !fn_.ends_with(".jpg") || fn_.starts_with("thumb") { continue; }
                    let thumb = format!("thumb/{}", fn_);
                    let direction = if fn_.contains("_in") { "in" } else { "out" };
                    // Timestamp is the filename without extension and without direction suffix
                    let stem = fn_.replace(".jpg", "");
                    let ts = if stem.ends_with("_in") || stem.ends_with("_out") {
                        stem[..stem.len() - 3].to_string()
                    } else {
                        stem
                    };
                    results.push(format!(
                        r#"{{"filename":"{}","url":"/captures/{}","thumb_url":"/captures/{}","direction":"{}","timestamp":"{}"}}"#,
                        fn_, fn_, thumb, direction, ts
                    ));
                }
            }
            resp_json(&format!("[{}]", results.join(",")))
        }

        ("POST", "/api/captures/delete") => {
            let body_str = body.to_string();
            let cap_dir = state.config.read().unwrap().capture_dir.clone();
            let mut deleted = 0;
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_str) {
                if let Some(files) = json.get("files").and_then(|v| v.as_array()) {
                    for f in files {
                        if let Some(name) = f.as_str() {
                            for p in &[name.to_string(), format!("thumb/{}", name)] {
                                let fp = format!("{}/{}", cap_dir, p);
                                if fs::remove_file(&fp).is_ok() {
                                    deleted += 1;
                                }
                            }
                        }
                    }
                }
            }
            resp_json(&format!(r#"{{"status":"ok","deleted":{}}}"#, deleted))
        }

        ("POST", "/api/reset") => {
            state.count_in.store(0, Ordering::Relaxed);
            state.count_out.store(0, Ordering::Relaxed);
            resp_json(r#"{"status":"ok"}"#)
        }

        ("POST", "/api/config") => {
            let body_str = body.to_string();
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_str) {
                if let Some(url) = json.get("stream_url").and_then(|v| v.as_str()) {
                    state.config.write().unwrap().stream_url = url.to_string();
                    state.config.write().unwrap().save(&state.config_path);
                    return resp_json(&format!(r#"{{"status":"ok","stream_url":"{}"}}"#, url));
                }
                if let Some(flip) = json.get("flip_sides").and_then(|v| v.as_bool()) {
                    state.config.write().unwrap().flip_sides = flip;
                    state.config.write().unwrap().save(&state.config_path);
                    return resp_json(&format!(r#"{{"status":"ok","flip_sides":{}}}"#, flip));
                }
                if let Some(classes) = json.get("enabled_classes").and_then(|v| v.as_array()) {
                    let v: Vec<i32> = classes.iter().filter_map(|c| c.as_i64().map(|i| i as i32)).collect();
                    state.config.write().unwrap().enabled_classes = v.clone();
                    state.config.write().unwrap().save(&state.config_path);
                    let json = serde_json::to_string(&v).unwrap_or_default();
                    return resp_json(&format!(r#"{{"status":"ok","enabled_classes":{}}}"#, json));
                }
            }
            resp_json(r#"{"status":"error"}"#)
        }

        _ => resp_404(),
    }
}

fn serve_video_feed(mut stream: std::net::TcpStream, state: Arc<AppState>) {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if stream.write_all(header.as_bytes()).is_err() { return; }
    stream.flush().ok();

    let boundary = b"\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n";
    for _ in 0..600 {
        let frame = state.frame_buffer.read().unwrap().clone();
        if let Some(jpeg) = frame {
            if stream.write_all(boundary).is_err() { break; }
            if stream.write_all(&jpeg).is_err() { break; }
            stream.write_all(b"\r\n").ok();
            stream.flush().ok();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

pub fn run_server(state: Arc<AppState>) {
    let listener = match std::net::TcpListener::bind("0.0.0.0:5000") {
        Ok(l) => l,
        Err(e) => {
            eprintln!("SERVER: failed to bind: {}", e);
            return;
        }
    };
    listener.set_nonblocking(true).ok();
    eprintln!("SERVER: listening on :5000");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                stream.set_nonblocking(false).ok();
                let state = state.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 8192];
                    match stream.read(&mut buf) {
                        Ok(n) if n > 0 => {
                            let req = String::from_utf8_lossy(&buf[..n]);
                            let first = req.lines().next().unwrap_or("");
                            let parts: Vec<&str> = first.split_whitespace().collect();
                            let path = parts.get(1).unwrap_or(&"");

                            if *path == "/video_feed" {
                                serve_video_feed(stream, state);
                            } else {
                                let resp = handle_request(&req, &state);
                                let _ = stream.write_all(&resp);
                            }
                        }
                        _ => {}
                    }
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => {
                eprintln!("SERVER: accept error: {}", e);
            }
        }
    }
}
