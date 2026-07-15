use actix_web::{web, App, HttpServer, HttpResponse, get, post};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::fs;
use crate::config::AppState;
use serde_json::{json, Value};

const INDEX_HTML: &str = include_str!("../../templates/index.html");
const CAPTURES_HTML: &str = include_str!("../../templates/captures.html");
const ANALYTICS_HTML: &str = include_str!("../../templates/analytics.html");

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok().content_type("text/html").body(INDEX_HTML)
}

#[get("/captures")]
async fn captures_page() -> HttpResponse {
    HttpResponse::Ok().content_type("text/html").body(CAPTURES_HTML)
}

#[get("/analytics")]
async fn analytics_page() -> HttpResponse {
    HttpResponse::Ok().content_type("text/html").body(ANALYTICS_HTML)
}

#[get("/video_feed")]
async fn video_feed(state: web::Data<Arc<AppState>>) -> HttpResponse {
    *state.viewers.write().unwrap() += 1;
    let s = state.get_ref().clone();

    let stream = async_stream::stream! {
        loop {
            let frame = s.frame_buffer.read().unwrap().clone();
            if let Some(jpeg) = frame {
                let mut buf = Vec::with_capacity(jpeg.len() + 100);
                buf.extend_from_slice(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n");
                buf.extend_from_slice(&jpeg);
                buf.extend_from_slice(b"\r\n");
                yield Ok::<_, actix_web::Error>(web::Bytes::from(buf));
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    };

    HttpResponse::Ok()
        .content_type("multipart/x-mixed-replace; boundary=frame")
        .streaming(stream)
}

#[get("/api/line")]
async fn api_line_get(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let line = state.config.read().unwrap().line;
    HttpResponse::Ok().json(json!({"line": line}))
}

#[post("/api/line")]
async fn api_line_post(state: web::Data<Arc<AppState>>, body: web::Json<Value>) -> HttpResponse {
    if let Some(line) = body.get("line") {
        if line.is_null() {
            state.config.write().unwrap().line = None;
            state.config.write().unwrap().save(&state.config_path);
            return HttpResponse::Ok().json(json!({"status": "ok", "line": null}));
        }
        if let Some(arr) = line.as_array() {
            if arr.len() == 4 {
                let l: [i32; 4] = [
                    arr[0].as_i64().unwrap_or(0) as i32,
                    arr[1].as_i64().unwrap_or(0) as i32,
                    arr[2].as_i64().unwrap_or(0) as i32,
                    arr[3].as_i64().unwrap_or(0) as i32,
                ];
                state.config.write().unwrap().line = Some(l);
state.config.write().unwrap().save(&state.config_path);
                return HttpResponse::Ok().json(json!({"status": "ok", "line": l}));
            }
        }
    }
    HttpResponse::BadRequest().json(json!({"status": "error"}))
}

#[get("/api/config")]
async fn api_config_get(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let cfg = state.config.read().unwrap();
    HttpResponse::Ok().json(json!({
        "stream_url": cfg.stream_url,
        "line": cfg.line,
        "conf_thresh": cfg.conf_thresh,
        "target_size": cfg.target_size,
        "flip_sides": cfg.flip_sides,
        "enabled_classes": cfg.enabled_classes,
        "motion_thresh": cfg.motion_thresh,
    }))
}

#[post("/api/config")]
async fn api_config_post(state: web::Data<Arc<AppState>>, body: web::Json<Value>) -> HttpResponse {
    if let Some(url) = body.get("stream_url").and_then(|v| v.as_str()) {
        state.config.write().unwrap().stream_url = url.to_string();
        state.config.write().unwrap().save(&state.config_path);
        return HttpResponse::Ok().json(json!({"status": "ok", "stream_url": url}));
    }
    if let Some(flip) = body.get("flip_sides").and_then(|v| v.as_bool()) {
        state.config.write().unwrap().flip_sides = flip;
        state.config.write().unwrap().save(&state.config_path);
        return HttpResponse::Ok().json(json!({"status": "ok", "flip_sides": flip}));
    }
    if let Some(classes) = body.get("enabled_classes").and_then(|v| v.as_array()) {
        let v: Vec<i32> = classes.iter().filter_map(|c| c.as_i64().map(|i| i as i32)).collect();
        state.config.write().unwrap().enabled_classes = v.clone();
        state.config.write().unwrap().save(&state.config_path);
        return HttpResponse::Ok().json(json!({"status": "ok", "enabled_classes": v}));
    }
    HttpResponse::BadRequest().json(json!({"status": "error"}))
}

#[get("/api/counts")]
async fn api_counts(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let c_in = state.count_in.load(Ordering::Relaxed);
    let c_out = state.count_out.load(Ordering::Relaxed);
    HttpResponse::Ok().json(json!({"in": c_in, "out": c_out}))
}

#[get("/api/captures")]
async fn api_captures(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let cfg = state.config.read().unwrap();
    let cap_dir = cfg.capture_dir.clone();
    let base_url = "http://localhost:5000";
    let mut results = Vec::new();
    if let Ok(entries) = fs::read_dir(&cap_dir) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        files.sort_by_key(|f| f.file_name());
        files.reverse();
        for entry in files.iter().take(20) {
            let fn_ = entry.file_name().to_string_lossy().to_string();
            if !fn_.ends_with(".jpg") || fn_.starts_with("thumb") { continue; }
            let thumb = format!("thumb/{}", fn_);
            results.push(json!({
                "filename": fn_,
                "url": format!("{}/captures/{}", base_url, fn_),
                "thumb_url": format!("{}/captures/{}", base_url, thumb),
                "direction": if fn_.contains("_in") { "in" } else { "out" },
                "timestamp": fn_.replace(".jpg", ""),
            }));
        }
    }
    HttpResponse::Ok().json(results)
}

#[get("/api/captures/all")]
async fn api_captures_all(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let cfg = state.config.read().unwrap();
    let cap_dir = cfg.capture_dir.clone();
    let base_url = "http://localhost:5000";
    let mut results = Vec::new();
    if let Ok(entries) = fs::read_dir(&cap_dir) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        files.sort_by_key(|f| f.file_name());
        for entry in &files {
            let fn_ = entry.file_name().to_string_lossy().to_string();
            if !fn_.ends_with(".jpg") || fn_.starts_with("thumb") { continue; }
            let thumb = format!("thumb/{}", fn_);
            let stem = fn_.replace(".jpg", "");
            let parts: Vec<&str> = stem.split('_').collect();
            let direction = if parts.len() > 2 { parts[parts.len()-1].to_string() } else { "?".to_string() };
            let timestamp = if parts.len() > 2 { parts[..2].join("_") } else { stem };
            results.push(json!({
                "filename": fn_,
                "url": format!("{}/captures/{}", base_url, fn_),
                "thumb_url": format!("{}/captures/{}", base_url, thumb),
                "direction": direction,
                "timestamp": timestamp,
            }));
        }
    }
    HttpResponse::Ok().json(results)
}

#[post("/api/captures/delete")]
async fn api_captures_delete(state: web::Data<Arc<AppState>>, body: web::Json<Value>) -> HttpResponse {
    let cap_dir = state.config.read().unwrap().capture_dir.clone();
    let files = match body.get("files").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return HttpResponse::BadRequest().json(json!({"status": "error"})),
    };
    let mut deleted = 0;
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
    HttpResponse::Ok().json(json!({"status": "ok", "deleted": deleted}))
}

#[post("/api/reset")]
async fn api_reset(state: web::Data<Arc<AppState>>) -> HttpResponse {
    state.count_in.store(0, Ordering::Relaxed);
    state.count_out.store(0, Ordering::Relaxed);
    crate::config::persist_counts(state.get_ref());
    HttpResponse::Ok().json(json!({"status": "ok"}))
}

#[get("/captures/{filename:.*}")]
async fn serve_capture(state: web::Data<Arc<AppState>>, path: web::Path<String>) -> HttpResponse {
    let cap_dir = state.config.read().unwrap().capture_dir.clone();
    let filepath = format!("{}/{}", cap_dir, path.into_inner());
    match fs::read(&filepath) {
        Ok(data) => {
            let mime = if filepath.ends_with(".jpg") || filepath.ends_with(".jpeg") {
                "image/jpeg"
            } else { "application/octet-stream" };
            HttpResponse::Ok().content_type(mime).body(data)
        }
        Err(_) => HttpResponse::NotFound().body("Not found"),
    }
}

pub async fn run_server(state: Arc<AppState>) -> std::io::Result<()> {
    let state_data = state.clone();
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state_data.clone()))
            .service(index)
            .service(captures_page)
            .service(analytics_page)
            .service(video_feed)
            .service(api_line_get)
            .service(api_line_post)
            .service(api_config_get)
            .service(api_config_post)
            .service(api_counts)
            .service(api_captures)
            .service(api_captures_all)
            .service(api_captures_delete)
            .service(api_reset)
            .service(serve_capture)
    })
    .bind("0.0.0.0:5000")?
    .run()
    .await
}
