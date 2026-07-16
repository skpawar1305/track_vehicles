#![allow(dead_code)]

mod config;
mod capture;
mod motion;
mod line_counter;
mod annotate;
mod server;
mod types;
mod detector;
mod ncnn_wrapper;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use config::AppState;
use capture::Capture;
use motion::MotionDetector;
use types::Detection;
use opencv::{
    core,
    imgcodecs,
    imgproc,
    prelude::*,
};

const CAPTURE_WIDTH: u32 = 640;
const CAPTURE_HEIGHT: u32 = 360;
const PERIODIC_INTERVAL: u32 = 30;
const FRAME_SKIP: u32 = 1;

fn save_capture(mat: &core::Mat, cap_dir: &str, direction: &str) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = now.as_secs();
    let msec = now.as_millis() % 1000;
    let time_secs = total_secs % 86400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    // Date from days since epoch using civil calendar
    let days = (total_secs / 86400) as i64;
    let mut y = 1970i64;
    let mut rem = days;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < days_in_year { break; }
        rem -= days_in_year;
        y += 1;
    }
    let month_days = [31, if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 1u64;
    let mut d = rem as u64;
    for &md in &month_days {
        if d < md { break; }
        d -= md;
        mo += 1;
    }
    let ts = format!("{:04}{:02}{:02}_{:02}{:02}{:02}_{:03}", y, mo, d + 1, h, m, s, msec);
    let filename = format!("{}_{}.jpg", ts, direction);

    std::fs::create_dir_all(cap_dir).ok();
    let mut buf = core::Vector::<u8>::new();
    let params = core::Vector::<i32>::new();
    if imgcodecs::imencode(".jpg", mat, &mut buf, &params).ok().unwrap_or(false) {
        std::fs::write(&format!("{}/{}", cap_dir, filename), buf.to_vec()).ok();
    }

    // Thumbnail
    std::fs::create_dir_all(&format!("{}/thumb", cap_dir)).ok();
    let (w, h) = (mat.cols(), mat.rows());
    let tw = 160i32;
    let th = (h as f64 * tw as f64 / w as f64) as i32;
    if th > 0 {
        let mut small = core::Mat::default();
        imgproc::resize(mat, &mut small, core::Size::new(tw, th), 0.0, 0.0, imgproc::INTER_LINEAR).ok();
        let mut sbuf = core::Vector::<u8>::new();
        if imgcodecs::imencode(".jpg", &small, &mut sbuf, &params).ok().unwrap_or(false) {
            std::fs::write(&format!("{}/thumb/{}", cap_dir, filename), sbuf.to_vec()).ok();
        }
    }
}

fn main() -> Result<(), String> {
    let state = Arc::new(AppState::new("config.json"));

    // Web server in a separate thread
    let server_state = state.clone();
    std::thread::spawn(move || {
        server::run_server(server_state);
    });

    std::thread::sleep(std::time::Duration::from_millis(500));

    // Processing loop in this thread
    processing_loop(state)
}

fn processing_loop(state: Arc<AppState>) -> Result<(), String> {
    let mut detector = detector::Detector::new(
        &state.config.read().unwrap().model_path,
        state.config.read().unwrap().target_size,
    ).ok();

    let mut cap: Option<Capture> = Capture::open(
        &state.config.read().unwrap().stream_url, CAPTURE_WIDTH, CAPTURE_HEIGHT
    ).ok();
    let mut motion = MotionDetector::new(500).map_err(|e| e)?;
    let mut tracker = jamtrack_rs::ByteTracker::new(15, 30, 0.5, 0.5, 0.8);

    {
        let cfg = state.config.read().unwrap();
        if let Some(line) = cfg.line {
            motion.update_line(line);
        }
    }

    let mut recent_crossings: Vec<(f64, f64, line_counter::Crossing)> = Vec::new();
    let mut prev_centroids: std::collections::HashMap<usize, (i32, i32)> = std::collections::HashMap::new();
    let mut last_cross_frames: std::collections::HashMap<usize, i64> = std::collections::HashMap::new();
    let mut frame_count: u32 = 0;
    let mut fps_timer = Instant::now();
    let mut current_fps = 0.0;

    loop {
        if !state.config.read().is_ok() {
            break Ok(());
        }

        let mut mat = if let Some(ref mut c) = cap {
            match c.read() {
                Some(m) => m,
                None => {
                    eprintln!("Stream disconnected, reconnecting...");
                    c.close();
                    cap = None;
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            }
        } else {
            cap = Capture::open(
                &state.config.read().unwrap().stream_url, CAPTURE_WIDTH, CAPTURE_HEIGHT
            ).ok();
            std::thread::sleep(std::time::Duration::from_secs(5));
            continue;
        };

        frame_count += 1;
        if fps_timer.elapsed().as_secs_f64() >= 1.0 {
            current_fps = frame_count as f64 / fps_timer.elapsed().as_secs_f64();
            frame_count = 0;
            fps_timer = Instant::now();
        }

        let cfg = state.config.read().unwrap().clone();

        if let Some(line) = cfg.line {
            if motion.line != Some(line) {
                motion.update_line(line);
            }
            motion.detect(&mat);
        }

        let detections: Vec<Detection> = if let Some(ref mut det) = detector {
            det.detect(&mat, &cfg.enabled_classes)
        } else {
            vec![]
        };

        // Convert detections to jamtrack-rs Objects
        let jm_objects: Vec<jamtrack_rs::Object> = detections.iter().map(|d| {
            jamtrack_rs::Object::new(
                jamtrack_rs::Rect::from_xyxy(
                    d.bbox[0] as f32, d.bbox[1] as f32,
                    d.bbox[2] as f32, d.bbox[3] as f32,
                ),
                d.confidence,
                None,
            )
        }).collect();

        let tracked = tracker.update(&jm_objects).unwrap_or_default();

        // Build a map of track_id → centroid for active tracks
        let mut active_tracks: Vec<(usize, (i32, i32))> = Vec::new();
        for obj in &tracked {
            let r = obj.get_rect();
            let cx = ((r.x() + r.x() + r.width()) / 2.0) as i32;
            let cy = ((r.y() + r.y() + r.height()) / 2.0) as i32;
            if let Some(tid) = obj.get_track_id() {
                active_tracks.push((tid, (cx, cy)));
            }
        }

        let mut did_cross = false;
        let mut cross_dir = String::new();

        if let Some(line) = cfg.line {
            // Collect: track_id → crossing_type for counted tracks
            let mut counted_tids: Vec<(usize, line_counter::Crossing)> = Vec::new();
            for (tid, centroid) in &active_tracks {
                let last_cross = last_cross_frames.get(tid).copied().unwrap_or(-60);
                if frame_count as i64 - last_cross < 15 {
                    continue;
                }
                if let Some(prev) = prev_centroids.get(tid) {
                    let cross = line_counter::detect_crossing(&line, *prev, *centroid, cfg.flip_sides);
                    if cross != line_counter::Crossing::None {
                        let cx = (prev.0 + centroid.0) as f64 / 2.0;
                        let cy = (prev.1 + centroid.1) as f64 / 2.0;
                        let is_dup = recent_crossings.iter().any(|(px, py, _)| {
                            let d = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
                            d < 50.0
                        });
                        if !is_dup {
                            recent_crossings.push((cx, cy, cross));
                            counted_tids.push((*tid, cross));
                        }
                    }
                }
            }
            while recent_crossings.len() > 30 {
                recent_crossings.remove(0);
            }
            for (tid, cross) in &counted_tids {
                last_cross_frames.insert(*tid, frame_count as i64);
                let d = match cross {
                    line_counter::Crossing::In => {
                        state.count_in.fetch_add(1, Ordering::Relaxed);
                        "in"
                    }
                    line_counter::Crossing::Out => {
                        state.count_out.fetch_add(1, Ordering::Relaxed);
                        "out"
                    }
                    _ => continue,
                };
                if !did_cross {
                    did_cross = true;
                    cross_dir = d.to_string();
                }
            }
        }

        // Store centroids for next frame
        for (tid, centroid) in &active_tracks {
            prev_centroids.insert(*tid, *centroid);
        }

        let c_in = state.count_in.load(Ordering::Relaxed);
        let c_out = state.count_out.load(Ordering::Relaxed);

        if let Some(line) = cfg.line {
            annotate::draw_line(&mut mat, &line, cfg.flip_sides);
        }

        // Draw boxes only for tracks near the line
        struct DrawBox { id: u32, bbox: [i32; 4], label: String }
        let mut draw_list: Vec<DrawBox> = Vec::new();
        for obj in &tracked {
            if let Some(tid) = obj.get_track_id() {
                let r = obj.get_rect();
                let x1 = r.x() as i32;
                let y1 = r.y() as i32;
                let x2 = (r.x() + r.width()) as i32;
                let y2 = (r.y() + r.height()) as i32;
                let cx = (x1 + x2) / 2;
                let cy = (y1 + y2) / 2;
                let near_line = cfg.line.map_or(true, |line| {
                    line_counter::point_line_distance(&line, (cx, cy)) < 60.0
                });
                if near_line {
                    draw_list.push(DrawBox { id: tid as u32, bbox: [x1, y1, x2, y2], label: format!("id:{}", tid) });
                }
            }
        }
        let draw_refs: Vec<(u32, &[i32; 4], &str)> = draw_list.iter()
            .map(|d| (d.id, &d.bbox, d.label.as_str()))
            .collect();
        annotate::draw_boxes(&mut mat, &draw_refs);
        annotate::draw_counts(&mut mat, c_in, c_out, current_fps as f32);

        if did_cross {
            save_capture(&mat, &cfg.capture_dir, &cross_dir);
        }

        if frame_count % FRAME_SKIP == 0 {
            let mut jpeg_buf = core::Vector::<u8>::new();
            let params = core::Vector::<i32>::new();
            if imgcodecs::imencode(".jpg", &mat, &mut jpeg_buf, &params).ok().unwrap_or(false) {
                *state.frame_buffer.write().unwrap() = Some(jpeg_buf.to_vec());
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
