#![allow(dead_code)]

mod config;
mod capture;
mod motion;
mod line_counter;
mod annotate;
mod server;
mod detector_ort;
mod python_tracker;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use config::AppState;
use capture::Capture;
use motion::MotionDetector;
use detector_ort::{Detection, NanoDetONNX};
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
    let model_path = {
        let cfg = state.config.read().unwrap();
        let base = cfg.model_path.trim_end_matches('/');
        if base.ends_with(".onnx") {
            base.to_string()
        } else {
            format!("../models/nanodet-plus-m-1.5x_416.onnx")
        }
    };
    let conf_thresh = state.config.read().unwrap().conf_thresh;
    let mut detector = NanoDetONNX::new(&model_path, conf_thresh, 0.45).ok();

    let mut cap: Option<Capture> = Capture::open(
        &state.config.read().unwrap().stream_url, CAPTURE_WIDTH, CAPTURE_HEIGHT
    ).ok();
    let mut motion = MotionDetector::new(500).map_err(|e| e)?;
    let tracker = python_tracker::PythonTracker::new(conf_thresh, 50, 0.7, 30).ok();
    let use_python_tracker = tracker.is_some();

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
    let mut total_frames: u64 = 0;
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
        total_frames += 1;
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
            det.detect(&mat, &cfg.enabled_classes).unwrap_or_default()
        } else {
            vec![]
        };

        // Track using Python bytetracker (identical to run.py)
        let active_tracks: Vec<(usize, (i32, i32), [i32; 4])> = if let Some(ref tr) = tracker {
            match tr.update(&detections) {
                Ok(results) => results.into_iter().map(|r| {
                    (r.tid, (r.centroid[0], r.centroid[1]), r.bbox)
                }).collect(),
                Err(e) => { eprintln!("Tracker error: {}", e); vec![] }
            }
        } else {
            vec![]
        };
        if total_frames % 30 == 0 && !active_tracks.is_empty() {
            let unique_tids: std::collections::HashSet<usize> = active_tracks.iter().map(|(tid,_,_)| *tid).collect();
            let near_line = active_tracks.iter().filter(|(_, _, bbox)| 
                cfg.line.map_or(false, |l| line_counter::bbox_touches_line(&l, bbox))
            ).count();
            eprintln!("D: frame={} det={} tracks={} uniq={} near_line={}",
                frame_count, detections.len(), active_tracks.len(), unique_tids.len(), near_line);
        }

        let mut did_cross = false;
        let mut cross_dir = String::new();

        if let Some(line) = cfg.line {
            // Collect: track_id → crossing_type for counted tracks
            let mut counted_tids: Vec<(usize, line_counter::Crossing)> = Vec::new();
            for (tid, centroid, bbox) in &active_tracks {
                let last_cross = last_cross_frames.get(tid).copied().unwrap_or(-60i64);
                if total_frames as i64 - last_cross < 15 {
                    continue;
                }
                if !line_counter::bbox_touches_line(&line, bbox) {
                    continue;
                }
                if let Some(prev) = prev_centroids.get(tid) {
                    if total_frames % 30 == 0 && total_frames > 0 {
                        let old_s = line_counter::which_side(&line, *prev);
                        let new_s = line_counter::which_side(&line, *centroid);
                        eprintln!("D3: track={} cent=({},{}) prev=({},{}) sides={}->{} bbox=[{},{},{},{}]",
                            tid, centroid.0, centroid.1, prev.0, prev.1, old_s, new_s,
                            bbox[0], bbox[1], bbox[2], bbox[3]);
                    }
                    let cross = line_counter::detect_crossing(&line, *prev, *centroid, cfg.flip_sides);
                    if cross != line_counter::Crossing::None {
                        eprintln!("D4: CROSSING! track={} cent=({},{}) prev=({},{}) sides={}->{} cross={:?}",
                            tid, centroid.0, centroid.1, prev.0, prev.1,
                            line_counter::which_side(&line, *prev), line_counter::which_side(&line, *centroid),
                            cross);
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
                last_cross_frames.insert(*tid, total_frames as i64);
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
        if frame_count % 30 == 0 {
            eprintln!("D: frame={} det={}", frame_count, detections.len());
        }

        // Store centroids for next frame
        for (tid, centroid, _) in &active_tracks {
            prev_centroids.insert(*tid, *centroid);
        }

        let c_in = state.count_in.load(Ordering::Relaxed);
        let c_out = state.count_out.load(Ordering::Relaxed);

        if did_cross {
            // Full annotation on a clone for capture saving
            if let Ok(mut cap_mat) = mat.try_clone() {
                if let Some(line) = cfg.line {
                    annotate::draw_line(&mut cap_mat, &line, cfg.flip_sides);
                }
                // Draw boxes for tracks near the line
                struct DrawBox { id: u32, bbox: [i32; 4], label: String }
                let mut draw_list: Vec<DrawBox> = Vec::new();
                for (tid, _, bbox) in &active_tracks {
                    let x1 = bbox[0]; let y1 = bbox[1];
                    let x2 = bbox[2]; let y2 = bbox[3];
                    let cx = (x1 + x2) / 2;
                    let cy = (y1 + y2) / 2;
                    let near_line = cfg.line.map_or(true, |line| {
                        line_counter::point_line_distance(&line, (cx, cy)) < 60.0
                    });
                    if near_line {
                        draw_list.push(DrawBox { id: *tid as u32, bbox: *bbox, label: format!("id:{}", tid) });
                    }
                }
                let draw_refs: Vec<(u32, &[i32; 4], &str)> = draw_list.iter()
                    .map(|d| (d.id, &d.bbox, d.label.as_str()))
                    .collect();
                annotate::draw_boxes(&mut cap_mat, &draw_refs);
                annotate::draw_counts(&mut cap_mat, c_in, c_out, current_fps as f32, false);
                save_capture(&cap_mat, &cfg.capture_dir, &cross_dir);
            }
        }

        // Simple annotation (IN/OUT only) for the stream
        annotate::draw_counts(&mut mat, c_in, c_out, current_fps as f32, true);

        if frame_count % FRAME_SKIP == 0 {
            let mut jpeg_buf = core::Vector::<u8>::new();
            let params = core::Vector::<i32>::new();
            if imgcodecs::imencode(".jpg", &mat, &mut jpeg_buf, &params).ok().unwrap_or(false) {
                *state.frame_buffer.write().unwrap() = Some(jpeg_buf.to_vec());
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(33));
    }
}
