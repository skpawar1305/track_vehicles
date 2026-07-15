#![allow(dead_code)]

mod config;
mod capture;
mod motion;
mod line_counter;
mod tracker;
mod annotate;
mod server;
mod types;
mod detector;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use config::AppState;
use capture::Capture;
use motion::MotionDetector;
use tracker::ByteTrack;
use types::Detection;
use opencv::{
    core,
    imgcodecs,
    prelude::*,
};

const CAPTURE_WIDTH: u32 = 640;
const CAPTURE_HEIGHT: u32 = 360;
const PERIODIC_INTERVAL: u32 = 30;
const FRAME_SKIP: u32 = 1;

fn main() -> Result<(), String> {
    let state = Arc::new(AppState::new("config.json"));

    let server_state = state.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let _ = server::run_server(server_state).await;
        });
    });

    std::thread::sleep(std::time::Duration::from_secs(1));

    let mut detector = detector::Detector::new(
        &state.config.read().unwrap().model_path,
        state.config.read().unwrap().target_size,
    ).ok();

    let mut cap: Option<Capture> = Capture::open(
        &state.config.read().unwrap().stream_url, CAPTURE_WIDTH, CAPTURE_HEIGHT
    ).ok();
    let mut motion = MotionDetector::new(500).map_err(|e| e)?;
    let mut tracker = ByteTrack::new(0.5);

    {
        let cfg = state.config.read().unwrap();
        if let Some(line) = cfg.line {
            motion.update_line(line);
        }
    }

    // Track which trackers have already triggered a crossing
    let mut crossed: std::collections::HashSet<u32> = std::collections::HashSet::new();

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
            if motion.motion_state || frame_count % PERIODIC_INTERVAL == 0 {
                det.detect(&mat, &cfg.enabled_classes)
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        let _track_ids = tracker.update(detections);

        // Line crossing detection
        if let Some(line) = cfg.line {
            let mut to_cross: Vec<(u32, line_counter::Crossing)> = Vec::new();
            for (&tid, obj) in &tracker.objects {
                if crossed.contains(&tid) {
                    continue;
                }
                if obj.centroids.len() >= 2 {
                    let n = obj.centroids.len();
                    let old = obj.centroids[n - 2];
                    let new = obj.centroids[n - 1];
                    let cross = line_counter::detect_crossing(&line, old, new, cfg.flip_sides);
                    if cross != line_counter::Crossing::None {
                        to_cross.push((tid, cross));
                    }
                }
            }
            for (tid, cross) in &to_cross {
                crossed.insert(*tid);
                match cross {
                    line_counter::Crossing::In => {
                        state.count_in.fetch_add(1, Ordering::Relaxed);
                    }
                    line_counter::Crossing::Out => {
                        state.count_out.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        }

        // Clean crossed set of dead tracks
        crossed.retain(|tid| tracker.objects.contains_key(tid));

        let c_in = state.count_in.load(Ordering::Relaxed);
        let c_out = state.count_out.load(Ordering::Relaxed);

        if let Some(line) = cfg.line {
            annotate::draw_line(&mut mat, &line, cfg.flip_sides);
        }

        let objects: Vec<(u32, &[i32; 4], &str)> = tracker.objects.iter()
            .filter(|(_, o)| o.active)
            .map(|(id, o)| (*id, o.bbox(), o.label.as_str()))
            .collect();
        annotate::draw_boxes(&mut mat, &objects);
        annotate::draw_counts(&mut mat, c_in, c_out, current_fps as f32);

        if frame_count % FRAME_SKIP == 0 {
            let mut jpeg_buf = core::Vector::<u8>::new();
            let params = core::Vector::<i32>::new();
            if imgcodecs::imencode(".jpg", &mat, &mut jpeg_buf, &params).ok().unwrap_or(false) {
                *state.frame_buffer.write().unwrap() = Some(jpeg_buf.to_vec());
            }
        }
    }
}
