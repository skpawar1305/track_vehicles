use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub stream_url: String,
    pub line: Option<[i32; 4]>,
    pub counts: Counts,
    pub conf_thresh: f32,
    pub flip_sides: bool,
    pub motion_thresh: i32,
    pub target_size: u32,
    pub capture_dir: String,
    pub max_captures: usize,
    pub model_path: String,
    pub enabled_classes: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counts {
    #[serde(rename = "in")]
    pub in_count: usize,
    pub out: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            stream_url: "rtsp://admin:cctv%401234@192.168.0.229:554/unicast/c12/s1/live".into(),
            line: None,
            counts: Counts { in_count: 0, out: 0 },
            conf_thresh: 0.5,
            flip_sides: false,
            motion_thresh: 500,
            target_size: 320,
            capture_dir: "captures".into(),
            max_captures: 1000,
            model_path: "models/yolo26n_ncnn_model".into(),
            enabled_classes: vec![2, 3, 5, 7],
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Self {
        let p = Path::new(path);
        if p.exists() {
            let data = fs::read_to_string(path).unwrap_or_default();
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            let cfg = Config::default();
            cfg.save(path);
            cfg
        }
    }

    pub fn save(&self, path: &str) {
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, data);
        }
    }
}

pub struct AppState {
    pub config: RwLock<Config>,
    pub config_path: String,
    pub frame_buffer: RwLock<Option<Vec<u8>>>,
    pub viewers: RwLock<u32>,
    pub count_in: AtomicUsize,
    pub count_out: AtomicUsize,
}

impl AppState {
    pub fn new(config_path: &str) -> Self {
        let cfg = Config::load(config_path);
        Self {
            count_in: AtomicUsize::new(cfg.counts.in_count),
            count_out: AtomicUsize::new(cfg.counts.out),
            config: RwLock::new(cfg),
            config_path: config_path.into(),
            frame_buffer: RwLock::new(None),
            viewers: RwLock::new(0),
        }
    }
}

pub fn spawn_persist(state: Arc<AppState>) {
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(30));
        let mut cfg = state.config.write().unwrap();
        cfg.counts.in_count = state.count_in.load(Ordering::Relaxed);
        cfg.counts.out = state.count_out.load(Ordering::Relaxed);
        cfg.save(&state.config_path);
    });
}
