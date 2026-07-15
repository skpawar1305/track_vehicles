/// Detection inference backend.
///
/// Currently returns empty results (stub). To enable YOLO detection:
///
/// ## Option 1: ONNX Runtime (ort)
/// ```toml
/// # Cargo.toml
/// ort = { version = "2.0.0-rc.12", features = ["download-binaries"] }
/// ```
/// Then implement `detect()` using `ort::Session`.
/// Export model: `yolo export model=yolo26n.pt format=onnx imgsz=320`
///
/// ## Option 2: ncnn (fastest on Pi)
/// ```toml
/// # Cargo.toml
/// [build-dependencies]
/// cc = "1"
/// ```
/// Then compile `ncnn_wrapper.cpp` in build.rs.
/// Requires ncnn headers + lib installed on the system.
///
/// Swapping the backend only requires changing the `detect()` implementation below.

use crate::types::Detection;

pub struct Detector;

impl Detector {
    pub fn new(_model_path: &str, _target_size: u32) -> Result<Self, String> {
        eprintln!("Detector: compiled without inference backend — no detections will run.");
        eprintln!("See src/detector.rs for instructions to enable ort or ncnn.");
        Ok(Self)
    }

    pub fn detect(&self, _frame: &[u8], _fw: u32, _fh: u32) -> Vec<Detection> {
        vec![]
    }
}
