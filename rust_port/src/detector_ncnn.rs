use std::ffi::{CString, c_char, c_int, c_float};
use crate::types::Detection;

#[repr(C)]
struct NcnnHandle {
    _unused: [u8; 0],
}

#[derive(Clone)]
#[repr(C)]
struct CDetection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: i32,
}

extern "C" {
    fn ncnn_init(param_path: *const c_char, bin_path: *const c_char) -> *mut NcnnHandle;
    fn ncnn_destroy(handle: *mut NcnnHandle);
    fn ncnn_detect(
        handle: *mut NcnnHandle,
        bgr_data: *const u8,
        img_w: c_int,
        img_h: c_int,
        target_size: c_int,
        conf_thresh: c_float,
        out_detections: *mut CDetection,
        max_detections: c_int,
    ) -> c_int;
}

pub struct NcnnDetector {
    handle: *mut NcnnHandle,
    target_size: u32,
    conf_thresh: f32,
}

impl NcnnDetector {
    pub fn new(model_dir: &str, target_size: u32, conf_thresh: f32) -> Result<Self, String> {
        // Resolve model path relative to project root
        let search_paths = vec![
            model_dir.to_string(),
            format!("../{}", model_dir),
            format!("../models/yolo26n_ncnn_model"),
        ];

        let mut param_path = String::new();
        let mut bin_path = String::new();

        for base in &search_paths {
            let p = std::path::Path::new(base);
            if p.is_dir() {
                for entry in std::fs::read_dir(base).map_err(|e| format!("read_dir: {}", e))? {
                    let entry = entry.map_err(|e| format!("entry: {}", e))?;
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".param") {
                        param_path = entry.path().to_string_lossy().to_string();
                    } else if name.ends_with(".bin") {
                        bin_path = entry.path().to_string_lossy().to_string();
                    }
                }
            } else {
                let pp = format!("{}.param", base);
                let bp = format!("{}.bin", base);
                if std::path::Path::new(&pp).exists() || std::path::Path::new(&bp).exists() {
                    param_path = pp;
                    bin_path = bp;
                }
            }
            if !param_path.is_empty() && !bin_path.is_empty() {
                break;
            }
        }

        if param_path.is_empty() || bin_path.is_empty() {
            return Err(format!("Model files not found for {}", model_dir));
        }

        let c_param = CString::new(param_path.clone()).map_err(|e| format!("CString: {}", e))?;
        let c_bin = CString::new(bin_path.clone()).map_err(|e| format!("CString: {}", e))?;

        let handle = unsafe { ncnn_init(c_param.as_ptr(), c_bin.as_ptr()) };
        if handle.is_null() {
            return Err(format!("ncnn_init failed for {}", model_dir));
        }

        Ok(Self {
            handle,
            target_size,
            conf_thresh,
        })
    }

    pub fn detect(&self, frame: &[u8], fw: u32, fh: u32) -> Vec<Detection> {
        const MAX_DET: i32 = 100;
        let mut cdets = vec![
            CDetection {
                x1: 0.0, y1: 0.0, x2: 0.0, y2: 0.0,
                confidence: 0.0, class_id: 0
            };
            MAX_DET as usize
        ];

        let count = unsafe {
            ncnn_detect(
                self.handle,
                frame.as_ptr(),
                fw as i32,
                fh as i32,
                self.target_size as i32,
                self.conf_thresh,
                cdets.as_mut_ptr(),
                MAX_DET,
            )
        };

        let mut results = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let d = &cdets[i];
            let x1 = d.x1 as i32;
            let y1 = d.y1 as i32;
            let x2 = d.x2 as i32;
            let y2 = d.y2 as i32;
            results.push(Detection {
                bbox: [x1, y1, x2, y2],
                centroid: ((x1 + x2) / 2, (y1 + y2) / 2),
                confidence: d.confidence,
                class_id: d.class_id,
                label: format!("cls_{}", d.class_id),
            });
        }

        results
    }
}

impl Drop for NcnnDetector {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ncnn_destroy(self.handle) }
        }
    }
}
