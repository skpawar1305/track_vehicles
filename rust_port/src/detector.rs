use crate::types::Detection;
use crate::ncnn_wrapper;
use opencv::prelude::*;

const COCO_CLASSES: &[&str] = &[
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
];

pub struct Detector {
    net: ncnn_wrapper::NcnnNet,
    target_size: i32,
    conf_thresh: f32,
}

impl Detector {
    pub fn new(model_path: &str, target_size: u32) -> Result<Self, String> {
        let ts = target_size as i32;
        let model_dir = if model_path.ends_with("_ncnn_model") || model_path.contains("ncnn") {
            model_path.to_string()
        } else {
            format!("{}_ncnn_model", model_path.trim_end_matches(".onnx"))
        };

        let param_path = if model_path.ends_with(".param") {
            model_path.to_string()
        } else {
            format!("{}/model.ncnn.param", model_dir)
        };
        let bin_path = if model_path.ends_with(".bin") {
            model_path.to_string()
        } else {
            format!("{}/model.ncnn.bin", model_dir)
        };

        let param_path = resolve_path(&param_path);
        let bin_path = resolve_path(&bin_path);

        let net = ncnn_wrapper::NcnnNet::new().ok_or("ncnn net create failed")?;
        net.load_param(&param_path)?;
        net.load_model(&bin_path)?;

        Ok(Self { net, target_size: ts, conf_thresh: 0.5 })
    }

    pub fn detect(&mut self, mat: &opencv::core::Mat, enabled_classes: &[i32]) -> Vec<Detection> {
        let ts = self.target_size;
        let (fh, fw) = (mat.rows(), mat.cols());
        if fh <= 0 || fw <= 0 { return vec![]; }

        // Letterbox
        let scale = (ts as f32 / fw as f32).min(ts as f32 / fh as f32);
        let new_w = (fw as f32 * scale) as i32;
        let new_h = (fh as f32 * scale) as i32;
        let pad_x = (ts - new_w) / 2;
        let pad_y = (ts - new_h) / 2;

        // Build padded letterbox image using OpenCV (fast)
        let mut resized = opencv::core::Mat::default();
        if opencv::imgproc::resize(mat, &mut resized, opencv::core::Size::new(new_w, new_h), 0.0, 0.0, opencv::imgproc::INTER_LINEAR).is_err() {
            return vec![];
        }

        let mut letterbox = match opencv::core::Mat::new_rows_cols_with_default(ts, ts, opencv::core::CV_8UC3, opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0)) {
            Ok(m) => m,
            Err(_) => return vec![],
        };
        let roi = opencv::core::Rect::new(pad_x, pad_y, new_w, new_h);
        if let Ok(mut target_roi) = letterbox.roi_mut(roi) {
            if resized.copy_to(&mut target_roi).is_err() { return vec![]; }
        } else {
            return vec![];
        }
        drop(resized);

        // Convert BGR letterbox → CHW float [0, 1] (fast bulk operation)
        let ts_u = ts as usize;
        let total = (ts_u * ts_u) as usize;
        let mut float_data = vec![0.0f32; total * 3];

        if let Ok(pixels) = letterbox.data_bytes() {
            for i in 0..total {
                let b = pixels[i * 3] as f32 / 255.0;
                let g = pixels[i * 3 + 1] as f32 / 255.0;
                let r = pixels[i * 3 + 2] as f32 / 255.0;
                float_data[i] = b;
                float_data[total + i] = g;
                float_data[2 * total + i] = r;
            }
        }

        // Run ncnn inference
        let ex = self.net.create_extractor();
        if ex.input_bgr_normalized("in0", &float_data, ts, ts, 3).is_err() {
            return vec![];
        }
        let out_mat = match ex.extract("out0") {
            Ok(m) => m,
            Err(_) => return vec![],
        };

        let data = out_mat.data_f32();
        let (w, h, _c) = out_mat.shape();

        // ncnn Mat shape: (w, h, c) = (spatial, channels, batch)
        let num_dets = w as usize;
        let num_channels = h as usize;

        if num_channels < 5 || data.len() < num_channels * num_dets {
            return vec![];
        }

        let num_classes = num_channels - 4; // 4 bbox + class scores

        let mut candidates: Vec<Detection> = Vec::new();

        let mut raw_count = 0usize;
        for i in 0..num_dets {
            let cx = data[i];
            let cy = data[1 * num_dets + i];
            let w = data[2 * num_dets + i];
            let h = data[3 * num_dets + i];

            let mut best_conf = 0.0f32;
            let mut best_cls = 0i32;
            for c in 0..num_classes {
                let conf = data[(4 + c) * num_dets + i];
                if conf > best_conf { best_conf = conf; best_cls = c as i32; }
            }

            if best_conf < self.conf_thresh { continue; }
            raw_count += 1;
            if !enabled_classes.is_empty() && !enabled_classes.contains(&best_cls) { continue; }

            // Map from padded letterbox space to original frame
            let x1 = ((cx - w / 2.0 - pad_x as f32) / scale) as i32;
            let y1 = ((cy - h / 2.0 - pad_y as f32) / scale) as i32;
            let x2 = ((cx + w / 2.0 - pad_x as f32) / scale) as i32;
            let y2 = ((cy + h / 2.0 - pad_y as f32) / scale) as i32;

            let x1 = x1.max(0).min(fw - 1);
            let y1 = y1.max(0).min(fh - 1);
            let x2 = x2.max(0).min(fw - 1);
            let y2 = y2.max(0).min(fh - 1);

            if x2 <= x1 || y2 <= y1 { continue; }
            candidates.push(Detection {
                bbox: [x1, y1, x2, y2],
                centroid: ((x1 + x2) / 2, (y1 + y2) / 2),
                confidence: best_conf,
                class_id: best_cls,
                label: COCO_CLASSES.get(best_cls as usize).unwrap_or(&"?").to_string(),
            });
        }

        candidates
    }
}

fn resolve_path(path: &str) -> String {
    if std::path::Path::new(path).exists() {
        return path.to_string();
    }
    // Try relative to cwd/models/
    let alt = format!("models/{}", path.trim_start_matches("../models/"));
    if std::path::Path::new(&alt).exists() {
        return alt;
    }
    path.to_string()
}

fn non_max_suppression(mut dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    if dets.is_empty() { return dets; }
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    while let Some(best) = dets.pop() {
        keep.push(best.clone());
        dets.retain(|d| {
            let iou = iou(&d.bbox, &best.bbox);
            iou < iou_thresh
        });
    }
    keep
}

fn iou(a: &[i32; 4], b: &[i32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0) * (y2 - y1).max(0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter as f32 / (area_a + area_b - inter) as f32 + 1e-6
}
