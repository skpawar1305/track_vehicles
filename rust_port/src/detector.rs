use crate::types::Detection;
use opencv::{
    core,
    dnn,
    prelude::*,
};

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
    net: dnn::Net,
    target_size: u32,
    conf_thresh: f32,
}

impl Detector {
    pub fn new(model_path: &str, target_size: u32) -> Result<Self, String> {
        let onnx_path = if model_path.ends_with(".onnx") {
            model_path.to_string()
        } else {
            format!("{}.onnx", model_path.trim_end_matches("_ncnn_model"))
        };
        let actual = if std::path::Path::new(&onnx_path).exists() {
            onnx_path
        } else if std::path::Path::new("models/yolo26n.onnx").exists() {
            "models/yolo26n.onnx".to_string()
        } else if std::path::Path::new("../models/yolo26n.onnx").exists() {
            "../models/yolo26n.onnx".to_string()
        } else {
            return Err("No ONNX model found".into());
        };
        let net = dnn::read_net_from_onnx(&actual)
            .map_err(|e| format!("dnn read_net: {}", e))?;
        Ok(Self { net, target_size, conf_thresh: 0.5 })
    }

    pub fn detect(&mut self, mat: &core::Mat, enabled_classes: &[i32]) -> Vec<Detection> {
        let ts = self.target_size as i32;
        let (fh, fw) = (mat.rows(), mat.cols());

        let blob = match dnn::blob_from_image(
            mat, 1.0 / 255.0,
            core::Size::new(ts, ts),
            core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            false, false, core::CV_32F,
        ) {
            Ok(b) => b,
            Err(_) => return vec![],
        };

        if self.net.set_input(&blob, "", 1.0, core::Scalar::new(0.0, 0.0, 0.0, 0.0)).is_err() {
            return vec![];
        }
        let output = match self.net.forward_single_def() {
            Ok(o) => o,
            Err(_) => return vec![],
        };

        let ms = output.mat_size();
        let dims = ms.dims();
        let ch = ms.get(1).unwrap_or(0);
        let num_dets = ms.get(2).unwrap_or(0) as usize;
        drop(ms);
        if dims != 3 || ch != 84 || num_dets == 0 {
            return vec![];
        }
        let num_classes = 80;

        let mat_f32 = match output.try_into_typed::<f32>() {
            Ok(m) => m,
            Err(_) => return vec![],
        };
        let data = match mat_f32.data_typed() {
            Ok(d) => d,
            Err(_) => return vec![],
        };

        let mut candidates: Vec<Detection> = Vec::new();

        for i in 0..num_dets {
            let cx = data[i];
            let cy = data[num_dets + i];
            let w = data[2 * num_dets + i];
            let h = data[3 * num_dets + i];

            let mut best_conf = 0.0f32;
            let mut best_cls = 0i32;
            for c in 0..num_classes {
                let conf = data[(4 + c) * num_dets + i];
                if conf > best_conf {
                    best_conf = conf;
                    best_cls = c as i32;
                }
            }

            if best_conf < self.conf_thresh {
                continue;
            }
            if !enabled_classes.is_empty() && !enabled_classes.contains(&best_cls) {
                continue;
            }

            let x1 = ((cx - w / 2.0).max(0.0).min((fw - 1) as f32)) as i32;
            let y1 = ((cy - h / 2.0).max(0.0).min((fh - 1) as f32)) as i32;
            let x2 = ((cx + w / 2.0).max(0.0).min((fw - 1) as f32)) as i32;
            let y2 = ((cy + h / 2.0).max(0.0).min((fh - 1) as f32)) as i32;
            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            let label = COCO_CLASSES.get(best_cls as usize).unwrap_or(&"?").to_string();
            candidates.push(Detection {
                bbox: [x1, y1, x2, y2],
                centroid: ((x1 + x2) / 2, (y1 + y2) / 2),
                confidence: best_conf,
                class_id: best_cls,
                label,
            });
        }

        if candidates.is_empty() {
            return candidates;
        }

        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut kept = Vec::new();
        let mut removed = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if removed[i] { continue; }
            kept.push(candidates[i].clone());
            for j in i + 1..candidates.len() {
                if removed[j] { continue; }
                let a = &candidates[i].bbox;
                let b = &candidates[j].bbox;
                let ix = a[0].max(b[0]);
                let iy = a[1].max(b[1]);
                let iw = (a[2].min(b[2]) - ix).max(0);
                let ih = (a[3].min(b[3]) - iy).max(0);
                let inter = iw * ih;
                let area_a = (a[2] - a[0]) * (a[3] - a[1]);
                let area_b = (b[2] - b[0]) * (b[3] - b[1]);
                let iou = inter as f32 / (area_a + area_b - inter) as f32 + 1e-6;
                if iou > 0.45 {
                    removed[j] = true;
                }
            }
        }
        kept
    }
}
