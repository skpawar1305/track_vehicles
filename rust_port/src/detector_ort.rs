use opencv::{
    core,
    imgproc,
    prelude::*,
};
use ort::{session, value::Tensor};

const INPUT_SIZE: u32 = 416;
const STRIDES: [u32; 4] = [8, 16, 32, 64];
const MEAN: [f32; 3] = [103.53, 116.28, 123.675];
const STD: [f32; 3] = [57.375, 57.12, 58.395];
const VEHICLE_NAMES: [(i32, &str); 4] = [(2, "car"), (3, "motorcycle"), (5, "bus"), (7, "truck")];

pub struct NanoDetONNX {
    session: session::Session,
    priors: Vec<(f32, f32, f32)>,
    conf_thresh: f32,
    iou_thresh: f32,
}

impl NanoDetONNX {
    pub fn new(model_path: &str, conf_thresh: f32, iou_thresh: f32) -> Result<Self, String> {
        let mut priors = Vec::new();
        for stride in STRIDES {
            let h = (INPUT_SIZE as f32 / stride as f32).ceil() as u32;
            let w = (INPUT_SIZE as f32 / stride as f32).ceil() as u32;
            for i in 0..h {
                for j in 0..w {
                    priors.push((j as f32 * stride as f32, i as f32 * stride as f32, stride as f32));
                }
            }
        }

        let session = session::Session::builder()
            .map_err(|e| format!("Session builder: {}", e))?
            .with_optimization_level(session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Optimization level: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| format!("Intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Commit model: {}", e))?;

        Ok(Self { session, priors, conf_thresh, iou_thresh })
    }

    pub fn detect(&mut self, frame: &core::Mat, enabled_classes: &[i32]) -> Result<Vec<Detection>, String> {
        let h = frame.rows();
        let w = frame.cols();
        let scale = (INPUT_SIZE as f32 / h as f32).min(INPUT_SIZE as f32 / w as f32);
        let new_h = (h as f32 * scale) as i32;
        let new_w = (w as f32 * scale) as i32;
        let pad_h = (INPUT_SIZE as i32 - new_h) / 2;
        let pad_w = (INPUT_SIZE as i32 - new_w) / 2;

        let mut resized = core::Mat::default();
        imgproc::resize(frame, &mut resized, core::Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)
            .map_err(|e| format!("Resize: {}", e))?;

        let mut canvas = core::Mat::new_rows_cols_with_default(INPUT_SIZE as i32, INPUT_SIZE as i32, core::CV_8UC3, core::Scalar::all(114.0))
            .map_err(|e| format!("Canvas: {}", e))?;

        {
            let roi = core::Rect::new(pad_w, pad_h, new_w, new_h);
            let mut target = core::Mat::roi_mut(&mut canvas, roi).map_err(|e| format!("Roi: {}", e))?;
            resized.copy_to(&mut target).map_err(|e| format!("Copy: {}", e))?;
        }

        let mut f32_mat = core::Mat::default();
        canvas.convert_to(&mut f32_mat, core::CV_32FC3, 1.0, 0.0).map_err(|e| format!("Convert: {}", e))?;

        // Build NCHW tensor with mean/std normalization (matching Python NanoDetONXX._preprocess)
        let total = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut blob = vec![0.0f32; 3 * total];
        for y in 0..INPUT_SIZE as i32 {
            for x in 0..INPUT_SIZE as i32 {
                let px = f32_mat.at_2d::<core::Vec3f>(y, x).map_err(|e| format!("at_2d: {}", e))?;
                let idx = (y as usize) * (INPUT_SIZE as usize) + x as usize;
                blob[0 * total + idx] = (px[0] - MEAN[0]) / STD[0];
                blob[1 * total + idx] = (px[1] - MEAN[1]) / STD[1];
                blob[2 * total + idx] = (px[2] - MEAN[2]) / STD[2];
            }
        }

        // Create input tensor and run inference
        let shape = vec![1i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let input_tensor: ort::value::Value = ort::value::Tensor::from_array((shape, blob.into_boxed_slice()))
            .map_err(|e| format!("Tensor: {}", e))?.into();

        let outputs = self.session.run(ort::inputs!["data" => input_tensor])
            .map_err(|e| format!("Run: {}", e))?;

        let (shape_dims, data) = outputs["output"].try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract: {}", e))?;
        let sh = shape_dims.as_ref();
        if sh.len() < 3 { return Ok(vec![]); }
        let num_priors = sh[1] as usize;
        let num_channels = sh[2] as usize;
        let output_data: Vec<f32> = data.to_vec();
        drop(outputs);

        let mut candidates: Vec<Detection> = Vec::new();
        for i in 0..num_priors {
            let base = i * num_channels;
            let mut best_cls = 0usize;
            let mut best_score = output_data[base];
            for c in 1..80 {
                let s = output_data[base + c];
                if s > best_score { best_score = s; best_cls = c; }
            }
            if best_score < self.conf_thresh { continue; }
            if !enabled_classes.is_empty() && !enabled_classes.contains(&(best_cls as i32)) { continue; }

            let (cx, cy, stride) = self.priors[i];
            let dist_base = base + 80;
            let mut dists = [0.0f32; 4];
            for d in 0..4 {
                let bin_base = dist_base + d * 8;
                let mut max_v = output_data[bin_base];
                for k in 1..8 { if output_data[bin_base + k] > max_v { max_v = output_data[bin_base + k]; } }
                let mut sum = 0.0f32;
                for k in 0..8 { sum += (output_data[bin_base + k] - max_v).exp(); }
                for k in 0..8 {
                    let p = (output_data[bin_base + k] - max_v).exp() / sum;
                    dists[d] += p * k as f32;
                }
                dists[d] *= stride;
            }

            let mut x1 = ((cx - dists[0] - pad_w as f32) / scale) as i32;
            let mut y1 = ((cy - dists[1] - pad_h as f32) / scale) as i32;
            let mut x2 = ((cx + dists[2] - pad_w as f32) / scale) as i32;
            let mut y2 = ((cy + dists[3] - pad_h as f32) / scale) as i32;

            x1 = x1.max(0).min(w - 1);
            y1 = y1.max(0).min(h - 1);
            x2 = x2.max(0).min(w - 1);
            y2 = y2.max(0).min(h - 1);

            if x2 <= x1 || y2 <= y1 { continue; }

            let label = VEHICLE_NAMES.iter().find(|&&(id, _)| id == best_cls as i32)
                .map(|&(_, n)| n.to_string())
                .unwrap_or_else(|| format!("cls_{}", best_cls));

            candidates.push(Detection {
                bbox: [x1, y1, x2, y2],
                centroid: ((x1 + x2) / 2, (y1 + y2) / 2),
                confidence: best_score,
                class_id: best_cls as i32,
                label,
            });
        }

        if candidates.is_empty() { return Ok(vec![]); }
        let boxes: Vec<[f32; 4]> = candidates.iter().map(|d| [d.bbox[0] as f32, d.bbox[1] as f32, d.bbox[2] as f32, d.bbox[3] as f32]).collect();
        let scores: Vec<f32> = candidates.iter().map(|d| d.confidence).collect();
        let keep = self.nms(&boxes, &scores);
        let result: Vec<Detection> = keep.into_iter().map(|i| candidates[i].clone()).collect();
        Ok(result)
    }

    fn nms(&self, boxes: &[[f32; 4]], scores: &[f32]) -> Vec<usize> {
        if boxes.is_empty() { return vec![]; }
        let len = boxes.len();
        let mut order: Vec<usize> = (0..len).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep = Vec::new();
        let mut suppressed = vec![false; len];
        for &i in &order {
            if suppressed[i] { continue; }
            keep.push(i);
            for &j in &order {
                if suppressed[j] { continue; }
                let inter_x1 = boxes[i][0].max(boxes[j][0]);
                let inter_y1 = boxes[i][1].max(boxes[j][1]);
                let inter_x2 = boxes[i][2].min(boxes[j][2]);
                let inter_y2 = boxes[i][3].min(boxes[j][3]);
                let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
                let area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]);
                let iou = inter / (area_i + area_j - inter + 1e-6);
                if iou > self.iou_thresh { suppressed[j] = true; }
            }
        }
        keep
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [i32; 4],
    pub centroid: (i32, i32),
    pub confidence: f32,
    pub class_id: i32,
    pub label: String,
}
