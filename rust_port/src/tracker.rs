use std::collections::HashMap;
use crate::types::Detection;

#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub track_id: u32,
    pub label: String,
    pub confidence: f32,
    pub active: bool,
    pub bboxes: Vec<[i32; 4]>,
    pub centroids: Vec<(i32, i32)>,
    pub disappeared: u32,
    pub age: u32,
    pub last_crossing: Option<&'static str>,
}

impl TrackedObject {
    pub fn centroid(&self) -> (i32, i32) {
        *self.centroids.last().unwrap_or(&(0, 0))
    }

    pub fn prev_centroid(&self) -> Option<(i32, i32)> {
        if self.centroids.len() >= 2 {
            Some(self.centroids[self.centroids.len() - 2])
        } else {
            None
        }
    }

    pub fn bbox(&self) -> &[i32; 4] {
        self.bboxes.last().unwrap()
    }

    pub fn predict(&self) -> [i32; 4] {
        let (dx, dy) = if self.centroids.len() >= 3 {
            let n = self.centroids.len();
            let dx1 = self.centroids[n - 1].0 - self.centroids[n - 2].0;
            let dx2 = self.centroids[n - 2].0 - self.centroids[n - 3].0;
            let dy1 = self.centroids[n - 1].1 - self.centroids[n - 2].1;
            let dy2 = self.centroids[n - 2].1 - self.centroids[n - 3].1;
            let dx = if dx1.abs() > dx2.abs() { dx1 } else { dx2 };
            let dy = if dy1.abs() > dy2.abs() { dy1 } else { dy2 };
            (dx, dy)
        } else if self.centroids.len() == 2 {
            let dx = self.centroids[1].0 - self.centroids[0].0;
            let dy = self.centroids[1].1 - self.centroids[0].1;
            (dx, dy)
        } else {
            (0, 0)
        };
        let b = self.bbox();
        [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
    }

    pub fn update(&mut self, bbox: [i32; 4], confidence: f32) {
        let cx = (bbox[0] + bbox[2]) / 2;
        let cy = (bbox[1] + bbox[3]) / 2;
        self.bboxes.push(bbox);
        self.centroids.push((cx, cy));
        if self.bboxes.len() > 10 {
            self.bboxes.remove(0);
            self.centroids.remove(0);
        }
        self.confidence = confidence;
        self.disappeared = 0;
        self.age += 1;
    }
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

fn iou_cost(tracks: &[&TrackedObject], dets: &[&Detection]) -> Vec<Vec<f32>> {
    let mut cost = vec![vec![0.0f32; dets.len()]; tracks.len()];
    for (i, t) in tracks.iter().enumerate() {
        let pred = t.predict();
        let last = *t.bbox();
        for (j, d) in dets.iter().enumerate() {
            let iou_best = iou(&pred, &d.bbox).max(iou(&last, &d.bbox));
            cost[i][j] = 1.0 - iou_best;
        }
    }
    cost
}

fn greedy_match(cost: &mut Vec<Vec<f32>>, max_cost: f32) -> (Vec<bool>, Vec<bool>, Vec<(usize, usize)>) {
    let rows = cost.len();
    let cols = if rows == 0 { 0 } else { cost[0].len() };
    let mut matched_t = vec![false; rows];
    let mut matched_d = vec![false; cols];
    let mut pairs = Vec::new();

    for _ in 0..rows.min(cols) {
        let mut best = max_cost + 1.0;
        let mut bi = 0;
        let mut bj = 0;
        for i in 0..rows {
            if matched_t[i] { continue; }
            for j in 0..cols {
                if matched_d[j] { continue; }
                if cost[i][j] < best {
                    best = cost[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        if best > max_cost {
            break;
        }
        matched_t[bi] = true;
        matched_d[bj] = true;
        pairs.push((bi, bj));
    }
    (matched_t, matched_d, pairs)
}

pub struct ByteTrack {
    next_id: u32,
    pub objects: HashMap<u32, TrackedObject>,
    conf_thresh: f32,
    max_lost: u32,
    iou_thresh: f32,
    second_iou_thresh: f32,
}

impl ByteTrack {
    pub fn new(conf_thresh: f32) -> Self {
        Self {
            next_id: 0,
            objects: HashMap::new(),
            conf_thresh,
            max_lost: 30,
            iou_thresh: 0.15,
            second_iou_thresh: 0.1,
        }
    }

    pub fn update(&mut self, detections: Vec<Detection>) -> Vec<u32> {
        if detections.is_empty() {
            let mut died = Vec::new();
            for (&tid, obj) in &mut self.objects {
                obj.disappeared += 1;
                if obj.disappeared > self.max_lost {
                    died.push(tid);
                }
            }
            for tid in died {
                self.objects.remove(&tid);
            }
            return self.objects.keys().copied().collect();
        }

        let high: Vec<&Detection> = detections.iter().filter(|d| d.confidence >= self.conf_thresh).collect();
        let _low: Vec<&Detection> = detections.iter().filter(|d| d.confidence < self.conf_thresh).collect();

        let active_ids: Vec<u32> = self.objects.iter().filter(|(_, o)| o.active).map(|(&id, _)| id).collect();
        let mut matched_active: Vec<usize> = Vec::new();
        let mut matched_high: Vec<usize> = Vec::new();

        if !active_ids.is_empty() && !high.is_empty() {
            let t_list: Vec<&TrackedObject> = active_ids.iter().map(|id| &self.objects[id]).collect();
            let mut cost = iou_cost(&t_list, &high);
            let (mt, md, pairs) = greedy_match(&mut cost, 1.0 - self.iou_thresh);
            for (ti, di) in &pairs {
                let tid = active_ids[*ti];
                if let Some(obj) = self.objects.get_mut(&tid) {
                    obj.update(high[*di].bbox, high[*di].confidence);
                }
            }
            matched_active = (0..active_ids.len()).filter(|&i| mt[i]).collect();
            matched_high = (0..high.len()).filter(|&i| md[i]).collect();
        }

        let unmatched_active: Vec<u32> = active_ids.iter().enumerate()
            .filter(|(i, _)| !matched_active.contains(i))
            .map(|(_, &id)| id)
            .collect();

        let unmatched_high: Vec<&Detection> = high.iter().enumerate()
            .filter(|(i, _)| !matched_high.contains(i))
            .map(|(_, d)| *d)
            .collect();

        // Unmatched active -> increment disappeared
        for &tid in &unmatched_active {
            if let Some(obj) = self.objects.get_mut(&tid) {
                obj.disappeared += 1;
                if obj.disappeared > self.max_lost {
                    // Don't remove here, will be cleaned next iteration
                }
            }
        }

        // New tracks for unmatched high
        for d in unmatched_high {
            let tid = self.next_id;
            self.next_id += 1;
            let cx = (d.bbox[0] + d.bbox[2]) / 2;
            let cy = (d.bbox[1] + d.bbox[3]) / 2;
            self.objects.insert(tid, TrackedObject {
                track_id: tid,
                label: d.label.clone(),
                confidence: d.confidence,
                active: true,
                bboxes: vec![d.bbox],
                centroids: vec![(cx, cy)],
                disappeared: 0,
                age: 0,
                last_crossing: None,
            });
        }

        // Clean dead tracks
        let died: Vec<u32> = self.objects.iter()
            .filter(|(_, o)| o.disappeared > self.max_lost)
            .map(|(&id, _)| id)
            .collect();
        for tid in died {
            self.objects.remove(&tid);
        }

        self.objects.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Detection;

    #[test]
    fn test_iou_overlap() {
        let a = [0, 0, 100, 100];
        let b = [50, 50, 150, 150];
        let i = iou(&a, &b);
        assert!(i > 0.0 && i < 1.0);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = [0, 0, 10, 10];
        let b = [100, 100, 110, 110];
        let i = iou(&a, &b);
        assert!(i < 0.01);
    }

    #[test]
    fn test_iou_identical() {
        let a = [0, 0, 100, 100];
        let i = iou(&a, &a);
        assert!((i - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tracker_empty_detections() {
        let mut tracker = ByteTrack::new(0.5);
        let ids = tracker.update(vec![]);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_tracker_new_detection() {
        let mut tracker = ByteTrack::new(0.5);
        let det = Detection {
            bbox: [0, 0, 50, 50],
            centroid: (25, 25),
            confidence: 0.9,
            class_id: 2,
            label: "car".into(),
        };
        let ids = tracker.update(vec![det]);
        assert_eq!(ids.len(), 1);
        assert_eq!(tracker.objects.len(), 1);
    }

    #[test]
    fn test_tracker_matching() {
        let mut tracker = ByteTrack::new(0.5);
        let det = Detection {
            bbox: [0, 0, 50, 50],
            centroid: (25, 25),
            confidence: 0.9,
            class_id: 2,
            label: "car".into(),
        };
        tracker.update(vec![det.clone()]);
        // Same position, should match existing track
        let ids = tracker.update(vec![det]);
        assert_eq!(ids.len(), 1);
        assert_eq!(tracker.objects.len(), 1);
    }

    #[test]
    fn test_tracker_disappear() {
        let mut tracker = ByteTrack::new(0.5);
        tracker.max_lost = 2;
        let det = Detection {
            bbox: [0, 0, 50, 50],
            centroid: (25, 25),
            confidence: 0.9,
            class_id: 2,
            label: "car".into(),
        };
        tracker.update(vec![det]);
        tracker.update(vec![]); // 1 missed
        tracker.update(vec![]); // 2 missed
        let ids = tracker.update(vec![]); // 3 missed > max_lost
        assert!(ids.is_empty());
        assert!(tracker.objects.is_empty());
    }

    #[test]
    fn test_greedy_match() {
        let mut cost = vec![vec![0.1f32, 0.8f32], vec![0.9f32, 0.2f32]];
        let (mt, md, pairs) = greedy_match(&mut cost, 0.5);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 1)));
        assert!(mt.iter().all(|&x| x));
        assert!(md.iter().all(|&x| x));
    }
}
