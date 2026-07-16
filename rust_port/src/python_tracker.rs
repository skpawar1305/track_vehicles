use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;

pub struct PythonTracker {
    stdin: Mutex<std::process::ChildStdin>,
    stdout: Mutex<BufReader<std::process::ChildStdout>>,
}

impl PythonTracker {
    pub fn new(track_thresh: f32, track_buffer: usize, match_thresh: f32, frame_rate: usize) -> Result<Self, String> {
        let script = format!(r#"
import ctypes as _ctypes
for _lib in ['libcudart.so.13', 'libcublas.so.13', 'libcublasLt.so.13']:
    try: _ctypes.CDLL(f'/usr/local/lib/ollama/cuda_v13/{{_lib}}', mode=_ctypes.RTLD_GLOBAL)
    except: pass

import sys as _sys
_sys.path = [p for p in _sys.path if 'robostack' not in p and 'tyno_ws' not in p and 'python3.12' not in p]

import sys, json
from bytetracker import BYTETracker
import numpy as np
import torch

tracker = BYTETracker(track_thresh={}, track_buffer={}, match_thresh={}, frame_rate={})
while True:
    line = sys.stdin.readline()
    if not line: break
    data = json.loads(line)
    if data["dets"]:
        arr = np.array(data["dets"], dtype=np.float32)
    else:
        arr = np.empty((0, 6), dtype=np.float32)
    tracked = tracker.update(torch.from_numpy(arr), None)
    result = []
    if len(tracked) > 0:
        for t in tracked:
            x1, y1, x2, y2, tid, cls_id, score = t
            cx = (int(x1) + int(x2)) // 2
            cy = (int(y1) + int(y2)) // 2
            result.append({{"tid": int(tid), "bbox": [int(x1), int(y1), int(x2), int(y2)], "centroid": [cx, cy]}})
    sys.stdout.write(json.dumps(result) + '\n')
    sys.stdout.flush()
"#, track_thresh, track_buffer, match_thresh, frame_rate);

        let mut child = Command::new("python3")
            .args(&["-c", &script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Python spawn: {}", e))?;

        let stdin = child.stdin.take().ok_or("No stdin")?;
        let stdout = child.stdout.take().ok_or("No stdout")?;

        Ok(Self {
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
        })
    }

    pub fn update(&self, detections: &[super::detector_ort::Detection]) -> Result<Vec<TrackResult>, String> {
        let dets: Vec<Vec<f32>> = detections.iter().map(|d| {
            vec![d.bbox[0] as f32, d.bbox[1] as f32, d.bbox[2] as f32, d.bbox[3] as f32,
                 d.confidence, d.class_id as f32]
        }).collect();

        let input = serde_json::json!({"dets": dets}).to_string();

        let mut stdin = self.stdin.lock().map_err(|e| format!("stdin lock: {}", e))?;
        stdin.write_all(input.as_bytes()).map_err(|e| format!("write: {}", e))?;
        stdin.write_all(b"\n").map_err(|e| format!("write newline: {}", e))?;
        stdin.flush().map_err(|e| format!("flush: {}", e))?;
        drop(stdin);

        let mut stdout = self.stdout.lock().map_err(|e| format!("stdout lock: {}", e))?;
        let mut line = String::new();
        stdout.read_line(&mut line).map_err(|e| format!("read: {}", e))?;

        let result: Vec<TrackResult> = serde_json::from_str(&line)
            .map_err(|e| format!("JSON: {} from: {}", e, line.trim()))?;
        Ok(result)
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrackResult {
    pub tid: usize,
    pub bbox: [i32; 4],
    pub centroid: [i32; 2],
}
