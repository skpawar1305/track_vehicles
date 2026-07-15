pub struct MotionDetector {
    background: Option<Vec<u8>>,
    threshold: i32,
    resize_width: u32,
    pub line: Option<[i32; 4]>,
    zone: Option<[i32; 4]>,
    pub motion_state: bool,
}

impl MotionDetector {
    pub fn new(threshold: i32) -> Self {
        Self {
            background: None,
            threshold,
            resize_width: 320,
            line: None,
            zone: None,
            motion_state: false,
        }
    }

    pub fn update_line(&mut self, line: [i32; 4]) {
        self.line = Some(line);
        let pad = 60i32;
        self.zone = Some([
            (line[0].min(line[2]) - pad).max(0),
            (line[1].min(line[3]) - pad).max(0),
            line[0].max(line[2]) + pad,
            line[1].max(line[3]) + pad,
        ]);
    }

    pub fn detect(&mut self, frame: &[u8], fw: u32, fh: u32) -> bool {
        let zone = match self.zone {
            Some(z) => z,
            None => return false,
        };

        let zx1 = zone[0].max(0) as usize;
        let zy1 = zone[1].max(0) as usize;
        let zx2 = (zone[2] as usize).min(fw as usize);
        let zy2 = (zone[3] as usize).min(fh as usize);
        if zx2 <= zx1 || zy2 <= zy1 {
            self.motion_state = false;
            return false;
        }

        let zw = zx2 - zx1;
        let zh = zy2 - zy1;

        let scale = self.resize_width as f32 / zw as f32;
        let rw = self.resize_width as usize;
        let rh = (zh as f32 * scale) as usize;
        if rh == 0 {
            self.motion_state = false;
            return false;
        }

        let mut small = vec![0u8; rw * rh];

        // Simple nearest-neighbor resize of the zone
        for y in 0..rh {
            for x in 0..rw {
                let sx = (x as f32 / scale) as usize + zx1;
                let sy = (y as f32 / scale) as usize + zy1;
                let src_idx = (sy * fw as usize + sx) * 3;
                let dst_idx = (y * rw + x) * 3;
                if src_idx + 3 <= frame.len() && dst_idx + 3 <= small.len() {
                    let gray = (frame[src_idx] as u32 + frame[src_idx + 1] as u32
                        + frame[src_idx + 2] as u32)
                        / 3;
                    small[dst_idx] = gray as u8;
                    small[dst_idx + 1] = gray as u8;
                    small[dst_idx + 2] = gray as u8;
                }
            }
        }

        let fg_pixels = match &self.background {
            Some(bg) if bg.len() == small.len() => {
                let mut count = 0i32;
                for i in (0..small.len()).step_by(3) {
                    let diff = (small[i] as i32 - bg[i] as i32).abs();
                    if diff > 25 {
                        count += 1;
                    }
                }
                count
            }
            _ => 0,
        };

        self.background = Some(small);
        self.motion_state = fg_pixels > self.threshold;
        self.motion_state
    }
}
