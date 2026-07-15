use opencv::{
    core,
    imgproc,
    prelude::*,
    video,
};

pub struct MotionDetector {
    mog2: core::Ptr<video::BackgroundSubtractorMOG2>,
    resize_width: u32,
    pub line: Option<[i32; 4]>,
    zone: Option<[i32; 4]>,
    pub motion_state: bool,
}

impl MotionDetector {
    pub fn new(_threshold: i32) -> Result<Self, String> {
        let mog2 = video::create_background_subtractor_mog2(500, 16.0, false)
            .map_err(|e| format!("MOG2: {}", e))?;
        Ok(Self {
            mog2,
            resize_width: 320,
            line: None,
            zone: None,
            motion_state: false,
        })
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

    pub fn detect(&mut self, frame: &core::Mat) -> bool {
        let zone = match self.zone {
            Some(z) => z,
            None => return false,
        };

        let (fw, fh) = (frame.cols(), frame.rows());
        let zx1 = zone[0].max(0);
        let zy1 = zone[1].max(0);
        let zx2 = (zone[2]).min(fw);
        let zy2 = (zone[3]).min(fh);
        if zx2 <= zx1 || zy2 <= zy1 {
            self.motion_state = false;
            return false;
        }

        let zw = zx2 - zx1;
        let zh = zy2 - zy1;

        let roi = core::Rect::new(zx1, zy1, zw, zh);
        let crop = core::Mat::roi(frame, roi).ok().unwrap();

        let rw = self.resize_width as i32;
        let rh = (zh as f64 * self.resize_width as f64 / zw as f64) as i32;
        if rh <= 0 {
            self.motion_state = false;
            return false;
        }

        let mut small = core::Mat::default();
        imgproc::resize(&crop, &mut small, core::Size::new(rw, rh), 0.0, 0.0, imgproc::INTER_LINEAR)
            .ok();

        let mut fgmask = core::Mat::default();
        opencv::video::BackgroundSubtractorTrait::apply(&mut self.mog2, &small, &mut fgmask, -1.0).ok();

        let fg_count = core::count_non_zero(&fgmask).ok().unwrap_or(0);
        self.motion_state = fg_count > self.resize_width as i32 * 2;
        self.motion_state
    }
}
