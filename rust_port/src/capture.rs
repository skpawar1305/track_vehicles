use opencv::{
    core::{self, AlgorithmHint},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
};

struct FFmpegCapture {
    cap: VideoCapture,
}

impl FFmpegCapture {
    fn open(url: &str, width: u32, height: u32) -> Result<Self, String> {
        let mut cap = VideoCapture::from_file(url, videoio::CAP_ANY)
            .map_err(|e| format!("opencv open: {}", e))?;
        cap.set(videoio::CAP_PROP_FRAME_WIDTH, width as f64).ok();
        cap.set(videoio::CAP_PROP_FRAME_HEIGHT, height as f64).ok();
        Ok(Self { cap })
    }

    fn read(&mut self) -> Option<Vec<u8>> {
        let mut mat = core::Mat::default();
        if !self.cap.read(&mut mat).ok().unwrap_or(false) {
            return None;
        }
        let mut rgb = core::Mat::default();
        imgproc::cvt_color(&mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT).ok()?;
        let data = rgb.data_bytes().ok()?;
        Some(data.to_vec())
    }

    fn close(&mut self) {
        let _ = self.cap.release();
    }
}

pub struct Capture {
    inner: Option<FFmpegCapture>,
}

impl Capture {
    pub fn open(url: &str, width: u32, height: u32) -> Result<Self, String> {
        let inner = FFmpegCapture::open(url, width, height)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn read(&mut self) -> Option<Vec<u8>> {
        self.inner.as_mut()?.read()
    }

    pub fn close(&mut self) {
        if let Some(ref mut cap) = self.inner {
            cap.close();
        }
        self.inner = None;
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.close();
    }
}
