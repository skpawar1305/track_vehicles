use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoCapture},
};

pub struct Capture {
    inner: Option<VideoCapture>,
}

impl Capture {
    pub fn open(url: &str, _width: u32, _height: u32) -> Result<Self, String> {
        let cap = VideoCapture::from_file(url, videoio::CAP_ANY)
            .map_err(|e| format!("opencv open: {}", e))?;
        Ok(Self { inner: Some(cap) })
    }

    pub fn read(&mut self) -> Option<core::Mat> {
        let cap = self.inner.as_mut()?;
        let mut mat = core::Mat::default();
        if cap.read(&mut mat).ok().unwrap_or(false) {
            Some(mat)
        } else {
            None
        }
    }

    pub fn close(&mut self) {
        if let Some(ref mut cap) = self.inner {
            let _ = cap.release();
        }
        self.inner = None;
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.close();
    }
}
