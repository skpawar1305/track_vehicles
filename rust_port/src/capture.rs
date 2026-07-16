use std::io::Read;
use std::process::{Command, Stdio};
use opencv::{
    core,
    prelude::*,
};

const W: i32 = 640;
const H: i32 = 360;

pub struct Capture {
    child: Option<std::process::Child>,
    stdout: Option<std::process::ChildStdout>,
    url: String,
}

impl Capture {
    pub fn open(url: &str, _width: u32, _height: u32) -> Result<Self, String> {
        // Pipe raw BGR24 frames from ffmpeg — no quality loss
        let mut child = Command::new("ffmpeg")
            .args(&[
                "-i", url,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", &format!("{}x{}", W, H),
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("ffmpeg spawn: {}", e))?;
        let stdout = child.stdout.take().ok_or("No stdout")?;
        Ok(Self {
            child: Some(child),
            stdout: Some(stdout),
            url: url.to_string(),
        })
    }

    pub fn read(&mut self) -> Option<core::Mat> {
        let reader = self.stdout.as_mut()?;
        let frame_size = (W * H * 3) as usize;
        let mut buffer = vec![0u8; frame_size];
        let mut pos = 0;
        while pos < frame_size {
            match reader.read(&mut buffer[pos..]) {
                Ok(0) => return None,
                Ok(n) => pos += n,
                Err(_) => return None,
            }
        }
        // Create Mat from raw buffer without copying
        let mat = unsafe {
            core::Mat::new_rows_cols_with_data_unsafe(H, W, core::CV_8UC3,
                buffer.as_mut_ptr() as *mut std::ffi::c_void, (W * 3) as usize)
        };
        match mat {
            Ok(m) => {
                std::mem::forget(buffer); // Keep buffer alive
                Some(m)
            }
            Err(_) => None,
        }
    }

    pub fn close(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.close();
    }
}
