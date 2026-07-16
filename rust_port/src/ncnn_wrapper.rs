#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
pub type size_t = usize;

type ncnn_allocator_t = *mut c_void;
type ncnn_option_t = *mut c_void;

pub type ncnn_net_t = *mut c_void;
pub type ncnn_mat_t = *mut c_void;
pub type ncnn_extractor_t = *mut c_void;

const NCNN_MAT_PIXEL_BGR: c_int = 2;

extern "C" {
    fn ncnn_net_create() -> ncnn_net_t;
    fn ncnn_net_destroy(net: ncnn_net_t);
    fn ncnn_net_load_param(net: ncnn_net_t, path: *const c_char) -> c_int;
    fn ncnn_net_load_model(net: ncnn_net_t, path: *const c_char) -> c_int;

    fn ncnn_mat_from_pixels_resize(
        pixels: *const u8, ptype: c_int, w: c_int, h: c_int, stride: c_int,
        target_w: c_int, target_h: c_int, allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;

    fn ncnn_mat_from_pixels(
        pixels: *const u8, ptype: c_int, w: c_int, h: c_int, stride: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;

    fn ncnn_mat_get_w(mat: ncnn_mat_t) -> c_int;
    fn ncnn_mat_get_h(mat: ncnn_mat_t) -> c_int;
    fn ncnn_mat_get_c(mat: ncnn_mat_t) -> c_int;
    fn ncnn_mat_get_data(mat: ncnn_mat_t) -> *mut c_void;
    fn ncnn_mat_destroy(mat: ncnn_mat_t);
    fn ncnn_mat_create_external_3d_elem(w: c_int, h: c_int, c: c_int, data: *mut c_void, elemsize: size_t, elempack: c_int, allocator: ncnn_allocator_t) -> ncnn_mat_t;

    fn ncnn_extractor_create(net: ncnn_net_t) -> ncnn_extractor_t;
    fn ncnn_extractor_destroy(ex: ncnn_extractor_t);
    fn ncnn_extractor_input(
        ex: ncnn_extractor_t, name: *const c_char, mat: ncnn_mat_t,
    ) -> c_int;
    fn ncnn_extractor_extract(
        ex: ncnn_extractor_t, name: *const c_char, mat: *mut ncnn_mat_t,
    ) -> c_int;
}

pub struct NcnnNet {
    net: ncnn_net_t,
}

impl NcnnNet {
    pub fn new() -> Option<Self> {
        let net = unsafe { ncnn_net_create() };
        if net.is_null() {
            return None;
        }
        Some(Self { net })
    }

    pub fn load_param(&self, path: &str) -> Result<(), String> {
        let cpath = CString::new(path).map_err(|e| format!("CString: {}", e))?;
        let ret = unsafe { ncnn_net_load_param(self.net, cpath.as_ptr()) };
        if ret != 0 {
            return Err(format!("ncnn load param failed: {}", ret));
        }
        Ok(())
    }

    pub fn load_model(&self, path: &str) -> Result<(), String> {
        let cpath = CString::new(path).map_err(|e| format!("CString: {}", e))?;
        let ret = unsafe { ncnn_net_load_model(self.net, cpath.as_ptr()) };
        if ret != 0 {
            return Err(format!("ncnn load model failed: {}", ret));
        }
        Ok(())
    }

    pub fn create_extractor(&self) -> NcnnExtractor {
        let ex = unsafe { ncnn_extractor_create(self.net) };
        NcnnExtractor { ex }
    }
}

impl Drop for NcnnNet {
    fn drop(&mut self) {
        unsafe { ncnn_net_destroy(self.net) }
    }
}

pub struct NcnnExtractor {
    ex: ncnn_extractor_t,
}

impl NcnnExtractor {
    pub fn input_bgr_normalized(&self, name: &str, data: &[f32], w: i32, h: i32, c: i32) -> Result<(), String> {
        let cname = CString::new(name).map_err(|e| format!("CString: {}", e))?;
        let ptr = data.as_ptr() as *mut c_void;
        let mat = unsafe {
            ncnn_mat_create_external_3d_elem(w, h, c, ptr, 4, 1, std::ptr::null_mut())
        };
        if mat.is_null() {
            return Err("ncnn mat create external failed".into());
        }
        let ret = unsafe { ncnn_extractor_input(self.ex, cname.as_ptr(), mat) };
        // Don't destroy the mat since it uses external data
        if ret != 0 {
            return Err(format!("ncnn extractor input failed: {}", ret));
        }
        Ok(())
    }

    pub fn extract(&self, name: &str) -> Result<NcnnMat, String> {
        let cname = CString::new(name).map_err(|e| format!("CString: {}", e))?;
        let mut out_mat: ncnn_mat_t = std::ptr::null_mut();
        let ret = unsafe { ncnn_extractor_extract(self.ex, cname.as_ptr(), &mut out_mat) };
        if ret != 0 {
            return Err(format!("ncnn extract failed: {}", ret));
        }
        if out_mat.is_null() {
            return Err("ncnn extract returned null".into());
        }
        Ok(NcnnMat { mat: out_mat })
    }
}

impl Drop for NcnnExtractor {
    fn drop(&mut self) {
        unsafe { ncnn_extractor_destroy(self.ex) }
    }
}

pub struct NcnnMat {
    mat: ncnn_mat_t,
}

impl NcnnMat {
    pub fn shape(&self) -> (i32, i32, i32) {
        let w = unsafe { ncnn_mat_get_w(self.mat) };
        let h = unsafe { ncnn_mat_get_h(self.mat) };
        let c = unsafe { ncnn_mat_get_c(self.mat) };
        (w, h, c)
    }

    pub fn data_f32(&self) -> &[f32] {
        let (w, h, c) = self.shape();
        let len = (w * h * c) as usize;
        let ptr = unsafe { ncnn_mat_get_data(self.mat) } as *const f32;
        if ptr.is_null() { return &[]; }
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl Drop for NcnnMat {
    fn drop(&mut self) {
        unsafe { ncnn_mat_destroy(self.mat) }
    }
}

// Link to ncnn library
#[link(name = "ncnn")]
extern "C" {}
