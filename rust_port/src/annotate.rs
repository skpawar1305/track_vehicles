use opencv::{
    core,
    imgproc,
    prelude::*,
};

const GREEN: core::Scalar = core::Scalar::new(0.0, 255.0, 0.0, 0.0);
const BLUE: core::Scalar = core::Scalar::new(255.0, 0.0, 0.0, 0.0);
const RED: core::Scalar = core::Scalar::new(0.0, 0.0, 255.0, 0.0);
const YELLOW: core::Scalar = core::Scalar::new(0.0, 255.0, 255.0, 0.0);
const GRAY: core::Scalar = core::Scalar::new(128.0, 128.0, 128.0, 0.0);
const WHITE: core::Scalar = core::Scalar::new(255.0, 255.0, 255.0, 0.0);

pub fn draw_line(frame: &mut core::Mat, line: &[i32; 4], flip: bool) {
    let (w, h) = (frame.cols(), frame.rows());

    let x1 = line[0] as f64;
    let y1 = line[1] as f64;
    let x2 = line[2] as f64;
    let y2 = line[3] as f64;

    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 0.0 {
        return;
    }

    let ux = dx / len;
    let uy = dy / len;
    let px = -uy * 30.0;
    let py = ux * 30.0;
    let zone_ext = 40.0;
    let zw = 30.0;

    let pt1 = core::Point::new(
        ((x1 - ux * zone_ext - px * zw / 30.0) as i32).clamp(0, w - 1),
        ((y1 - uy * zone_ext - py * zw / 30.0) as i32).clamp(0, h - 1),
    );
    let pt2 = core::Point::new(
        ((x1 - ux * zone_ext + px * zw / 30.0) as i32).clamp(0, w - 1),
        ((y1 - uy * zone_ext + py * zw / 30.0) as i32).clamp(0, h - 1),
    );
    let pt3 = core::Point::new(
        ((x2 + ux * zone_ext + px * zw / 30.0) as i32).clamp(0, w - 1),
        ((y2 + uy * zone_ext + py * zw / 30.0) as i32).clamp(0, h - 1),
    );
    let pt4 = core::Point::new(
        ((x2 + ux * zone_ext - px * zw / 30.0) as i32).clamp(0, w - 1),
        ((y2 + uy * zone_ext - py * zw / 30.0) as i32).clamp(0, h - 1),
    );

    imgproc::line(frame, pt1, pt2, GRAY, 1, imgproc::LINE_8, 0).ok();
    imgproc::line(frame, pt2, pt3, GRAY, 1, imgproc::LINE_8, 0).ok();
    imgproc::line(frame, pt3, pt4, GRAY, 1, imgproc::LINE_8, 0).ok();
    imgproc::line(frame, pt4, pt1, GRAY, 1, imgproc::LINE_8, 0).ok();

    let l1 = core::Point::new(x1 as i32, y1 as i32);
    let l2 = core::Point::new(x2 as i32, y2 as i32);
    imgproc::line(frame, l1, l2, BLUE, 3, imgproc::LINE_8, 0).ok();

    let mx = ((x1 + x2) / 2.0) as i32;
    let my = ((y1 + y2) / 2.0) as i32;
    let label_x = (mx + px as i32).clamp(0, w - 1);
    let label_y = (my + py as i32).clamp(0, h - 1);
    let label2_x = (mx - px as i32).clamp(0, w - 1);
    let label2_y = (my - py as i32).clamp(0, h - 1);
    let scale = 0.5;

    if flip {
        imgproc::put_text(frame, "IN", core::Point::new(label2_x, label2_y), imgproc::FONT_HERSHEY_SIMPLEX, scale, GREEN, 2, imgproc::LINE_8, false).ok();
        imgproc::put_text(frame, "OUT", core::Point::new(label_x, label_y), imgproc::FONT_HERSHEY_SIMPLEX, scale, RED, 2, imgproc::LINE_8, false).ok();
    } else {
        imgproc::put_text(frame, "IN", core::Point::new(label_x, label_y), imgproc::FONT_HERSHEY_SIMPLEX, scale, GREEN, 2, imgproc::LINE_8, false).ok();
        imgproc::put_text(frame, "OUT", core::Point::new(label2_x, label2_y), imgproc::FONT_HERSHEY_SIMPLEX, scale, RED, 2, imgproc::LINE_8, false).ok();
    }
}

pub fn draw_boxes(frame: &mut core::Mat, objects: &[(u32, &[i32; 4], &str)]) {
    for (_, bbox, label) in objects {
        let x1 = bbox[0].max(0);
        let y1 = bbox[1].max(0);
        let x2 = bbox[2].min(frame.cols() - 1);
        let y2 = bbox[3].min(frame.rows() - 1);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        let pt1 = core::Point::new(x1, y1);
        let pt2 = core::Point::new(x2, y2);
        imgproc::rectangle_points(frame, pt1, pt2, GREEN, 2, imgproc::LINE_8, 0).ok();

        let text = format!("#{}", label);
        if y1 >= 12 {
            imgproc::put_text(frame, &text, core::Point::new(x1 + 2, y1 - 4), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, imgproc::LINE_8, false).ok();
        }

        let cx = (x1 + x2) / 2;
        let cy = (y1 + y2) / 2;
        imgproc::circle(frame, core::Point::new(cx, cy), 3, YELLOW, -1, imgproc::LINE_8, 0).ok();
    }
}

pub fn draw_counts(frame: &mut core::Mat, in_count: usize, out_count: usize, fps: f32) {
    let w = frame.cols();
    let h = frame.rows();
    let scale = 0.5;

    if w >= 150 {
        let in_text = format!("IN: {}", in_count);
        let out_text = format!("OUT: {}", out_count);
        imgproc::put_text(frame, &in_text, core::Point::new(w - 140, 20), imgproc::FONT_HERSHEY_SIMPLEX, scale, GREEN, 2, imgproc::LINE_8, false).ok();
        imgproc::put_text(frame, &out_text, core::Point::new(w - 150, 42), imgproc::FONT_HERSHEY_SIMPLEX, scale, RED, 2, imgproc::LINE_8, false).ok();
    }

    let fps_text = format!("{:.0} FPS", fps);
    imgproc::put_text(frame, &fps_text, core::Point::new(6, h - 8), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, imgproc::LINE_8, false).ok();
}
