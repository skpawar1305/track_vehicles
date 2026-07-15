use image::{RgbImage, Rgb};
use imageproc::drawing::draw_text_mut;
use ab_glyph::{FontRef, PxScale};

const FONT_DATA: &[u8] = include_bytes!("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

pub fn draw_line(frame: &mut RgbImage, line: &[i32; 4], flip: bool) {
    let x1 = line[0] as f64;
    let y1 = line[1] as f64;
    let x2 = line[2] as f64;
    let y2 = line[3] as f64;

    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    if len > 0.0 {
        let ux = dx / len;
        let uy = dy / len;
        let px = -uy * 30.0;
        let py = ux * 30.0;
        let zone_ext = 40.0;

        // Zone parallelogram
        let zw = 30.0;
        let p1x = ((x1 - ux * zone_ext - px * zw / 30.0) as i32).max(0) as u32;
        let p1y = ((y1 - uy * zone_ext - py * zw / 30.0) as i32).max(0) as u32;
        let p2x = ((x1 - ux * zone_ext + px * zw / 30.0) as i32).max(0) as u32;
        let p2y = ((y1 - uy * zone_ext + py * zw / 30.0) as i32).max(0) as u32;
        let p3x = ((x2 + ux * zone_ext + px * zw / 30.0) as i32).max(0) as u32;
        let p3y = ((y2 + uy * zone_ext + py * zw / 30.0) as i32).max(0) as u32;
        let p4x = ((x2 + ux * zone_ext - px * zw / 30.0) as i32).max(0) as u32;
        let p4y = ((y2 + uy * zone_ext - py * zw / 30.0) as i32).max(0) as u32;

        let c = Rgb([136u8, 136, 136]);
        draw_thick_line(frame, p1x, p1y, p2x, p2y, c, 1);
        draw_thick_line(frame, p2x, p2y, p3x, p3y, c, 1);
        draw_thick_line(frame, p3x, p3y, p4x, p4y, c, 1);
        draw_thick_line(frame, p4x, p4y, p1x, p1y, c, 1);

        // Counting line
        draw_thick_line(frame, x1 as u32, y1 as u32, x2 as u32, y2 as u32, Rgb([59u8, 130, 246]), 3);

        // IN/OUT labels with proper font
        let mx = (x1 + x2) / 2.0;
        let my = (y1 + y2) / 2.0;
        let label_x = (mx + px).max(0.0) as i32;
        let label_y = (my + py).max(0.0) as i32;
        let label2_x = (mx - px).max(0.0) as i32;
        let label2_y = (my - py).max(0.0) as i32;

        let font = FontRef::try_from_slice(FONT_DATA).unwrap();
        let scale = PxScale { x: 14.0, y: 14.0 };
        let bg = Rgb([30u8, 30, 30]);

        if flip {
            draw_text_with_bg(frame, &font, scale, bg, Rgb([34u8, 197, 94]), label2_x, label2_y - 4, "IN");
            draw_text_with_bg(frame, &font, scale, bg, Rgb([239u8, 68, 68]), label_x, label_y - 4, "OUT");
        } else {
            draw_text_with_bg(frame, &font, scale, bg, Rgb([34u8, 197, 94]), label_x, label_y - 4, "IN");
            draw_text_with_bg(frame, &font, scale, bg, Rgb([239u8, 68, 68]), label2_x, label2_y - 4, "OUT");
        }
    } else {
        draw_thick_line(frame, x1 as u32, y1 as u32, x2 as u32, y2 as u32, Rgb([59u8, 130, 246]), 3);
    }
}

pub fn draw_boxes(frame: &mut RgbImage, objects: &[(u32, &[i32; 4], &str)]) {
    for (_, bbox, label) in objects {
        let x1 = bbox[0].max(0) as u32;
        let y1 = bbox[1].max(0) as u32;
        let x2 = bbox[2].max(0) as u32;
        let y2 = bbox[3].max(0) as u32;
        draw_rect(frame, x1, y1, x2, y2, Rgb([34, 197, 94]), 2);
        if y1 >= 12 {
            let font = FontRef::try_from_slice(FONT_DATA).unwrap();
            let scale = PxScale { x: 12.0, y: 12.0 };
            let txt = format!("#{}", label);
            draw_text_with_bg(frame, &font, scale, Rgb([30u8, 30, 30]), Rgb([34u8, 197, 94]),
                (x1 + 2) as i32, y1 as i32 - 14, &txt);
        }
        let cx = (x1 + x2) / 2;
        let cy = (y1 + y2) / 2;
        draw_filled_circle(frame, cx, cy, 3, Rgb([251, 191, 36]));
    }
}

pub fn draw_counts(frame: &mut RgbImage, in_count: usize, out_count: usize, fps: f32) {
    let w = frame.width() as i32;
    let h = frame.height() as i32;
    let font = FontRef::try_from_slice(FONT_DATA).unwrap();

    if w >= 150 {
        let in_text = format!("IN: {}", in_count);
        let out_text = format!("OUT: {}", out_count);
        draw_text_with_bg(frame, &font, PxScale { x: 16.0, y: 16.0 }, Rgb([30u8, 30, 30]), Rgb([34u8, 197, 94]),
            w - 140, 8, &in_text);
        draw_text_with_bg(frame, &font, PxScale { x: 16.0, y: 16.0 }, Rgb([30u8, 30, 30]), Rgb([239u8, 68, 68]),
            w - 150, 30, &out_text);
    }

    let fps_text = format!("{:.0} FPS", fps);
    draw_text_with_bg(frame, &font, PxScale { x: 12.0, y: 12.0 }, Rgb([30u8, 30, 30]), Rgb([136u8, 136, 136]),
        6, h - 18, &fps_text);
}

fn draw_text_with_bg(frame: &mut RgbImage, font: &FontRef, scale: PxScale, bg: Rgb<u8>, fg: Rgb<u8>,
                     x: i32, y: i32, text: &str) {
    let w = frame.width() as i32;
    let h = frame.height() as i32;

    // Draw dark pill background
    let pad = 3i32;
    let text_w = text.len() as i32 * 9 + 4;  // rough estimate
    let text_h = (scale.y as i32) + pad * 2;
    let bx = (x - pad).max(0);
    let by = (y - pad).max(0);
    let bw = (text_w + pad * 2).min(w - bx);
    let bh = text_h.min(h - by);

    for py in by..(by + bh).min(h) {
        for px in bx..(bx + bw).min(w) {
            let p = frame.get_pixel(px as u32, py as u32);
            let blended = Rgb([
                ((bg[0] as u16 * 6 + p[0] as u16 * 4) / 10) as u8,
                ((bg[1] as u16 * 6 + p[1] as u16 * 4) / 10) as u8,
                ((bg[2] as u16 * 6 + p[2] as u16 * 4) / 10) as u8,
            ]);
            frame.put_pixel(px as u32, py as u32, blended);
        }
    }

    // Draw text
    draw_text_mut(frame, fg, x, y, scale, font, text);
}

fn draw_rect(frame: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>, thickness: u32) {
    for t in 0..thickness {
        for x in x1..=x2 {
            if y1 + t < frame.height() { frame.put_pixel(x, y1 + t, color); }
            if y2 >= t && y2 - t < frame.height() { frame.put_pixel(x, y2 - t, color); }
        }
        for y in y1..=y2 {
            if x1 + t < frame.width() { frame.put_pixel(x1 + t, y, color); }
            if x2 >= t && x2 - t < frame.width() { frame.put_pixel(x2 - t, y, color); }
        }
    }
}

fn draw_thick_line(frame: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>, thickness: u32) {
    let dx = (x2 as i32 - x1 as i32).abs();
    let dy = -(y2 as i32 - y1 as i32).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x1 as i32;
    let mut cy = y1 as i32;

    loop {
        for t in 0..thickness {
            for u in 0..thickness {
                let px = (cx + t as i32 - thickness as i32 / 2).max(0) as u32;
                let py = (cy + u as i32 - thickness as i32 / 2).max(0) as u32;
                if px < frame.width() && py < frame.height() {
                    frame.put_pixel(px, py, color);
                }
            }
        }
        if cx == x2 as i32 && cy == y2 as i32 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; cx += sx; }
        if e2 <= dx { err += dx; cy += sy; }
    }
}

fn draw_filled_circle(frame: &mut RgbImage, cx: u32, cy: u32, r: u32, color: Rgb<u8>) {
    for dy in 0..=r {
        for dx in 0..=r {
            if dx * dx + dy * dy <= r * r {
                let px = (cx as i32 + dx as i32).max(0) as u32;
                let py = (cy as i32 + dy as i32).max(0) as u32;
                let nx = (cx as i32 - dx as i32).max(0) as u32;
                let ny = (cy as i32 - dy as i32).max(0) as u32;
                if px < frame.width() && py < frame.height() { frame.put_pixel(px, py, color); }
                if nx < frame.width() && py < frame.height() { frame.put_pixel(nx, py, color); }
                if px < frame.width() && ny < frame.height() { frame.put_pixel(px, ny, color); }
                if nx < frame.width() && ny < frame.height() { frame.put_pixel(nx, ny, color); }
            }
        }
    }
}
