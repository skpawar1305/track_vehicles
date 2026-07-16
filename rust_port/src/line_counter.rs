#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Crossing {
    None,
    In,
    Out,
}

pub fn cross_product(ax: i32, ay: i32, bx: i32, by: i32, px: i32, py: i32) -> i64 {
    (bx - ax) as i64 * (py - ay) as i64 - (by - ay) as i64 * (px - ax) as i64
}

pub fn which_side(line: &[i32; 4], point: (i32, i32)) -> i8 {
    let cp = cross_product(line[0], line[1], line[2], line[3], point.0, point.1);
    if cp >= 0 { 1 } else { -1 }
}

/// Distance from a point to a line segment (in pixels)
pub fn point_line_distance(line: &[i32; 4], point: (i32, i32)) -> f64 {
    let ax = line[0] as f64;
    let ay = line[1] as f64;
    let bx = line[2] as f64;
    let by = line[3] as f64;
    let px = point.0 as f64;
    let py = point.1 as f64;
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let near_x = ax + t * dx;
    let near_y = ay + t * dy;
    ((px - near_x).powi(2) + (py - near_y).powi(2)).sqrt()
}

pub fn detect_crossing(
    line: &[i32; 4],
    old_centroid: (i32, i32),
    new_centroid: (i32, i32),
    flip: bool,
) -> Crossing {
    let old_side = which_side(line, old_centroid);
    let new_side = which_side(line, new_centroid);
    if old_side != new_side {
        if flip {
            return if new_side == 1 { Crossing::Out } else { Crossing::In };
        } else {
            return if new_side == 1 { Crossing::In } else { Crossing::Out };
        }
    }
    Crossing::None
}

/// Check if a bounding box overlaps the line's bounding rectangle in both axes.
pub fn bbox_touches_line(line: &[i32; 4], bbox: &[i32; 4]) -> bool {
    let lx1 = line[0].min(line[2]);
    let lx2 = line[0].max(line[2]);
    let ly1 = line[1].min(line[3]);
    let ly2 = line[1].max(line[3]);
    let bx1 = bbox[0];
    let bx2 = bbox[2];
    let by1 = bbox[1];
    let by2 = bbox[3];
    // Allow a 20px margin vertically so smaller detections still register
    bx1 <= lx2 && bx2 >= lx1 && by1 <= ly2 + 20 && by2 >= ly1 - 20
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_product() {
        // Line from (0,0) to (100,0), point (50, 10) below the line
        let cp = cross_product(0, 0, 100, 0, 50, 10);
        assert!(cp > 0);
    }

    #[test]
    fn test_which_side() {
        let line = [0, 180, 640, 180]; // horizontal line at y=180
        assert_eq!(which_side(&line, (100, 0)), -1);   // above (smaller y)
        assert_eq!(which_side(&line, (100, 360)), 1);  // below (larger y)
    }

    #[test]
    fn test_detect_crossing_none() {
        let line = [0, 180, 640, 180];
        assert_eq!(detect_crossing(&line, (100, 0), (100, 50), false), Crossing::None);
    }

    #[test]
    fn test_detect_crossing_in() {
        let line = [0, 180, 640, 180];
        // above (-1) -> below (1) = IN (new_side=1 -> In)
        assert_eq!(detect_crossing(&line, (100, 0), (100, 360), false), Crossing::In);
    }

    #[test]
    fn test_detect_crossing_out() {
        let line = [0, 180, 640, 180];
        // below (1) -> above (-1) = OUT (new_side=-1 -> Out)
        assert_eq!(detect_crossing(&line, (100, 360), (100, 0), false), Crossing::Out);
    }

    #[test]
    fn test_detect_crossing_flip() {
        let line = [0, 180, 640, 180];
        // flip swaps IN/OUT
        assert_eq!(detect_crossing(&line, (100, 360), (100, 0), true), Crossing::In);
        assert_eq!(detect_crossing(&line, (100, 0), (100, 360), true), Crossing::Out);
    }
}
