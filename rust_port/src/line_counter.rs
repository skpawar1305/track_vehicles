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
