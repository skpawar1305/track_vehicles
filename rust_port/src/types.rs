#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [i32; 4],
    pub centroid: (i32, i32),
    pub confidence: f32,
    pub class_id: i32,
    pub label: String,
}
