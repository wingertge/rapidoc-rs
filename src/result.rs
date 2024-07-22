use geo::Polygon;

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub bounds: TextBox,
    pub text: TextLine,
}

#[derive(Debug, Clone)]
pub struct TextBox {
    pub score: f32,
    pub rect: Polygon<f32>,
}

#[derive(Debug, Clone)]
pub struct TextLine {
    pub text: String,
    pub character_scores: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Angle {
    pub index: usize,
    pub score: f32,
}
