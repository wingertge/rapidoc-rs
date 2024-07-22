use std::env;

use geo::BoundingRect;
use image::Rgba;
use imageproc::{drawing::draw_hollow_polygon_mut, point::Point};
use rapidocr::{DetectionOptions, OcrResult, RapidOCRBuilder};
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

pub fn main() {
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let onnx_path = env::current_dir()
        .unwrap()
        .join("lib/onnxruntime.dll")
        .to_str()
        .unwrap()
        .to_owned();
    ort::init_from(onnx_path)
        //.with_execution_providers([DirectMLExecutionProvider::default().build()])
        .commit()
        .unwrap();

    let mut image = image::open("Sample_Screenshot.png").unwrap();

    let ocr = RapidOCRBuilder::new().build().unwrap();
    let results = ocr
        .detect(
            &image,
            DetectionOptions {
                max_side_len: 1024,
                ..DetectionOptions::default()
            },
        )
        .unwrap();

    for result in &results {
        println!(
            "[\"{}\", {:?}",
            result.text.text,
            result.bounds.rect.bounding_rect()
        );
    }

    for OcrResult { bounds: b_box, .. } in &results {
        let points = b_box
            .rect
            .exterior()
            .points()
            .take(4)
            .map(|p| Point { x: p.x(), y: p.y() })
            .collect::<Vec<_>>();

        draw_hollow_polygon_mut(&mut image, &points, Rgba([0u8, 0u8, 255u8, 255u8]));
    }

    image.save("tmp3.png").unwrap();
}
