use std::{path::PathBuf, time::Instant};

use rapidocr::{DetectionOptions, RapidOCRBuilder};
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

fn main() {
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let image = image::open("tests/data/test_image.png").expect("Failed  to load test image");
    let cache = PathBuf::from(".cache");
    std::fs::create_dir_all(&cache).expect("Failed to create temp dir");
    let ocr = RapidOCRBuilder::new()
        .det_model("tests/data/models/det.onnx")
        .cls_model("tests/data/models/cls.onnx")
        .rec_model(
            "tests/data/models/rec_easyocr.onnx",
            "tests/data/models/ch_sim_char.txt",
        )
        .with_engine_cache_path(cache)
        .max_side_len(2048)
        .build()
        .expect("Failed to build engine");
    let start = Instant::now();
    let text = ocr
        .detect(&image, DetectionOptions::default())
        .expect("Failed recognition.");
    let end = start.elapsed();
    log::debug!("{end:?}");
    assert!(text.len() > 1);
    log::debug!("{text:#?}");
    assert!(text
        .into_iter()
        .map(|it| it.text.text)
        .find(|it| it == "不行，头好痛-接下来要处理的事情太多了，现在必须好好休息·！")
        .is_some());
}
