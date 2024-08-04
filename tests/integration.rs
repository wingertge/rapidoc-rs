use std::{path::PathBuf, time::Instant};

use rapidocr::{DetectionOptions, RapidOCRBuilder};

// This doesn't work when the execution provider can't be loaded because it silently fails and you
// can't disable fallback, so that sucks
#[test]
fn execution_provider_doesnt_crash() {
    let _ = env_logger::builder().is_test(true).try_init();

    let image = image::open("tests/data/test_image.png").expect("Failed  to load test image");
    let cache = std::env!("CARGO_TARGET_TMPDIR");
    let cache = PathBuf::from(cache).join(".engine_cache");
    std::fs::create_dir_all(&cache).expect("Failed to create temp dir");
    let ocr = RapidOCRBuilder::new()
        .det_model("tests/data/models/det.onnx")
        .cls_model("tests/data/models/cls.onnx")
        .rec_model(
            "tests/data/models/rec.onnx",
            "tests/data/models/ppocr_keys_v1.txt",
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
    assert!(text
        .into_iter()
        .map(|it| it.text.text)
        .find(|it| it == "不行，头好痛-接下来要处理的事情太多了，现在必须好好休息·！")
        .is_some());
}
