use std::path::PathBuf;

use angle_net::AngleNet;
use crnn_net::CrnnNet;
use dbnet::DbNet;

mod angle_net;
mod crnn_net;
pub mod dbnet;
mod result;
pub mod util;

use image::DynamicImage;
pub use result::*;
use tracing::instrument;
use util::{part_image, scale_normalized};

pub use ort as runtime;

pub struct RapidOCRBuilder {
    threads: usize,
    gpu_index: Option<u32>,
    det_path: Option<PathBuf>,
    cls_path: Option<PathBuf>,
    rec_paths: Option<(PathBuf, PathBuf)>,
    max_side_len: u32,
    most_angle: bool,
    cache_path: Option<PathBuf>,
    execution_providers: Vec<ExecutionProvider>,
}

impl RapidOCRBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn gpu_index(mut self, index: Option<u32>) -> Self {
        self.gpu_index = index;
        self
    }

    pub fn det_model(mut self, path: impl Into<PathBuf>) -> Self {
        self.det_path = Some(path.into());
        self
    }

    pub fn cls_model(mut self, path: impl Into<PathBuf>) -> Self {
        self.cls_path = Some(path.into());
        self
    }

    pub fn rec_model(
        mut self,
        model_path: impl Into<PathBuf>,
        keys_path: impl Into<PathBuf>,
    ) -> Self {
        self.rec_paths = Some((model_path.into(), keys_path.into()));
        self
    }

    pub fn most_angle(mut self, most_angle: bool) -> Self {
        self.most_angle = most_angle;
        self
    }

    pub fn max_side_len(mut self, max_side_len: u32) -> Self {
        self.max_side_len = max_side_len;
        self
    }

    pub fn with_engine_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_path = Some(path.into());
        self
    }

    pub fn with_execution_providers(
        mut self,
        providers: impl IntoIterator<Item = ExecutionProvider>,
    ) -> Self {
        self.execution_providers = providers.into_iter().collect();
        self
    }

    #[instrument(skip(self), level = "debug")]
    fn init_models(&mut self) -> ort::Result<(DbNet, Option<AngleNet>, CrnnNet)> {
        let det_path = self
            .det_path
            .take()
            .unwrap_or_else(|| "models/ch_PP-OCRv4_det_infer/ch_PP-OCRv4_det_infer.onnx".into());
        let cls_path = self.cls_path.take();
        let (rec_path, keys_path) = self.rec_paths.take().unwrap_or_else(|| {
            (
                "models/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.onnx".into(),
                "models/ppocr_keys_v1.txt".into(),
            )
        });
        Ok((
            DbNet::init(
                det_path,
                self.threads,
                self.max_side_len,
                &self.execution_providers,
                self.cache_path.clone(),
            )?,
            cls_path
                .map(|cls_path| AngleNet::init(cls_path, self.threads))
                .transpose()?,
            CrnnNet::init(
                rec_path,
                keys_path,
                self.threads,
                self.max_side_len,
                &self.execution_providers,
                self.cache_path.clone(),
            )?,
        ))
    }

    #[instrument(skip(self))]
    pub fn build(mut self) -> ort::Result<RapidOCR> {
        let (det_model, cls_model, rec_model) = self.init_models()?;
        Ok(RapidOCR {
            det_model,
            cls_model,
            rec_model,
            max_side_len: self.max_side_len,
            most_angle: self.most_angle,
        })
    }
}

impl Default for RapidOCRBuilder {
    fn default() -> Self {
        Self {
            threads: 4,
            gpu_index: None,
            det_path: None,
            cls_path: None,
            rec_paths: None,
            max_side_len: 1024,
            most_angle: false,
            cache_path: None,
            execution_providers: DEFAULT_PROVIDERS.to_vec(),
        }
    }
}

pub struct RapidOCR {
    det_model: DbNet,
    cls_model: Option<AngleNet>,
    rec_model: CrnnNet,
    max_side_len: u32,
    most_angle: bool,
}

impl RapidOCR {
    #[instrument(skip(self, image))]
    pub fn detect(
        &self,
        image: &DynamicImage,
        options: DetectionOptions,
    ) -> ort::Result<Vec<OcrResult>> {
        let DetectionOptions {
            max_side_len,
            box_threshold,
            box_score_threshold,
            unclip_ratio,
            ..
        } = options;
        let max_side_len = if max_side_len != 0 {
            max_side_len.min(max_side_len)
        } else {
            self.max_side_len
        };
        let scale = if max_side_len > 0 {
            scale_normalized(image, max_side_len)
        } else {
            scale_normalized(image, u32::MAX)
        };
        let boxes = self.det_model.get_text_boxes(
            image,
            scale,
            box_threshold,
            box_score_threshold,
            unclip_ratio,
        )?;
        let mut part_images = boxes
            .iter()
            .map(|it| part_image(image, &it.rect))
            .collect::<Vec<_>>();
        #[cfg(feature = "debug")]
        for (i, image) in part_images.iter().enumerate() {
            image.save(format!("part_images/{i}.png")).unwrap();
        }

        if let Some(angle_net) = &self.cls_model {
            let angles = angle_net.get_angles(&part_images, self.most_angle)?;
            for (image, angle) in part_images.iter_mut().zip(angles) {
                if angle.index == 1 {
                    *image = image.rotate180();
                }
            }
        }

        let text_lines = self.rec_model.get_text_lines(&part_images)?;

        Ok(boxes
            .into_iter()
            .zip(text_lines.into_iter())
            .map(|(bounds, text)| OcrResult { bounds, text })
            .collect())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    pub padding: u32,
    pub max_side_len: u32,
    pub box_score_threshold: f32,
    pub box_threshold: f32,
    pub unclip_ratio: f32,
    pub most_angle: bool,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            padding: 50,
            max_side_len: 0,
            box_score_threshold: 0.5,
            box_threshold: 0.3,
            unclip_ratio: 1.6,
            most_angle: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Default,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "directml")]
    DirectML,
}

const DEFAULT_PROVIDERS: &[ExecutionProvider] = &[
    #[cfg(feature = "tensorrt")]
    ExecutionProvider::TensorRT,
    #[cfg(feature = "coreml")]
    ExecutionProvider::CoreML,
    #[cfg(feature = "directml")]
    ExecutionProvider::DirectML,
    #[cfg(feature = "cuda")]
    ExecutionProvider::Cuda,
    ExecutionProvider::Default,
];
