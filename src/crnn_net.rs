use float_ord::FloatOrd;
use image::{imageops::FilterType, DynamicImage};
use ndarray::{ArrayView2, Axis};
use ort::{inputs, GraphOptimizationLevel, Session};
use std::path::PathBuf;
use tracing::instrument;

use crate::{util::subtract_mean_normalize, TextLine};

const MEAN_VALUES: [f32; 3] = [0.5, 0.5, 0.5];
const NORM_VALUES: [f32; 3] = [2.0, 2.0, 2.0];

pub struct CrnnNet {
    session: Session,
    keys: Vec<String>,
}

impl CrnnNet {
    #[instrument(level = "debug")]
    pub fn init(model_path: PathBuf, keys_path: PathBuf, num_threads: usize) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_inter_threads(num_threads)?
            .with_intra_threads(num_threads)?
            .commit_from_file(model_path)?;

        let keys =
            std::fs::read_to_string(&keys_path).map_err(|_| ort::Error::FileDoesNotExist {
                filename: keys_path, // TODO: Better error handling
            })?;
        let keys = keys.lines().map(|line| line.to_string());
        let keys = ["#".to_string()]
            .into_iter()
            .chain(keys)
            .chain([" ".to_string()]);

        log::debug!("CRNN Inputs: {:?}", session.inputs);
        log::debug!("CRNN Outputs: {:?}", session.outputs);

        Ok(Self {
            session,
            keys: keys.collect(),
        })
    }

    #[instrument(level = "debug", skip(self, images))]
    pub fn get_text_lines(&self, images: &[DynamicImage]) -> ort::Result<Vec<TextLine>> {
        images
            .iter()
            .map(|image| self.get_text_line(image))
            .collect()
    }

    #[instrument(level = "trace", skip(self, image))]
    fn get_text_line(&self, image: &DynamicImage) -> ort::Result<TextLine> {
        let dest_height = 48;
        let scale = dest_height as f32 / image.height() as f32;
        let dest_width = (image.width() as f32 * scale) as u32;
        let image = image.resize(dest_width, dest_height, FilterType::Nearest);

        let tensor_values =
            subtract_mean_normalize(&image, &MEAN_VALUES, &NORM_VALUES).insert_axis(Axis(0));
        let outputs = self.session.run(inputs!["x" => tensor_values]?)?;
        let output_tensor = outputs
            .first_key_value()
            .unwrap()
            .1
            .try_extract_tensor::<f32>()?;

        log::trace!("Output tensor size: {:?}", output_tensor.dim());
        let width = output_tensor.len_of(Axis(1));

        let output_tensor = output_tensor.remove_axis(Axis(0));
        let output = output_tensor.to_shape((width, 6625)).unwrap();

        Ok(self.score_to_text_line(output.view()))
    }

    fn score_to_text_line(&self, data: ArrayView2<f32>) -> TextLine {
        let keys_size = self.keys.len();

        let max_scores = data
            .outer_iter()
            .map(|it| {
                let (i, value) = it
                    .indexed_iter()
                    .max_by_key(|(_, value)| FloatOrd(**value))
                    .unwrap();
                (i, *value)
            })
            .filter(|(i, _)| *i > 0 && *i < keys_size)
            .map(|(i, score)| (self.keys[i].as_str(), score))
            .collect::<Vec<_>>();

        let text = max_scores.iter().map(|(text, _)| *text).collect::<String>();
        let scores = max_scores
            .iter()
            .map(|(_, score)| *score)
            .collect::<Vec<_>>();

        TextLine {
            text,
            character_scores: scores,
        }
    }
}
