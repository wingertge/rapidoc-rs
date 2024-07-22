use std::path::PathBuf;

use float_ord::FloatOrd;
use image::DynamicImage;
use ndarray::Axis;
use ort::{inputs, GraphOptimizationLevel, Session};
use tracing::instrument;

use crate::{util::subtract_mean_normalize, Angle};

const DEST_WIDTH: u32 = 192;
const DEST_HEIGHT: u32 = 48;

const MEAN_VALUES: [f32; 3] = [0.5, 0.5, 0.5];
const NORM_VALUES: [f32; 3] = [2.0, 2.0, 2.0];

pub struct AngleNet {
    session: Session,
}

impl AngleNet {
    #[instrument(level = "debug")]
    pub fn init(path: PathBuf, num_threads: usize) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_inter_threads(num_threads)?
            .with_intra_threads(num_threads)?
            .commit_from_file(path)?;

        log::debug!("Angle session inputs: {:?}", session.inputs);
        log::debug!("Angle session outputs: {:?}", session.outputs);

        Ok(Self { session })
    }

    #[instrument(level = "debug", skip(self, images))]
    pub fn get_angles(&self, images: &[DynamicImage], most_angle: bool) -> ort::Result<Vec<Angle>> {
        let mut angles = images
            .iter()
            .map(|image| self.get_angle(image))
            .collect::<ort::Result<Vec<_>>>()?;

        if most_angle {
            let sum = angles.iter().map(|angle| angle.index).sum::<usize>() as f32;
            let half_percent = angles.len() as f32 / 2.0;
            let most_angle_index = if sum < half_percent { 0 } else { 1 };

            for angle in angles.iter_mut() {
                angle.index = most_angle_index;
            }
        }

        Ok(angles)
    }

    #[instrument(level = "trace", skip(self, image))]
    fn get_angle(&self, image: &DynamicImage) -> ort::Result<Angle> {
        let image = image.resize_exact(
            DEST_WIDTH,
            DEST_HEIGHT,
            image::imageops::FilterType::Nearest,
        );
        let image =
            subtract_mean_normalize(&image, &MEAN_VALUES, &NORM_VALUES).insert_axis(Axis(0));
        let outputs = self.session.run(inputs!["x" => image]?)?;
        let output = outputs
            .first_key_value()
            .unwrap()
            .1
            .try_extract_tensor::<f32>()?
            .remove_axis(Axis(0));

        let angle = output
            .iter()
            .enumerate()
            .max_by_key(|(_, score)| FloatOrd(**score))
            .map(|(index, score)| Angle {
                index,
                score: *score,
            })
            .unwrap();

        Ok(angle)
    }
}
