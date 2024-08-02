use float_ord::FloatOrd;
use image::{imageops::FilterType, DynamicImage};
use ndarray::{ArrayView2, Axis};
use ort::ExecutionProviderDispatch;
use ort::{inputs, GraphOptimizationLevel, Session};
use std::path::PathBuf;
use tracing::instrument;

use crate::{util::subtract_mean_normalize, ExecutionProvider, TextLine};

const MEAN_VALUES: [f32; 3] = [0.5, 0.5, 0.5];
const NORM_VALUES: [f32; 3] = [2.0, 2.0, 2.0];

pub struct CrnnNet {
    session: Session,
    keys: Vec<String>,
}

#[cfg(feature = "tensorrt")]
fn setup_tensorrt(cache_path: PathBuf) -> ExecutionProviderDispatch {
    use ort::TensorRTExecutionProvider;

    TensorRTExecutionProvider::default()
        .with_profile_min_shapes("x:1x3x48x1")
        .with_profile_max_shapes(format!("x:1x3x48x{}", u16::MAX))
        .with_profile_opt_shapes("x:1x3x48x256")
        .with_engine_cache(true)
        .with_engine_cache_path(cache_path.to_string_lossy())
        .with_timing_cache(true)
        .with_builder_optimization_level(5)
        .with_detailed_build_log(true)
        .build()
}

#[cfg(feature = "cuda")]
fn setup_cuda() -> ExecutionProviderDispatch {
    use ort::CUDAExecutionProvider;

    CUDAExecutionProvider::default().build()
}

#[cfg(feature = "directml")]
fn setup_directml() -> ExecutionProviderDispatch {
    use ort::DirectMLExecutionProvider;

    DirectMLExecutionProvider::default().build()
}

impl CrnnNet {
    #[instrument(level = "debug")]
    pub fn init(
        model_path: PathBuf,
        keys_path: PathBuf,
        num_threads: usize,
        max_side_len: u32,
        execution_providers: &[ExecutionProvider],
        cache_path: Option<PathBuf>,
    ) -> ort::Result<Self> {
        #[cfg(feature = "directml")]
        let parallel = execution_providers.contains(&ExecutionProvider::DirectML);
        #[cfg(not(feature = "directml"))]
        let parallel = true;

        let execution_providers = execution_providers.iter().filter_map(
            |provider| -> Option<ExecutionProviderDispatch> {
                match provider {
                    ExecutionProvider::Default => None,
                    #[cfg(feature = "tensorrt")]
                    ExecutionProvider::TensorRT => {
                        Some(setup_tensorrt(cache_path.clone().unwrap_or_else(|| {
                            model_path.parent().unwrap().join(".cache")
                        })))
                    }
                    #[cfg(feature = "cuda")]
                    ExecutionProvider::Cuda => Some(setup_cuda()),
                    #[cfg(feature = "directml")]
                    ExecutionProvider::DirectML => Some(setup_directml()),
                }
            },
        );

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(parallel)?
            .with_inter_threads(num_threads)?
            .with_intra_threads(num_threads)?
            .with_execution_providers(execution_providers)?
            /*             .with_optimized_model_path(
                cache_path
                    .unwrap_or(model_path.parent().unwrap().join(".cache"))
                    .to_string_lossy(),
            )? */
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
        let dest_width = dest_width.min(u16::MAX as u32);
        let image = image.resize_exact(dest_width, dest_height, FilterType::Nearest);

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

    #[instrument(level = "trace", skip(self, data))]
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
