use std::path::PathBuf;

use geo::{Coord, MinimumRotatedRect, Scale};
use image::{imageops::FilterType, DynamicImage, GrayImage};
use imageproc::{
    contours::find_contours,
    contrast::{threshold_mut, ThresholdType},
    distance_transform::Norm,
    morphology::dilate_mut,
};
use ndarray::{ArrayView2, Axis};
use ort::{inputs, GraphOptimizationLevel, Session};
use tracing::instrument;

use crate::{
    util::{
        self, box_score_fast, max_side, subtract_mean_normalize, to_geo_poly, to_luma_image, unclip,
    },
    TextBox,
};

const MEAN_VALUES: [f32; 3] = [0.485, 0.456, 0.406];
const NORM_VALUES: [f32; 3] = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225];

pub struct DbNet {
    session: Session,
}

impl DbNet {
    #[instrument(level = "debug")]
    pub fn init(path: PathBuf, num_threads: usize) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_inter_threads(num_threads)?
            .with_intra_threads(num_threads)?
            .commit_from_file(path)?;

        Ok(Self { session })
    }

    #[instrument(skip(self, image), level = "debug")]
    pub fn get_text_boxes(
        &self,
        image: &DynamicImage,
        scale: util::Scale,
        box_score_thresh: f32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> ort::Result<Vec<TextBox>> {
        let image =
            image.resize_exact(scale.target_width, scale.target_height, FilterType::Nearest);
        let input_values =
            subtract_mean_normalize(&image, &MEAN_VALUES, &NORM_VALUES).insert_axis(Axis(0));
        //let input_values = Array4::<f32>::zeros((1, 3, 256, 256));
        let inputs = inputs!["x" => input_values]?;
        let outputs = self.session.run(inputs)?;
        let pred_mat = outputs
            .first_key_value()
            .unwrap()
            .1
            .try_extract_tensor::<f32>()?;

        let width = pred_mat.len_of(Axis(3));
        let height = pred_mat.len_of(Axis(2));

        let pred_data = pred_mat
            .to_owned()
            .remove_axis(Axis(0))
            .remove_axis(Axis(0));
        let pred_data = pred_data.to_shape((height, width)).unwrap();

        let mut image = to_luma_image(pred_data.view());

        let threshold = (box_thresh * 255.0) as u8;
        threshold_mut(&mut image, threshold, ThresholdType::Binary);
        dilate_mut(&mut image, Norm::L1, 2);

        Ok(find_rs_boxes(
            pred_data.view(),
            image,
            scale,
            box_score_thresh,
            unclip_ratio,
        ))
    }
}

#[instrument(skip(pred_data, image), level = "trace")]
fn find_rs_boxes(
    pred_data: ArrayView2<f32>,
    image: GrayImage,
    util::Scale {
        factor_x, factor_y, ..
    }: util::Scale,
    box_score_threshold: f32,
    unclip_ratio: f32,
) -> Vec<TextBox> {
    let long_side_threshold = 3.0;
    let max_candidates = 1000;

    let contours = find_contours::<i32>(&image)
        .into_iter()
        .take(max_candidates)
        .filter(|it| it.points.len() > 2)
        .map(|it| to_geo_poly(&it.points).minimum_rotated_rect().unwrap())
        .map(|rect| {
            let side = max_side(&rect);
            (rect, side)
        })
        .filter(|(_, side)| *side >= long_side_threshold)
        .map(|(rect, side)| {
            let score = box_score_fast(&rect, pred_data.view());
            (rect, side, score)
        })
        .filter(|(_, _, rect)| *rect >= box_score_threshold)
        .filter_map(|(rect, side, score)| Some((unclip(rect, unclip_ratio)?, side, score)))
        .filter(|(clip_rect, _, _)| max_side(clip_rect) >= long_side_threshold + 2.0)
        .map(|(rect, _, score)| TextBox {
            score,
            rect: rect.scale_around_point(factor_y, factor_x, Coord::zero()),
        })
        .collect();
    contours
}
