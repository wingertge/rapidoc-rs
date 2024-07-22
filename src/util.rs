use geo::{
    point, Area, BoundingRect, Contains, Coord, EuclideanLength, HasDimensions, LineString,
    MinimumRotatedRect, Polygon, Translate,
};
use geo_clipper::{Clipper, EndType, JoinType};
use image::{DynamicImage, GrayImage, ImageBuffer, Pixel, Rgb};
use imageproc::point::Point;
use ndarray::{s, Array3, ArrayView2, Axis};
use tracing::instrument;

#[instrument(level = "debug", skip(image))]
pub(crate) fn subtract_mean_normalize(
    image: &DynamicImage,
    mean_vals: &[f32; 3],
    norm_vals: &[f32; 3],
) -> Array3<f32> {
    let mut image = image.to_rgb32f();
    let norm = Rgb::<f32>(*norm_vals);
    let mean_vals = Rgb::<f32>(*mean_vals).map2(&norm, |c1, c2| c1 * c2);
    for pixel in image.pixels_mut() {
        *pixel = pixel
            .map2(&norm, |c1, c2| c1 * c2)
            .map2(&mean_vals, |c1, c2| c1 - c2);
    }
    /*     let SampleLayout {
        channels,
        channel_stride,
        height,
        height_stride,
        width,
        width_stride,
    } = image.sample_layout();
    let shape = (channels as usize, height as usize, width as usize);
    let strides = (channel_stride, height_stride, width_stride);
    Array3::from_shape_vec(shape.strides(strides), image.into_raw()).unwrap() */
    Array3::<f32>::from_shape_fn(
        (3, image.height() as usize, image.width() as usize),
        |(ch, y, x)| {
            let pixel = image.get_pixel(x as u32, y as u32).channels()[ch] as f32;
            pixel
        },
    )
}

pub(crate) fn to_luma_image(data: ArrayView2<f32>) -> GrayImage {
    let height = data.len_of(Axis(0));
    let width = data.len_of(Axis(1));
    let pixel_data = data
        .axis_iter(Axis(0))
        .flat_map(|it| it.into_iter())
        .map(|p| (p * 255.0) as u8)
        .collect::<Vec<u8>>();
    ImageBuffer::from_raw(width as u32, height as u32, pixel_data).unwrap()
}

pub(crate) fn to_geo_poly(points: &[Point<i32>]) -> Polygon<f32> {
    let points = points
        .iter()
        .map(|point| Coord {
            x: point.x as f32,
            y: point.y as f32,
        })
        .collect();
    Polygon::new(LineString::new(points), vec![])
}

pub(crate) fn max_side(rect: &Polygon<f32>) -> f32 {
    rect.exterior()
        .lines()
        .map(|it| it.euclidean_length() as i32)
        .max()
        .unwrap() as f32
}

pub(crate) fn box_score_fast(rect: &Polygon<f32>, pred_data: ArrayView2<f32>) -> f32 {
    let bounds = rect.bounding_rect().unwrap();
    let min = bounds.min();
    let max = bounds.max();

    let sliced = pred_data.slice(s![
        min.y as usize..max.y as usize,
        min.x as usize..max.x as usize
    ]);

    let local_rect = rect.translate(-min.x, -min.y);

    let contained_values = sliced
        .indexed_iter()
        .filter(|((y, x), _)| local_rect.contains(&point![x: *x as f32, y: *y as f32]))
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();

    let len = contained_values.len() as f32;
    contained_values.into_iter().sum::<f32>() / len //mean
}

pub(crate) fn unclip(rect: Polygon<f32>, unclip_ratio: f32) -> Option<Polygon<f32>> {
    let distance = (rect.unsigned_area() * 0.5 * unclip_ratio) / rect.exterior().euclidean_length(); // Try 0.65 if problem

    let clipped_rect = rect.offset(distance, JoinType::Round(0.25), EndType::ClosedPolygon, 1.0);

    if clipped_rect.is_empty() {
        None
    } else {
        clipped_rect.minimum_rotated_rect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Scale {
    pub factor_x: f32,
    pub factor_y: f32,
    pub target_width: u32,
    pub target_height: u32,
}

pub fn scale_normalized(image: &DynamicImage, target_size: u32) -> Scale {
    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let (mut target_width, mut target_height) = if aspect_ratio >= 1.0 {
        let width = image.width().min(target_size);
        let height = (width as f32 / aspect_ratio) as u32;
        (width, height)
    } else {
        let height = image.height().min(target_size);
        let width = (height as f32 * aspect_ratio) as u32;
        (width, height)
    };
    if target_width % 32 != 0 {
        let new_width = (target_width / 32 * 32).max(32);
        log::debug!(
            "Target width of {target_width} wasn't a multiple of 32, flooring to {new_width}."
        );
        target_width = new_width;
    }
    if target_height % 32 != 0 {
        let new_height = (target_height / 32 * 32).max(32);
        log::debug!(
            "Target height of {target_height} wasn't a multiple of 32, flooring to {new_height}."
        );
        target_height = new_height;
    }
    let scale_x = image.width() as f32 / target_width as f32;
    let scale_y = image.height() as f32 / target_height as f32;
    log::debug!("Resize will change image dimensions from (w: {}, h: {}) to (w: {target_width}, h: {target_height}) with scaling factor ({scale_x}, {scale_y}).", image.width(), image.height());
    Scale {
        target_width,
        target_height,
        factor_x: scale_x,
        factor_y: scale_y,
    }
}

pub(crate) fn part_image(image: &DynamicImage, b_box: &Polygon<f32>) -> DynamicImage {
    // TODO: Do this properly
    let rect = b_box.bounding_rect().unwrap();
    let x = (rect.min().x as u32).clamp(0, image.width());
    let y = (rect.min().y as u32).clamp(0, image.height());
    let width = (rect.width() as u32).clamp(0, image.width() - x);
    let height = (rect.height() as u32).clamp(0, image.height() - y);
    log::trace!("Slicing subimage to {rect:?}");
    image.crop_imm(x, y, width, height)
}
