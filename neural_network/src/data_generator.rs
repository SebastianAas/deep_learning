/*
1. The generator is object-oriented
2. Image dimensions are n x n, for all n where 10≤n≤50; the user can choose n when requesting a newimage set
3.  At least 4 different classes of images can be generated, e.g.  circles, rectangles, triangles,
    and crosses;or a’s, b’s, q’s and x’s, or any other type of images that you prefer
4.  Any generated image set should have approximately the same number of images of each class
5.  A user-specified noise parameter will control the fraction of randomly-set pixels in each image
6.  The generator will normally return 3 image sets corresponding to training, validation and testing,
w   ith the relative sizes of each also specified by the user – for example 70% training, 20% validation, 10%testing
7.  A user-specified size range for the height and width of the image (compared to the n x n background)
    will support images of the same class but of different sizes
8.  A flattening option is included such that images can be returned either as 2-d arrays or as vectors.
 */

use image::{ImageBuffer, Rgb};
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;
use std::collections::hash_map::RandomState;
use std::collections::HashSet;

type Dataset = Array1<(Array2<f64>, Array1<f64>)>;

/// ## Generates a dataset
/// # Arguments
/// * `size` - size of dataset
/// * `dimension` - dimension of the array
/// * `traning_size` - size of training set
/// * `test_size`  - size of test set
/// * `noise_parameter` - percentage of random particles
/// * `image_size` - size of the image
pub fn generate_dataset(
    size: i64,
    dimension: i64,
    image_size: i64,
    centering: bool,
    noise_parameter: f64,
) -> Dataset {
    let mut dataset: Vec<(Array2<f64>, Array1<f64>)> = Vec::new();
    let generators: [fn(i64, i64, bool) -> Array2<f64>; 4] = [
        generate_circle,
        generate_rectangles,
        generate_horizontal_bars,
        generate_vertical_bars,
    ];
    for _x in 0..size {
        for generator in generators.iter() {
            let image = generator(dimension, image_size, centering);
            let target = get_target(generator);
            dataset.push((random_particles(&image, noise_parameter), target));
            if dataset.len() == size as usize {
                break;
            }
        }
        if dataset.len() == size as usize {
            break;
        }
    }
    dataset
        .into_iter()
        .collect::<Array1<(Array2<f64>, Array1<f64>)>>()
}

pub fn split_dataset(
    dataset: Array1<(Array2<f64>, Array1<f64>)>,
    train_size: f64,
    validation_size: f64,
) -> (Dataset, Dataset, Dataset) {
    let mut rng = rand::thread_rng();
    let mut train: Vec<(Array2<f64>, Array1<f64>)> = Vec::new();
    let mut validation: Vec<(Array2<f64>, Array1<f64>)> = Vec::new();
    let mut test: Vec<(Array2<f64>, Array1<f64>)> = Vec::new();
    for data in dataset.iter() {
        let r = rng.gen::<f64>();
        if train_size > r {
            train.push(data.clone())
        } else if (train_size + validation_size) > r {
            validation.push(data.clone())
        } else {
            test.push(data.clone())
        }
    }
    (
        Array1::from(train),
        Array1::from(validation),
        Array1::from(test),
    )
}

fn get_target(generator: &fn(i64, i64, bool) -> Array2<f64>) -> Array1<f64> {
    let generators: [fn(i64, i64, bool) -> Array2<f64>; 4] = [
        generate_circle,
        generate_rectangles,
        generate_horizontal_bars,
        generate_vertical_bars,
    ];
    generators
        .iter()
        .map(|x| if x == generator { 1.0 } else { 0.0 })
        .collect::<Array1<f64>>()
}

pub fn save_images(n: i64, arrays: Dataset) {
    for (index, (array, _)) in arrays.indexed_iter() {
        save_image(array, format!("example_images/image-{}.png", index as i64));
        if index == n as usize {
            break;
        }
    }
}

pub fn save_image(array: &Array2<f64>, save_path: String) {
    let dimensions: u64 = array.len_of(Axis(1)) as u64;
    let mut imgbuf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::new(dimensions as u32, dimensions as u32);
    for x in 0..dimensions {
        for y in 0..dimensions {
            let pixel = imgbuf.get_pixel_mut(y as u32, x as u32);
            if array[[x as usize, y as usize]] == 1.0 {
                *pixel = image::Rgb([0, 0 as u8, 0]);
            } else {
                *pixel = image::Rgb([255, 255 as u8, 255]);
            }
        }
    }
    imgbuf.save(save_path).unwrap();
}

pub fn generate_circle(dimensions: i64, image_size: i64, center: bool) -> Array2<f64> {
    let frame: Array2<i64> = Array2::zeros((dimensions as usize, dimensions as usize));
    let mut rng = rand::thread_rng();
    let radius: i64 = rng.gen_range(1..image_size / 2);
    let safe_area = [radius, radius];
    let starting_point: [i64; 2] = if center {
        [image_size / 2, image_size / 2]
    } else {
        safe_area.map(|_x| rng.gen_range(radius..image_size - radius))
    };
    let array: Array1<f64> = frame
        .indexed_iter()
        .map(|(index, _)| {
            match index {
                (x, y)
                    if (x as i64 - starting_point[0]).pow(2)
                        + (y as i64 - starting_point[1]).pow(2)
                        <= radius.pow(2) =>
                {
                    1.0
                } //&(*&x as i64)
                _ => 0.0,
            }
        })
        .collect::<Array1<f64>>();
    reshape_array(array)
}

pub fn generate_horizontal_bars(dimensions: i64, image_size: i64, center: bool) -> Array2<f64> {
    let frame: Array2<i64> = Array::zeros((dimensions as usize, dimensions as usize));
    let mut rng = rand::thread_rng();
    let bars: i64 = rng.gen_range(2..image_size / 2);
    let bar_positions: HashSet<i64> = random_set(bars, image_size);
    let array: Array1<f64> = frame
        .indexed_iter()
        .map(|(index, _)| match index {
            (x, _) if bar_positions.contains(&(*&x as i64)) => 1.0,
            _ => 0.0,
        })
        .collect::<Array1<f64>>();
    reshape_array(array)
}

pub fn generate_vertical_bars(dimensions: i64, image_size: i64, center: bool) -> Array2<f64> {
    let frame: Array2<f64> = Array::zeros((dimensions as usize, dimensions as usize));
    let mut rng = rand::thread_rng();
    let bars: i64 = rng.gen_range(2..image_size / 2);
    let bar_positions: HashSet<i64, RandomState> = random_set(bars, image_size);
    let array: Array1<f64> = frame
        .indexed_iter()
        .map(|(index, _)| match index {
            (_, y) if bar_positions.contains(&(*&y as i64)) => 1.0,
            _ => 0.0,
        })
        .collect::<Array1<f64>>();
    reshape_array(array)
}

pub fn generate_rectangles(dimensions: i64, image_size: i64, center: bool) -> Array2<f64> {
    let frame: Array2<i64> = Array::zeros((image_size as usize, image_size as usize));
    let mut rng = rand::thread_rng();
    let height: i64 = rng.gen_range(2..image_size);
    let width: i64 = rng.gen_range(2..image_size);
    let safe_area = [image_size - height, image_size - width];
    let starting_point = safe_area.map(|x| rng.gen_range(0..x));
    let array: Array1<f64> = frame
        .indexed_iter()
        .map(|(index, _)| match index {
            (x, y)
                if x >= starting_point[0] as usize
                    && x < (starting_point[0] + height) as usize
                    && (y == starting_point[1] as usize
                        || y == (starting_point[1] + width) as usize) =>
            {
                1.0
            }
            (x, y)
                if y >= starting_point[1] as usize
                    && y < (starting_point[1] + width) as usize
                    && (x == starting_point[0] as usize
                        || x == (starting_point[0] + height - 1) as usize) =>
            {
                1.0
            }
            _ => 0.0,
        })
        .collect::<Array1<f64>>();
    reshape_array(array)
}

fn reshape_array(array: Array1<f64>) -> Array2<f64> {
    let shape = (array.len() as f64).sqrt() as usize;
    let reshaped = Array2::from_shape_vec((shape, shape), array.to_vec());
    match reshaped {
        Ok(res) => res,
        _ => panic!(),
    }
}

fn random_particles(array: &Array2<f64>, noise_parameter: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    array.map(|cell| {
        if noise_parameter > rng.gen::<f64>() {
            if *cell == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            if *cell == 0.0 {
                0.0
            } else {
                1.0
            }
        }
    })
}

fn random_set(n: i64, image_size: i64) -> HashSet<i64, RandomState> {
    let mut set: HashSet<i64> = HashSet::new();
    let mut rng = rand::thread_rng();
    while set.len() != n as usize {
        set.insert(rng.gen_range(0..image_size));
    }
    set
}
