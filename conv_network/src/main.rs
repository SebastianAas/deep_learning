#![feature(array_map)]

use ndarray::{Array1, Array2};

use crate::neural_network::neural_network::NeuralNetwork;
use neural_network::config_parser;

mod data_generator;
mod neural_network;
mod visualize;

extern crate image;

fn main() {
    let config = config_parser::get_config();
    let dataset: Array1<(Array2<f64>, Array1<f64>)> = data_generator::generate_dataset(
        config.dataset_size,
        config.image_size,
        config.image_size,
        false,
        0.001,
    );
    let (train, validation, test) = data_generator::split_dataset(dataset, 0.7, 0.15);
    println!(
        "train: {}, val: {}, test: {}",
        train.len(),
        validation.len(),
        test.len()
    );
    data_generator::save_images(10, train.clone());
    let mut nn = NeuralNetwork::init(&config);
    nn.fit(
        train.clone(),
        config.epochs.clone(),
        32,
        validation.clone(),
        false,
    );
    nn.evaluate(test.clone())
}

