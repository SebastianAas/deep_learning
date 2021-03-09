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

/*
1. The data generator and image viewer.
(5 points)
2.  The (verbal) description of the format for configuration files, along with code to parse those files.  (3points)
3.  The forward pass of the deep network, with core functions performed with matrix and vector operationsin numpy (or a similar package) (5 points)
4.  The backward pass (backpropagation) for modifying the network weights and biases.
    This must clearly perform the explicit computation of the key Jacobian matrices and vectors, as specified in the lecture notes.  (15 points)
5.  A graphic display of learning progress (2 points)
 */
