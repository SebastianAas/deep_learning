use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Config {
    pub(crate) conv_layers: Vec<ConvParams>,
    pub(crate) dense_layers: Vec<DenseParams>,
    pub(crate) image_size: i64,
    pub(crate) dataset_size: i64,
    pub(crate) epochs: i64,
    pub regularization: String,
    pub reg_constant: f64,
    pub learning_rate: f64,
    pub loss: String,
    pub softmax: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ConvParams {
    pub(crate) size: i64,
    pub(crate) learning_rate: Option<f64>,
    pub(crate) activation: String,
    pub(crate) weight_range: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DenseParams {
    pub(crate) size: i64,
    pub(crate) learning_rate: Option<f64>,
    pub(crate) activation: String,
    pub(crate) weight_range: Option<Vec<f64>>,
}

pub(crate) fn get_config() -> Config {
    let config_file_path = Path::new("src/neural_network/config.json");
    let config_file = File::open(config_file_path).unwrap();
    serde_json::from_reader(config_file).expect("error while parsing config file")
}
