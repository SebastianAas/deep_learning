use crate::neural_network::activations::*;
use crate::neural_network::config_parser::{Config, DenseParams};
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::layers::reshape_layer::ReshapeLayer;
use crate::neural_network::layers::layer::Layer;
use crate::neural_network::layers::softmax::Softmax;
use crate::neural_network::losses::*;
use crate::neural_network::regularizer::*;
use crate::visualize::*;
use ndarray::{stack, Array1, Array2, ArrayView1, Axis, ArrayD, Array3, Ix2};
use ndarray_stats::QuantileExt;
use pbr::ProgressBar;
use std::io::Stdout;

pub(crate) struct NeuralNetwork {
    pub(crate) layers: Vec<Box<dyn Layer>>,
    loss: Box<dyn Loss>,
}

impl NeuralNetwork {
    pub(crate) fn init(config: &Config) -> Self {
        let input = config.image_size.pow(2);
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        if config.dense_layers.len() > 0 {
            layers.push(Box::new(ReshapeLayer{}));
            layers.push(Box::new(Dense::init(
                input,
                config.dense_layers[0].size,
                get_weight_range(&config, 0),
                get_activation(&config.dense_layers[0]),
                get_learning_rate(&config, 0),
                get_regularization(&config.regularization, config.reg_constant),
            )));
            for i in 1..config.dense_layers.len() {
                let activation = get_activation(&config.dense_layers[i - 1]);
                let learning_rate = get_learning_rate(&config, i);
                let weight_range = get_weight_range(&config, i);
                let layer = Box::new(Dense::init(
                    config.dense_layers[i - 1].size,
                    config.dense_layers[i].size,
                    weight_range,
                    activation,
                    learning_rate,
                    get_regularization(&config.regularization, config.reg_constant),
                ));
                layers.push(layer)
            }
        }
        let learning_rate = config.learning_rate;
        if config.softmax {
            layers.push(Box::new(Softmax::init()));
        };
        let loss = get_loss(config.loss.clone());
        NeuralNetwork { layers, loss }
    }

    pub(crate) fn fit(
        &mut self,
        dataset: Array1<(Array2<f64>, Array1<f64>)>,
        epochs: i64,
        batch_size: i64,
        validation_data: Array1<(Array2<f64>, Array1<f64>)>,
        verbose: bool,
    ) {
        let mut pb: ProgressBar<Stdout> = ProgressBar::new(epochs as u64);
        let mut train_loss: Vec<f64> = Vec::new();
        let mut validation_loss: Vec<f64> = Vec::new();
        for _i in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut val_loss = 0.0;
            for (inputs, targets) in iterator(&dataset, batch_size).iter() {
                let inputs = inputs.clone().into_dyn();
                let predicted: ArrayD<f64> = self.forward(&inputs.clone());
                let loss: f64 = self.loss.loss(&predicted, &targets.clone().into_dyn());
                epoch_loss += loss;
                let loss_grad: ArrayD<f64> = self.loss.grad(&predicted, &targets.clone().into_dyn());
                self.backward(&loss_grad.into_dyn());
                if verbose {
                    println!(
                        "Input: {},\n Output: {},\n Target: {},\n Loss: {}\n\n",
                        inputs, predicted, targets, loss
                    )
                }
            }
            for (inputs, targets) in iterator(&validation_data, batch_size).iter() {
                let predicted: ArrayD<f64> = self.forward(&inputs.clone().into_dyn());
                let loss: f64 = self.loss.loss(&predicted.into_dyn(), &targets.clone().into_dyn());
                val_loss += loss;
            }
            pb.inc();
            println!("epoch loss: {}", epoch_loss);
            println!("validation loss: {}", val_loss);
            train_loss.push(epoch_loss);
            validation_loss.push(val_loss);
        }
        pb.finish();
        Visualize::plot(train_loss, validation_loss)
    }

    pub(crate) fn forward(&mut self, inputs: &ArrayD<f64>) -> ArrayD<f64> {
        self.layers
            .iter_mut()
            .fold(inputs.clone(), |acc, layer| layer.forward(&acc))
    }

    fn backward(&mut self, grad: &ArrayD<f64>) -> ArrayD<f64> {
        self.layers
            .iter_mut()
            .rev()
            .fold(grad.clone(), |acc, layer| layer.backward(&acc))
    }

    pub(crate) fn evaluate(&mut self, test_set: Array1<(Array2<f64>, Array1<f64>)>) {
        let mut score = 0.0;
        for (inputs, targets) in iterator(&test_set, 1) {
            let targets = &targets.into_dimensionality::<Ix2>().unwrap();
            let pred = self.forward(&inputs.into_dyn());
            score += if true == false {
                1.0
            } else {
                0.0
            }
        }
        println!("Accuracy: {} %", (score / test_set.len() as f64) * 100.0);
        unimplemented!()
    }
}

// TODO: Clean up
pub(crate) fn iterator(
    x: &Array1<(Array2<f64>, Array1<f64>)>,
    batch_size: i64,
) -> Vec<(Array3<f64>, Array2<f64>)> {
    let mut batch: Vec<(Array3<f64>, Array2<f64>)> = Vec::new();
    let temp = x.map(|(inputs, targets)| inputs);
    let s = temp.map(|z| z.view());
    let targets: Array1<ArrayView1<f64>> = x.map(|(inputs, targets)| targets.view());
    for (input, target) in s
        .to_vec()
        .chunks(batch_size as usize)
        .zip(targets.to_vec().chunks(batch_size as usize))
    {
        let a = stack(Axis(0), input);
        let b = stack(Axis(0), target);
        let x = (a, b);
        match x {
            (Ok(v), Ok(t)) => batch.push((v, t)),
            _ => panic!(),
        }
    }
    batch
}

fn flatten(x: &Array2<f64>) -> Array1<f64> {
    Array1::from(x.clone().into_raw_vec())
}

fn get_activation(layer: &DenseParams) -> Box<dyn Activation> {
    match layer.activation.as_str() {
        "relu" => Box::new(Relu { inputs: None }),
        "tanh" => Box::new(Tanh { inputs: None }),
        "linear" => Box::new(Linear { inputs: None }),
        _ => Box::new(Sigmoid { inputs: None }),
    }
}

fn get_loss(loss: String) -> Box<dyn Loss> {
    match loss.as_str() {
        "cross_entropy" => Box::new(CrossEntropy {}),
        _ => Box::new(MSE {}),
    }
}

fn get_regularization(reg: &String, reg_constant: f64) -> Option<Box<dyn Regularization>> {
    match reg.as_str() {
        "L1" => Some(Box::new(L1 {
            alpha: reg_constant,
        })),
        "L2" => Some(Box::new(L2 {
            alpha: reg_constant,
        })),
        _ => None,
    }
}

fn get_learning_rate(config: &Config, i: usize) -> f64 {
    match config.dense_layers[i].learning_rate {
        Some(lr) => lr,
        _ => config.learning_rate,
    }
}

fn get_weight_range(config: &Config, i: usize) -> Option<(f64, f64)> {
    match &config.dense_layers[i].weight_range {
        Some(x) => Some((x[0], x[1])),
        _ => None,
    }
}
