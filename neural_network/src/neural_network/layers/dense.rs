use crate::neural_network::activations::Activation;
use crate::neural_network::layers::layer::Layer;
use crate::neural_network::regularizer::Regularization;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::fmt::Debug;

/// `Dense` layer
///  Weight matrix `w` of size (in x out)
/// Bias vector  `b` of length `out`.
/// The forward pass is given by: y = identity.(W * x .+ b)
#[derive(Debug)]
pub(crate) struct Dense {
    pub(crate) w: Array2<f64>,
    b: Array1<f64>,
    activation: Box<dyn Activation>,
    inputs: Option<Array2<f64>>,
    regularizer: Option<Box<dyn Regularization>>,
    learning_rate: f64,
    grads: Option<Array1<f64>>,
}

impl Dense {
    pub(crate) fn init(
        input: i64,
        out: i64,
        weight_range: Option<(f64, f64)>,
        activation: Box<dyn Activation>,
        learning_rate: f64,
        regularizer: Option<Box<dyn Regularization>>,
    ) -> Dense {
        let w: Array2<f64> = match weight_range {
            Some(x) => Array::random((input as usize, out as usize), Uniform::new(x.0, x.1)),
            _ => glorot_uniform((input as usize, out as usize)),
        };
        let b: Array1<f64> = Array::random(out as usize, Uniform::new(0., 1.));
        Dense {
            w,
            b,
            activation,
            inputs: None,
            learning_rate,
            grads: None,
            regularizer,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(x.clone());
        let s = &(x.dot(&self.w) + &self.b);
        let res = self.activation.forward(s);
        res
    }

    fn backward(&mut self, grad: &Array2<f64>) -> Array2<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };
        let activation_grad: Array2<f64> = self.activation.backward(grad);
        let verbose = false;
        if verbose {
            println!("w {}", self.w.clone());
            println!("b {}", self.b.clone());
            println!("grad {}", grad.clone());
            println!("activation grad {}", activation_grad.clone());
        }
        let weight_grad: Array2<f64> =
            inputs.t().dot(&activation_grad) / activation_grad.shape()[0] as f64;
        let bias_grad = activation_grad.sum() / activation_grad.shape()[0] as f64;

        let reg_loss = match &self.regularizer {
            Some(reg) => {
                self.w = self.w.clone() - (self.learning_rate * reg.regularize(&self.w));
                self.b = self.b.clone() - (self.learning_rate * reg.regularize_bias(&self.b));
                reg.loss(&self.w) + reg.loss_bias(&self.b)
            }
            _ => 0.0,
        };

        self.w = self.w.clone() - (self.learning_rate * weight_grad);
        self.b = self.b.clone() - (self.learning_rate * bias_grad);

        return reg_loss + activation_grad.dot(&self.w.t());
    }

    fn display(&self) -> String {
        format!(
            "Dense: Input: {}, Output: {}",
            self.w.shape()[0],
            self.w.shape()[1]
        )
    }
}

/// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
pub fn glorot_uniform(shape: (usize, usize)) -> ndarray::Array2<f64> {
    let s: f64 = (6. / shape.1 as f64).sqrt();
    let uniform: Uniform<f64> = Uniform::new(-s, s);
    Array::random(shape, uniform)
}
