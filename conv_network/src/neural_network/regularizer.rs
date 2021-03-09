use ndarray::{array, Array1, Array2};
use std::fmt::Debug;

pub trait Regularization {
    fn call(&self, x: &Array2<f64>) -> Array2<f64>;
    fn derivative(&self, grad: &Array2<f64>) -> Array2<f64>;
    fn loss(&self, x: &Array2<f64>) -> f64;
    fn loss_bias(&self, x: &Array1<f64>) -> f64;
    fn regularize(&self, x: &Array2<f64>) -> Array2<f64>;
    fn regularize_bias(&self, x: &Array1<f64>) -> Array1<f64>;
}

impl Debug for dyn Regularization {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", "activation function")
    }
}

pub(crate) struct L1 {
    pub(crate) alpha: f64,
}

impl Regularization for L1 {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn derivative(&self, grad: &Array2<f64>) -> Array2<f64> {
        Array2::ones((grad.shape()[0], grad.shape()[1]))
    }

    fn loss(&self, x: &Array2<f64>) -> f64 {
        self.alpha * self.call(x).sum()
    }

    fn loss_bias(&self, x: &Array1<f64>) -> f64 {
        self.alpha * (0.5 * x * x).sum()
    }

    fn regularize(&self, x: &Array2<f64>) -> Array2<f64> {
        self.alpha * self.derivative(x)
    }

    fn regularize_bias(&self, x: &Array1<f64>) -> Array1<f64> {
        self.alpha * Array1::ones(x.shape()[0])
    }
}

pub(crate) struct L2 {
    pub(crate) alpha: f64,
}

impl Regularization for L2 {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        0.5 * x * x
    }

    fn derivative(&self, grad: &Array2<f64>) -> Array2<f64> {
        grad.clone()
    }

    fn loss(&self, x: &Array2<f64>) -> f64 {
        self.alpha * self.call(x).sum()
    }

    fn loss_bias(&self, x: &Array1<f64>) -> f64 {
        self.alpha * (0.5 * x * x).sum()
    }

    fn regularize(&self, x: &Array2<f64>) -> Array2<f64> {
        self.alpha * self.derivative(x)
    }
    fn regularize_bias(&self, x: &Array1<f64>) -> Array1<f64> {
        self.alpha * x.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_should_give_correct_L2_loss() {
        let a = array![[1.0, 2.], [2.0, 3.0]];
        let r = L2 { alpha: 0.001 };
        let left = r.loss(&a);
        assert_eq!(left, 0.009000000000000001)
    }

    #[test]
    fn it_should_give_correct_L2_regularization() {
        let a = array![[1.0, 2.], [2.0, 3.0]];
        let r = L2 { alpha: 0.001 };
        let left = r.regularize(&a);
        assert_eq!(left, array![[0.001, 0.002], [0.002, 0.003]])
    }

    #[test]
    fn it_should_give_correct_L1_loss() {
        let a = array![[1.0, 2.], [2.0, 3.0]];
        let r = L1 { alpha: 0.001 };
        let left = r.loss(&a);
        assert_eq!(left, 0.008)
    }

    #[test]
    fn it_should_give_correct_L1_regularization() {
        let a = array![[1.0, 2.], [2.0, 3.0]];
        let r = L1 { alpha: 0.001 };
        let left = r.regularize(&a);
        assert_eq!(left, array![[0.001, 0.001], [0.001, 0.001]])
    }
}
