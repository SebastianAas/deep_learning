use ndarray::{Array, Array2, Ix2};
use ndarray_rand::rand_distr::num_traits::{abs, Inv, Pow};

pub trait Activation {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, grad: &Array2<f64>) -> Array2<f64>;
    fn call(&self, x: &Array2<f64>) -> Array2<f64>;
    fn derivative(&self, input: &Array2<f64>) -> Array2<f64>;
}
use core::fmt::Debug;

impl Debug for dyn Activation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", "activation function")
    }
}

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation function.
/// σ(x) = 1 / (1 + exp(-x))
#[derive(Debug)]
pub(crate) struct Sigmoid {
    pub(crate) inputs: Option<Array2<f64>>,
}

impl Activation for Sigmoid {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(x.clone());
        self.call(x)
    }

    fn backward(&mut self, grad: &Array2<f64>) -> Array2<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };
        self.derivative(inputs) * grad
    }

    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        fn sigmoid(x: f64) -> f64 {
            let t = (-abs(x)).exp();
            if t >= 0.0 {
                (t - 1.0).inv()
            } else {
                t / (1.0 + t)
            }
        }
        x.mapv(sigmoid)
    }

    fn derivative(&self, input: &Array2<f64>) -> Array2<f64> {
        self.call(input) * self.call(&(1.0 - input))
    }
}

///[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
// activation function.
/// relu(x) = max(0, x)
#[derive(Debug)]
pub(crate) struct Relu {
    pub(crate) inputs: Option<Array2<f64>>,
}

impl Activation for Relu {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(x.clone());
        self.call(x)
    }

    fn backward(&mut self, grad: &Array2<f64>) -> Array2<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };
        self.derivative(inputs) * grad
    }
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|x| x.max(0.0))
    }

    fn derivative(&self, input: &Array2<f64>) -> Array2<f64> {
        input.map(|x| if x.clone() > 0.0 { 1.0 } else { 0.0 })
    }
}

/// Hyperbolic tangent activation function

#[derive(Debug)]
pub(crate) struct Tanh {
    pub(crate) inputs: Option<Array2<f64>>,
}

impl Activation for Tanh {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(x.clone());
        self.call(x)
    }

    fn backward(&mut self, grad: &Array2<f64>) -> Array2<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };
        self.derivative(inputs) * grad
    }
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|z| z.tanh())
    }

    fn derivative(&self, input: &Array2<f64>) -> Array<f64, Ix2> {
        let y = input.map(|x| x.tanh());
        1.0 - y.map(|x| x.pow(2.0))
    }
}

/// Linear activation function
#[derive(Debug)]
pub(crate) struct Linear {
    pub(crate) inputs: Option<Array2<f64>>,
}

impl Activation for Linear {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(x.clone());
        self.call(x)
    }

    fn backward(&mut self, grad: &Array2<f64>) -> Array2<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };
        self.derivative(inputs) * grad
    }
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn derivative(&self, input: &Array<f64, Ix2>) -> Array<f64, Ix2> {
        input.map(|_x| 1.0)
    }
}
