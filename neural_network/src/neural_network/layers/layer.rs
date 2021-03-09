use ndarray::Array2;
use std::fmt::Debug;

pub trait Layer {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, x: &Array2<f64>) -> Array2<f64>;
    fn display(&self) -> String;
}

impl Debug for dyn Layer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.display())
    }
}
