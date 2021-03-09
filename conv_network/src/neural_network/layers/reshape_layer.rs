use crate::neural_network::layers::layer::Layer;
use ndarray::{ArrayD,Ix4, Ix3, Ix2, Array2};

pub(crate) struct ReshapeLayer;

impl Layer for ReshapeLayer {
    fn forward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64> {
        let x = x.clone().into_dimensionality::<Ix4>().unwrap();
        let n= x.shape()[0];
        let c = x.shape()[1];
        let h = x.shape()[2];
        let w = x.shape()[3];
        let res = x.into_shape((n*c*h, w.clone())).unwrap();
        res.into_dyn()
    }

    fn backward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64> {
        x.clone()
    }

    fn display(&self) -> String {
        unimplemented!()
    }
}
