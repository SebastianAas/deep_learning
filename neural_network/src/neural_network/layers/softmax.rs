use crate::neural_network::layers::layer::Layer;
use ndarray::{array, s, stack, Array, Array1, Array2, Array3, Axis, Zip};
use ndarray_stats::QuantileExt;
use std::cmp::max;
use std::iter::FromIterator;

#[derive(Debug)]
pub(crate) struct Softmax {
    inputs: Option<Array2<f64>>,
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut z: Array2<f64> = Array2::zeros((x.shape()[0], x.shape()[1]));
    z.assign(&x);
    for i in 0..x.shape()[0] {
        let row = x.index_axis(Axis(0), i);
        let max = row.max().unwrap().clone();
        for j in 0..x.shape()[1] {
            z[[i, j]] = (z[[i, j]] - max);
        }
    }
    let mut res = z.map(|y| y.exp());
    let denominator: Array1<f64> = res.sum_axis(Axis(1));
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            res[[i, j]] = res[[i, j]] / denominator[i];
        }
    }
    res
}

impl Softmax {
    pub(crate) fn init() -> Softmax {
        Softmax { inputs: None }
    }

    pub(crate) fn derivative(&mut self) -> Array3<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };
        let s = softmax(inputs);
        let mut softmax_jacobian = Array3::zeros((s.shape()[0], s.shape()[1], s.shape()[1]));
        for n in 0..s.shape()[0] {
            for i in 0..softmax_jacobian.shape()[1] {
                for j in 0..softmax_jacobian.shape()[2] {
                    if i == j {
                        softmax_jacobian[[n, i, j]] = s[[n, i]] * (1.0 - s[[n, i]]);
                    } else {
                        softmax_jacobian[[n, i, j]] = -s[[n, i]] * s[[n, j]];
                    }
                }
            }
        }
        softmax_jacobian
    }
}

impl Layer for Softmax {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(x.clone());
        let mut z: Array2<f64> = Array2::zeros((x.shape()[0], x.shape()[1]));
        z.assign(&x);
        for i in 0..x.shape()[0] {
            let row = x.index_axis(Axis(0), i);
            let max = row.max().unwrap().clone();
            for j in 0..x.shape()[1] {
                z[[i, j]] = (z[[i, j]] - max);
            }
        }
        let mut res = z.map(|y| y.exp());
        let denominator: Array1<f64> = res.sum_axis(Axis(1));
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                res[[i, j]] = res[[i, j]] / denominator[i];
            }
        }
        res
    }

    /*
    fn backward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        fn kronecker(x: f64) -> f64 {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        };
        let kronecker_delta = x.mapv(kronecker);
        self.forward(x) * (kronecker_delta - self.forward(x))
    }

     */
    /*
    fn backward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let s = self.forward(x);
        s.clone().map(|n| n * (1f64 - n))
    }

    */
    fn backward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let softmax_jacobian = self.derivative().clone();
        let mut r = Vec::new();
        for n in 0..softmax_jacobian.shape()[0] {
            let matrix = softmax_jacobian.index_axis(Axis(0), n);
            let grad = x.index_axis(Axis(0), n);
            let k = matrix.dot(&grad);
            r.push(k);
        }
        let mut temp = Vec::new();
        for t in r.iter() {
            temp.push(t.view())
        }
        stack(Axis(0), temp.as_slice()).unwrap()
    }

    /*
    fn backward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }
     */

    fn display(&self) -> String {
        "Softmax".parse().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_should_give_correct_softmax() {
        let a = array![[1.0, 3.0, 2.0]];
        let b = array![[3.0, 1.0, 0.2]];
        let mut s = Softmax::init();
        let left = s.forward(&a);
        let left2 = s.forward(&b);
        assert_eq!(
            left,
            array![[0.09003057317038046, 0.6652409557748218, 0.24472847105479764]]
        );
        assert_eq!(
            left2,
            array![[0.8360188027814407, 0.11314284146556013, 0.05083835575299916]]
        )
    }

    #[test]
    fn it_should_give_correct_softmax_v2() {
        let a = array![[1., 3., -1.], [1., -100., 3.]];
        let mut s = Softmax::init();
        let left = s.forward(&a);
        assert_eq!(
            left,
            array![
                [
                    0.11731042782619835,
                    0.8668133321973347,
                    0.015876239976466762
                ],
                [
                    0.11920292202211755,
                    0.0000000000000000000000000000000000000000000016313390386652676,
                    0.8807970779778823
                ]
            ]
        )
    }

    #[test]
    fn it_should_give_correct_softmax_derivative() {
        let a = array![[1., 3., -1.], [1., -100., 3.]];
        let temp_grad = array![[0.3, -0.6, 1.], [1., 1., 1.]];
        let mut s = Softmax::init();
        let forward_call = s.forward(&a);
        let left = s.derivative();
        let right = array![
            [
                [
                    0.10354869134943266,
                    -0.10168624284552193,
                    -0.0018624485039107092
                ],
                [
                    -0.10168624284552193,
                    0.11544797932228779,
                    -0.013761736476765688
                ],
                [
                    -0.0018624485039107092,
                    -0.013761736476765688,
                    0.0156241849806764
                ]
            ],
            [
                [
                    0.1049935854035065,
                    -0.0000000000000000000000000000000000000000000001944603802176521,
                    -0.10499358540350649
                ],
                [
                    -0.0000000000000000000000000000000000000000000001944603802176521,
                    0.0000000000000000000000000000000000000000000016313390386652676,
                    -0.0000000000000000000000000000000000000000000014368786584476152
                ],
                [
                    -0.10499358540350649,
                    -0.0000000000000000000000000000000000000000000014368786584476152,
                    0.10499358540350662
                ]
            ]
        ];
        assert_eq!(left, right);
    }

    #[test]
    fn it_should_give_correct_softmax_backward() {
        let a = array![[1., 3., -1.], [1., -100., 3.]];
        let temp_grad = array![[0.3, -0.6, 1.], [1., 1., 1.]];
        let mut s = Softmax::init();
        let forward_call = s.forward(&a);
        let left = s.backward(&temp_grad);
        assert_eq!(
            forward_call,
            array![
                [
                    0.11731042782619835,
                    0.8668133321973347,
                    0.015876239976466762
                ],
                [
                    0.11920292202211755,
                    0.0000000000000000000000000000000000000000000016313390386652676,
                    0.8807970779778823
                ]
            ]
        );
        assert_eq!(
            left,
            array![
                [
                    0.09021390460823224,
                    -0.11353639692379494,
                    0.0233224923155626
                ],
                [
                    0.000000000000000013877787807814457,
                    0.0000000000000000000000000000000000000000000000000000000000003111507638930571,
                    0.0000000000000001249000902703301
                ]
            ]
        )
    }
}
