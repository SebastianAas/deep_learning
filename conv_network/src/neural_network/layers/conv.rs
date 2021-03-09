use ndarray::{array, Array1, Array2, Array, Array3, s, Dim, ArrayView, ArrayView2, ArrayD, Ix3, Ix4, ArrayBase, OwnedRepr, Array4, Zip, Axis, ArrayView3};
use crate::neural_network::activations::Activation;
use crate::neural_network::layers::layer::Layer;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde_json::to_string;


pub struct Conv {
    inputs: Option<Array3<f64>>,
    grads: Option<Array1<f64>>,
    filters: Array3<f64>,
    filter_size: usize,
    num_filters: usize,
    stride: usize,
    mode: String,
}

impl Conv {
    pub(crate) fn init(
        num_filters: usize,
        filter_size: usize,
        stride: usize,
        mode: String,
    ) -> Conv {
        let filters: Array3<f64> = Array3::random((num_filters, filter_size, filter_size), Uniform::new(0., 0.1));
        Conv {
            inputs: None,
            grads: None,
            filters,
            filter_size,
            num_filters,
            stride,
            mode,
        }
    }

    fn iterate_regions(&self, image: &Array2<f64>) {
        let h = image.shape()[0];
        let w = image.shape()[1];
        let mut vec = Vec::new();
        for i in 0..(w - self.filter_size + 1) {
            for j in 0..(h - self.filter_size + 1) {
                let im_region = image.slice(s![i..i+self.filter_size, j..j+self.filter_size]).clone();
                println!("slice: {}", im_region);
                vec.push((im_region, i, j));
            }
        }
    }

    fn get_padding(&self, input: &Array2<f64>) -> (usize, usize) {
        match self.mode.as_str() {
            "valid" => (input.shape()[0], input.shape()[1]),
            "full" => (input.shape()[0] + self.filter_size, input.shape()[1] + self.filter_size),
            _ => {
                (input.shape()[0] + self.filter_size - 1, input.shape()[1] + self.filter_size - 1)
            }
        }
    }

    fn zero_padding(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let h = inputs.shape()[0];
        let w = inputs.shape()[1];
        let (new_h, new_w) = self.get_padding(inputs);
        let mut out = Array2::zeros((new_h, new_w));
        let padding_h = (new_h - h) / 2;
        let padding_w = (new_w - w) / 2;
        for i in 0..h {
            for j in 0..w {
                out[[i + padding_h, j + padding_w]] = inputs.clone()[[i, j]]
            }
        }
        out
    }
}

impl Layer for Conv {
    /*
    Input size: (N, W, H)
    Output size: (N, F, H, W)
     */
    fn forward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64> {
        let x: Array3<f64> = x.clone().into_dimensionality::<Ix3>().unwrap();
        let N = x.shape()[0];
        let H = x.shape()[1];
        let W = x.shape()[2];
        let mut padded_inputs = Array3::zeros((N, H, W));
        for i in 0..N {
            let mut row = padded_inputs.index_axis_mut(Axis(0), i);
            let image = x.index_axis(Axis(0), i);
            let padded = self.zero_padding(&image.to_owned());
            row.assign(&padded);
        }
        self.inputs = Some(padded_inputs.clone());
        let hh = (H - self.filter_size) / self.stride + 1;
        let ww = (W - self.filter_size) / self.stride + 1;
        let mut feature_maps: Array4<f64> = Array4::zeros((N, self.num_filters, hh, ww));
        for i in 0..N {
            for f in 0..self.num_filters {
                for h in 0..hh {
                    for w in 0..ww {
                        let image_patch = padded_inputs.slice(s![.., h..h+self.filter_size, w..w+self.filter_size]);
                        let weight = self.filters.index_axis(Axis(0), f);
                        feature_maps[[i, f, w, h]] = (&image_patch * &weight).sum();
                    }
                }
            }
        }
        feature_maps.into_dyn()
    }

    fn backward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64> {
        let inputs = match &self.inputs {
            Some(x) => x,
            _ => panic!(),
        };

        let N = x.shape()[0];
        let H = x.shape()[1];
        let W = x.shape()[2];

        let mut dx: Array3<f64> = Array3::zeros(inputs.raw_dim());
        let mut dw: Array3<f64> = Array3::zeros(self.filters.raw_dim());

        for n in 0..N {
            for h in 0..H {
                for w in 0..W {
                    // Update Weights
                    let image_patch:ArrayView3<f64> = inputs.slice(s![.., h..h+self.filter_size, w..w+self.filter_size]);
                    let mut subview = dw.index_axis_mut(Axis(0), n);
                    let image: Array3<f64> = image_patch.map(|z| z * x[[n,h,w]]);
                    let res = &subview + &image;
                    subview.assign(&res);

                    // Update gradient
                    let subview_dx = dx.slice(s![.., h..h+self.filter_size, w..w+self.filter_size]);
                    let updated_dx = x[[n,h,w]] * &self.filters.slice(s![n, .., ..]);
                }
            }
        }

        self.filters = self.filters.clone() - self.learning_rate * dw;
        dx.into_dyn()
    }

    fn display(&self) -> String {
        unimplemented!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Zip, arr2, arr3};

    #[test]
    fn it_should_give_correct_zero_padding() {
        let mut a = arr2(&[[1.0, 3.0, 2.0], [3.0, 1.0, 0.2], [3.0, 1.0, 0.2]]);
        let stride = 1;
        let mut b: Array1<f64> = array![1.0, 1.0,1.0];
        let mut row = a.view_mut();
        let mut c = a.index_axis_mut(Axis(0), 1);
        let res = &c + &b;
        c.assign(&res);
        println!("a: {}", a.clone());
        let mode = "full";
        let mut s = Conv::init(10, 3, stride, mode.to_string());
    }

    #[test]
    fn it_should_give_correct_zero_padding2() {
        let a = array![
            [[
                    0.1,
                    0.1,
                    1.0
                ],
                [
                    0.1,
                    0.1,
                    1.0
                ],
                [
                    0.1,
                    0.1,
                    1.0
                ],
            ],
            [
                [
                    1.1,
                    1.1,
                    1.1
                ],
                [
                    0.1,
                    0.1,
                    1.0
                ],
                [
                    0.1,
                    0.1,
                    1.0
                ],
            ],
            [
                [
                    1.1,
                    1.1,
                    1.1
                ],
                [
                    0.1,
                    0.1,
                    1.0
                ],
                [
                    0.1,
                    0.1,
                    1.0
                ],
            ]
        ];
        let stride = 1;
        let mode = "full";
        let mut s = Conv::init(5, 2, stride, mode.to_string());
        let left = s.forward(&a.into_dyn());
        let right: Array4<f64> = Array4::zeros((4, 4, 4, 4));
        assert_eq!(left.into_dimensionality::<Ix4>().unwrap(), right);
    }
}
