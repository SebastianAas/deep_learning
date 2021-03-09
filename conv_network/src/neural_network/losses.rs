use ndarray::{Array2, ArrayD};

pub trait Loss {
    fn loss(&self, predicted: &ArrayD<f64>, actual: &ArrayD<f64>) -> f64;
    fn grad(&self, predicted: &ArrayD<f64>, actual: &ArrayD<f64>) -> ArrayD<f64>;
}

/// Mean Squared Error
///
/// Return the loss corresponding to mean square error
///
/// * y = actual value
/// * y_pred = predicted value
pub(crate) struct MSE {}

impl Loss for MSE {
    fn loss(&self, predicted: &ArrayD<f64>, actual: &ArrayD<f64>) -> f64 {
        let res = (predicted - actual).map(|x| x.powi(2)).mean();
        match res {
            Some(x) => x,
            _ => panic!(),
        }
    }

    fn grad(&self, predicted: &ArrayD<f64>, actual: &ArrayD<f64>) -> ArrayD<f64> {
        2.0 * (predicted - actual) / predicted.shape()[0] as f64
    }
}

/// [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
///
/// Return the loss corresponding to cross entropy
///
/// * y = actual value
/// * y_pred = predicted value
pub(crate) struct CrossEntropy {}

impl Loss for CrossEntropy {
    fn loss(&self, predicted: &ArrayD<f64>, actual: &ArrayD<f64>) -> f64 {
        let s = actual * &predicted.map(|x| (x + 1e-20).log2());
        -1.0 * (s.sum()) / predicted.shape()[0] as f64
    }

    fn grad(&self, predicted: &ArrayD<f64>, actual: &ArrayD<f64>) -> ArrayD<f64> {
        let mut res = Array2::zeros((predicted.shape()[0], predicted.shape()[1]));
        for i in 0..predicted.shape()[0] {
            for j in 0..predicted.shape()[1] {
                if predicted[[i, j]] != 0.0 {
                    res[[i, j]] = (-actual[[i, j]] / predicted[[i, j]])
                } else {
                    res[[i, j]] = 0.0
                }
            }
        }
        res.clone().into_dyn()
    }
}

#[cfg(test)]
mod tests {
    use crate::neural_network::losses::{CrossEntropy, Loss, MSE};
    use ndarray::{Ix2,array};

    #[test]
    fn it_should_give_correct_mse() {
        let actual = array![[0.1, 0.2, 0.3, 0.4, 0.5]];
        let predicted = array![[0.11, 0.19, 0.29, 0.41, 0.5]];
        let mse = MSE {};
        let left = mse.loss(&predicted.into_dyn(), &actual.into_dyn());
        assert_eq!(left, 0.00007999999999999986)
    }

    #[test]
    fn it_should_give_correct_mse_grad() {
        let actual = array![[0.1, 0.2, 0.3, 0.4, 0.5]];
        let predicted = array![[0.11, 0.19, 0.29, 0.41, 0.5]];
        let mse = MSE {};
        let left = mse.loss(&predicted.into_dyn(), &actual.into_dyn());
        assert_eq!(left, 0.00007999999999999986)
    }

    #[test]
    fn it_should_give_correct_cross_entropy_score() {
        let actual = array![[0.10, 0.40, 0.50]];
        let predicted = array![[0.80, 0.15, 0.05]];
        let cross_entropy = CrossEntropy {};
        let left = cross_entropy.loss(&predicted.into_dyn(), &actual.into_dyn());
        assert_eq!(left, 3.2879430945989)
    }

    #[test]
    fn it_should_give_correct_cross_entropy_v2() {
        let pred = array![[0.15, 0.35, 0.25, 0.25], [0.05, 0.05, 0.9, 0.0000001]];
        let target = array![[0., 0., 1., 0.], [0., 0., 1., 0.]];
        let s = CrossEntropy {};
        let left = s.loss(&pred.into_dyn(), &target.into_dyn());
        assert_eq!(left, 2.15200309344505)
    }

    #[test]
    fn it_should_give_correct_cross_entropy_grad() {
        let pred = array![[0.15, 0.35, 0.25, 0.25], [0.05, 0.05, 0.9, 0.0000001]];
        let target = array![[0., 0., 1., 0.], [0., 0., 1., 0.]];
        let s = CrossEntropy {};
        let left = s.grad(&pred.into_dyn(), &target.into_dyn());
        let right = array![
            [-0.0, -0.0, -4.0, -0.0],
            [-0.0, -0.0, -1.1111111111111112, -0.0]
        ];
        assert_eq!(left.into_dimensionality::<Ix2>().unwrap(), right)
    }
}
