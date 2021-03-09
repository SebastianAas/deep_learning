import numpy as np


class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class MSE(Loss):
    def loss(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def grad(self, predicted, actual):
        return 2 * (predicted - actual)


class CrossEntropy(Loss):
    def loss(self, predicted, actual):
        return -actual * np.log2(predicted + 1e-20)

    def grad(self, predicted, actual):
        return np.where(predicted != 0, -actual / predicted, 0)


