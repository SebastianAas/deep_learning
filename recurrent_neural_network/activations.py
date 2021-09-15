import numpy as np


class Activation:

    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative

    def forward(self, x):
        self.inputs = x
        return self.activation(x)

    def backward(self, grad):
        return self.derivative(self.inputs) * grad


class Relu(Activation):

    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_derivative(x):
            return (x > 0).astype(float)


        super().__init__(relu, relu_derivative)


class Tanh(Activation):

    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            y = np.tanh(x)
            return 1 - np.power(y, 2)

        super().__init__(tanh, tanh_derivative)


class Sigmoid(Activation):

    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))

        super().__init__(sigmoid, sigmoid_derivative)


class Linear(Activation):

    def __init__(self):
        def linear(x):
            return x

        def linear_derivative(x):
            return 1

        super().__init__(linear, linear_derivative)
