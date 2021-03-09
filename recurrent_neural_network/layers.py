import numpy as np

from activations import Linear


class Layer:

    def __init__(self, learning_rate, activation):
        self.grads = {}
        self.learning_rate = learning_rate
        self.activation = activation

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def reset_weights(self):
        raise NotImplementedError


class Input(Layer):

    def __init__(self, input_size):
        super().__init__(0, Linear())
        self.input_size = input_size

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def update(self):
        pass

    def reset_weights(self):
        pass


class Dense(Layer):

    def __init__(self, input_size, output_size, activation, learning_rate, weight_range):
        super().__init__(learning_rate, activation)
        self.weights = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(input_size, output_size))
        self.bias = np.random.rand(output_size)

    def forward(self, x):
        self.inputs = x
        output = self.activation.forward(x @ self.weights + self.bias)
        self.grads["output"] = output
        return output

    def backward(self, grad):
        activation_grad = self.activation.backward(grad)
        """
        print("Dense")
        print("grad: ", grad.shape)
        print("shape weights: ", self.weights.T.shape)
        print("shape activation: ", np.diag(np.diag(activation_grad)).shape)
        print("length diag: ", len(np.diag(activation_grad)))
        print("diag: ", np.diag(np.diag(activation_grad)).shape)
        print("act: ", activation_grad.shape, "\n")
        """

        neighbor_grad = np.einsum("Bi, io -> Bio", activation_grad, self.weights.T)
        weight_grad = update_weight_grad(self, grad, activation_grad)
        bias_grad = activation_grad.sum()

        self.grads["weights"] = weight_grad
        self.grads["bias"] = bias_grad
        """
        print("grad: ", grad.shape)
        print("neighbor: ", neighbor_grad.shape)
        print("prod: ", (grad @ neighbor_grad).shape)
        """

        output_grad = np.einsum("Bi, Bio -> Bo", grad, neighbor_grad)

        return output_grad

    def update(self):
        self.weights -= self.learning_rate * self.grads["weights"]
        self.bias -= self.learning_rate * self.grads["bias"]

    def reset_weights(self):
        self.grads.pop("weights")
        self.grads.pop("output")


class RNN(Layer):

    # 'B' is the batch dimension
    # 'i' is the input dimension
    # 'h' is the hidden dimension
    # 'o' is the output dimension

    def __init__(self, input_size, output_size, learning_rate, activation, weight_range):
        super().__init__(learning_rate, activation)
        self.inputs = []
        self.weights = np.random.uniform(low=0.0, high=0.1, size=(input_size, output_size))
        self.internal_weights = np.random.uniform(low=weight_range[0], high=weight_range[1],
                                                  size=(output_size, output_size))
        self.bias = np.random.rand(output_size)

    def forward(self, x):
        first_iteration = len(self.inputs) == 0
        self.inputs = x
        if first_iteration:
            output = self.activation.forward(x @ self.weights + self.bias)
        else:
            """
            print("shape weight: ", self.weights.shape)
            print("x: ", x.shape)
            print("shape internal: ", self.internal_weights.shape)
            print("output: ", self.outputs.shape)
            """
            output = self.activation.forward(self.outputs @ self.internal_weights + x @ self.weights)
        self.outputs = output
        return output

    def backward(self, grad):
        """
        print("RNN")
        print("grad: ", grad.shape)
        print("shape weight: ", self.weights.T.shape)
        print("shape internal: ", self.internal_weights.T.shape)
        print("act: ", activation_grad.shape)
        """

        activation_grad = self.activation.backward(grad)
        recurrent_grad = np.einsum("Bi, io -> Bio", activation_grad, self.weights.T)
        if "delta_jacobian" in self.grads:
            delta_jacobian = grad + self.grads["delta_jacobian"] @ recurrent_grad
        else:
            delta_jacobian = grad
        weight_grad = update_weight_grad(self, grad, activation_grad)
        bias_grad = activation_grad.sum()
        internal_weight_grad = update_internal_weight_grad(self, grad, activation_grad)
        neighbor_grad = np.einsum("Bi, io -> Bio", activation_grad, self.weights.T)
        out_grad = delta_jacobian @ neighbor_grad

        self.grads["delta_jacobian"] = delta_jacobian
        self.grads["weights"] = weight_grad
        self.grads["internal_weights"] = internal_weight_grad
        self.grads["bias"] = bias_grad
        self.grads["out"] = out_grad

        return out_grad

    def update(self):
        self.weights -= self.learning_rate * self.grads["weights"]
        self.internal_weights -= self.learning_rate * self.grads["internal_weights"]
        self.bias -= self.learning_rate * self.grads["bias"]

    def reset_weights(self):
        self.grads.pop("delta_jacobian")
        self.grads.pop("internal_weights")
        self.grads.pop("weights")
        self.grads.pop("bias")
        self.grads.pop("out")


def update_weight_grad(self, grad, activation_grad):
    if "weights" in self.grads:
        weight_grad = self.grads["weights"] + np.einsum("Bi, ih -> hi", grad, activation_grad.T @ self.inputs)
        return weight_grad
    else:
        weight_grad = np.einsum("Bi, ih -> hi", grad, activation_grad.T @ self.inputs)
    return weight_grad


def update_internal_weight_grad(self, grad, activation_grad):
    if "internal_weights" in self.grads:
        weight_grad = self.grads["internal_weights"] + np.einsum("Bi, ih -> hi", grad, activation_grad.T @ self.outputs)
    else:
        weight_grad = np.einsum("Bi, ih -> hi", grad, activation_grad.T @ self.outputs)
    return weight_grad
