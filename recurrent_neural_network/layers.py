import numpy as np

from activations import Linear


class Layer:

    def __init__(self, learning_rate, activation, name):
        self.grads = {}
        self.learning_rate = learning_rate
        self.activation = activation
        self.name = name

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def reset_inputs(self):
        raise NotImplementedError

    def reset_weights(self):
        raise NotImplementedError


class Input(Layer):

    def __init__(self, input_size):
        super().__init__(0, Linear(), "Input")
        self.input_size = input_size

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def update(self):
        pass

    def reset_cache(self):
        pass


class Dense(Layer):

    def __init__(self, input_size, output_size, activation, learning_rate, weight_range):
        super().__init__(learning_rate, activation, "Dense")
        self.weights = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(input_size, output_size))
        self.bias = np.random.rand(output_size)
        self.inputs = None
        self.outputs = []

    def forward(self, x):
        self.inputs = x
        output = self.activation.forward(x @ self.weights)  # + self.bias)
        self.outputs.append(output)
        return output

    def backward(self, grad):
        activation_grad = self.activation.backward(self.outputs.pop())

        neighbor_grad = np.mean([np.diag(activation_grad[i]) @ self.weights.T for i in range(len(activation_grad))],
                                axis=0)
        weight_grad = update_weight_grad(self, grad, activation_grad)
        bias_grad = activation_grad.sum()

        self.grads["weights"] = weight_grad
        self.grads["bias"] = bias_grad

        output = grad @ neighbor_grad

        return output

    def update(self):
        self.weights -= self.learning_rate * self.grads["weights"]
        self.bias -= self.learning_rate * self.grads["bias"]

    def reset_cache(self):
        self.inputs = None
        self.outputs = []
        if "weights" in self.grads:
            self.grads.pop("weights")
            self.grads.pop("bias")


class RNN(Layer):

    def __init__(self, input_size, output_size, learning_rate, activation, weight_range):
        super().__init__(learning_rate, activation, "RNN")
        self.inputs = None
        self.outputs = []
        self.weights = np.random.uniform(low=0.0, high=0.1, size=(input_size, output_size))
        self.internal_weights = np.random.uniform(low=weight_range[0], high=weight_range[1],
                                                  size=(output_size, output_size))
        self.bias = np.random.rand(output_size)

    def forward(self, x):
        if self.inputs is None:
            first_iteration = True
        else:
            first_iteration = False
        self.inputs = x
        if first_iteration:
            output = self.activation.forward(x @ self.weights + self.bias)
        else:
            output = self.activation.forward(
                self.outputs[-1] @ self.internal_weights + x @ self.weights)  # + self.bias)
        self.outputs.append(output)
        return output

    def backward(self, grad):

        if "delta_jacobian" in self.grads:
            previous_output = self.outputs.pop()
            activation_grad = self.activation.backward(previous_output)
            recurrent = np.sum([np.diag(row) * self.internal_weights.T for row in activation_grad], axis=0)
            delta_jacobian = grad + self.grads["delta_jacobian"] @ recurrent
        else:
            activation_grad = np.ones(grad.shape)
            delta_jacobian = grad
        act_k = self.activation.backward(self.outputs[-1])
        if len(self.outputs) == 1:
            next_output = np.zeros(grad.shape)
        else:
            next_output = self.outputs[-2]
        weight_grad = update_weight_grad(self, grad, act_k)
        bias_grad = activation_grad.sum()
        internal_weight_grad = update_internal_weight_grad(self, grad, act_k, next_output)
        neighbor = np.sum([np.diag(row) @ self.weights.T for row in activation_grad], axis=0)
        out_grad = delta_jacobian @ neighbor

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

    def reset_cache(self):
        self.inputs = None
        self.outputs = []
        if "delta_jacobian" in self.grads:
            self.grads.pop("delta_jacobian")
            self.grads.pop("internal_weights")
            self.grads.pop("weights")
            self.grads.pop("bias")
            self.grads.pop("out")


def update_weight_grad(self, grad, activation_grad):
    new_grad = np.sum([np.diag(grad[i]) @ np.outer(activation_grad[i], self.inputs[i]) for i in range(grad.shape[0])],
                      axis=0)
    if "weights" in self.grads:
        weight_grad = self.grads["weights"] + new_grad.T
        return weight_grad
    return new_grad.T


def update_internal_weight_grad(self, grad, activation_grad, next_output):
    new_grad = np.sum([np.diag(grad[i]) @ np.outer(activation_grad[i], next_output[i]) for i in range(grad.shape[0])],
                      axis=0)
    if "internal_weights" in self.grads:
        weight_grad = self.grads["internal_weights"] + new_grad.T
        return weight_grad
    return new_grad.T
