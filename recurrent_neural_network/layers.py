import numpy as np

from activations import Linear

print_shapes = False

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

    def reset_inputs(self):
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

    def reset_cache(self):
        pass


class Dense(Layer):

    def __init__(self, input_size, output_size, activation, learning_rate, weight_range):
        super().__init__(learning_rate, activation)
        self.weights = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(input_size, output_size))
        self.bias = np.random.rand(output_size)
        self.inputs = None
        self.outputs = []

    def forward(self, x):
        self.inputs = x
        output = self.activation.forward(x @ self.weights + self.bias)
        self.outputs.append(output)
        return output

    def backward(self, grad):
        activation_grad = self.activation.backward(self.outputs.pop())

        neighbor_grad = np.einsum("Bi, io -> Bio", activation_grad, self.weights.T)
        weight_grad = update_weight_grad(self, grad, activation_grad)
        bias_grad = activation_grad.sum()

        self.grads["weights"] = weight_grad
        self.grads["bias"] = bias_grad

        output_grad = np.einsum("Bi, Bio -> Bo", grad, neighbor_grad)

        return output_grad

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

    # 'B' is the batch dimension
    # 'i' is the input dimension
    # 'h' is the hidden dimension
    # 'o' is the output dimension

    def __init__(self, input_size, output_size, learning_rate, activation, weight_range):
        super().__init__(learning_rate, activation)
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
            if print_shapes:
                print("shape weight: ", self.weights.shape)
                print("x: ", x.shape)
                print("shape internal: ", self.internal_weights.shape)
                print("output: ", self.outputs.shape)
            output = self.activation.forward(self.outputs[-1] @ self.internal_weights + x @ self.weights)
        self.outputs.append(output)
        return output

    def backward(self, grad):
        if print_shapes:
            print("RNN")
            print("grad: ", grad.shape)
            print("shape weight: ", self.weights.T.shape)
            print("shape internal: ", self.internal_weights.T.shape)

        if "delta_jacobian" in self.grads:
            previous_output = self.outputs.pop()
            activation_grad = self.activation.backward(previous_output)
            recurrent_grad = np.einsum("Bi, io -> Bio", activation_grad, self.weights.T)
            delta_jacobian = grad + np.einsum("Bi, Bio -> Bi", self.grads["delta_jacobian"], recurrent_grad)
        else:
            activation_grad = np.zeros(grad.shape)
            delta_jacobian = grad
        act_k = self.activation.backward(self.outputs[-1])
        if len(self.outputs) == 1:
            next_output = np.zeros(grad.shape)
        else:
            next_output = self.outputs[-2]
        weight_grad = update_weight_grad(self, grad, act_k)
        bias_grad = activation_grad.sum()
        internal_weight_grad = update_internal_weight_grad(self, grad, act_k, next_output)
        neighbor_grad = np.einsum("Bi, io -> Bio", activation_grad, self.weights.T)
        out_grad = delta_jacobian @ neighbor_grad

        self.grads["delta_jacobian"] = delta_jacobian
        self.grads["weights"] = weight_grad
        self.grads["internal_weights"] = internal_weight_grad
        self.grads["bias"] = bias_grad
        self.grads["out"] = out_grad

        return out_grad

    def update(self):
        #print("update: ", self.grads["weights"])
        self.weights -= self.learning_rate * self.grads["weights"]
        self.internal_weights -= self.learning_rate * self.grads["internal_weights"]
        #self.bias -= self.learning_rate * self.grads["bias"]


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
    if "weights" in self.grads:
        weight_grad = self.grads["weights"] + np.einsum("Bi, ih -> hi", grad, activation_grad.T @ self.inputs)
        return weight_grad
    else:
        weight_grad = np.einsum("Bi, ih -> hi", grad, activation_grad.T @ self.inputs)
    return weight_grad


def update_internal_weight_grad(self, grad, activation_grad, next_output):
    if "internal_weights" in self.grads:
        weight_grad = self.grads["internal_weights"] + np.einsum("Bi, ih -> hi", grad, activation_grad.T @ next_output)
    else:
        weight_grad = np.einsum("Bi, ih -> hi", grad, activation_grad.T @ next_output)
    return weight_grad
