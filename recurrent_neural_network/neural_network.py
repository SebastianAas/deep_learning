import numpy as np
import matplotlib.pyplot as plt

from data_generator import batch_iterator
from layers import Layer


class NeuralNetwork:

    def __init__(self, layers, loss, learning_rate):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, inputs, targets, validation_inputs, validation_targets, epochs, batch_size, verbose):
        epoch_losses = np.zeros(epochs)
        validation_loss = np.zeros(epochs)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (input_batch, target_batch) in batch_iterator(batch_size, inputs, targets):
                for input in input_batch:
                    predicted = self.forward(input)
                    epoch_loss += self.loss.loss(predicted, target_batch)
                    grad = self.loss.grad(predicted, target_batch)
                    if verbose:
                        print("Pred: ", predicted)
                        print("Target: ", target_batch)
                        print("Grad: ", grad)
                    self.backward(grad)
                self.update()
                self.reset_weights()
            val_loss = self.calculate_validation_loss(validation_inputs, validation_targets)
            print("Epoch {}, Loss {}".format(epoch, epoch_loss))
            print("Validation Loss {}".format(val_loss))
            epoch_losses[epoch] = epoch_loss
            validation_loss[epoch] = val_loss
            plt.plot(epoch_losses)
            plt.plot(validation_loss)

    def predict(self, input):
        return self.forward(input)[-1]

    def update(self):
        for layer in self.layers:
            layer.update()

    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()

    def calculate_validation_loss(self, inputs, targets):
        val_loss = 0
        for i in range(len(inputs)):
            prediction = self.forward(inputs[i])
            val_loss += self.loss.loss(prediction, targets[i])
        return val_loss

    def evaluate(self, inputs, targets):
        acc = 0
        np.set_printoptions(precision=4)
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            print("Pred: {} \n "
                  "Target: {} \n".format(prediction, targets[i]))
        print("Accuracy: ", acc * 100, " %")
