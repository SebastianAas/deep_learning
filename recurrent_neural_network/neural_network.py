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

    def backward(self, grads):
        for grad in reversed(grads):
            for layer in reversed(self.layers):
                grad = layer.backward(grad)

    def fit(self, inputs, targets, validation_inputs, validation_targets, epochs, batch_size, verbose):
        epoch_losses = np.zeros(epochs)
        validation_loss = np.zeros(epochs)
        for epoch in range(epochs):
            batch_losses = []
            for (input_batch, target_batch) in batch_iterator(batch_size, inputs, targets):
                grads = []
                for i in range(len(input_batch)):
                    predicted = self.forward(input_batch[i])
                    grads.append(self.loss.grad(predicted, target_batch[i]))
                    if verbose:
                        print("Pred: ", predicted)
                        #print("Target: ", target_batch[i].shape)
                        #print("Grad: ", grad)
                batch_loss = self.loss.loss(predicted, target_batch[i])
                batch_losses.append(batch_loss)
                self.backward(grads)
                self.update()
                self.reset_cache()
            epoch_losses[epoch] = np.mean(np.array(batch_losses))
            val_loss = self.calculate_validation_loss(validation_inputs, validation_targets)
            validation_loss[epoch] = val_loss
            print("Epoch {}, Loss {}".format(epoch, epoch_losses[epoch]))
            print("Validation Loss {}".format(val_loss))


        plt.plot([i for i in range(epochs)], epoch_losses, label="Training loss")
        plt.plot([i for i in range(epochs)], validation_loss, label="Validation loss")
        plt.legend()
        plt.savefig("plots/training_validation_loss.png")

    def predict(self, input):
        return self.forward(input)[-1]

    def update(self):
        for layer in self.layers:
            layer.update()


    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    def calculate_validation_loss(self, inputs, targets):
        val_loss = np.zeros(len(inputs))
        for i in range(len(inputs)):
            prediction = self.forward(inputs[i])
            val_loss[i] = self.loss.loss(prediction, targets[i])
        self.reset_cache()
        return np.mean(val_loss)

    def evaluate(self, inputs, targets):
        acc = 0
        np.set_printoptions(precision=4)
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            print("Pred: {} \n "
                  "Target: {} \n".format(prediction, targets[i]))
        print("Accuracy: ", acc * 100, " %")
