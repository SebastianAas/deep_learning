import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input

import config


class NeuralNetwork(Model):
    def __init__(self,image_shape, num_labels):
        super(NeuralNetwork, self).__init__()
        self.flatten = Flatten(input_shape=image_shape)
        self.d1 = Dense(128, activation='relu')
        self.d2 = layers.Dense(64, activation='relu')
        self.d3 = Dense(num_labels, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)



def get_neural_network(image_shape, num_labels):
    input = layers.Input(shape=image_shape)
    x = layers.Flatten()(input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(config.latent_vector_size, activation='relu')(x)
    classifier = layers.Dense(num_labels, activation='relu')(x),
    return Model(input, classifier)




