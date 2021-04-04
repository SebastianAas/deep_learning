import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import visualize


class Autoencoder(Model):
    def __init__(self, latent_dim, image_shape, num_labels, freeze=False):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(image_shape[0] * image_shape[1], activation='sigmoid'),
            layers.Reshape(image_shape)
        ])
        self.classifier = tf.keras.Sequential([
            layers.Dense(image_shape[0] * image_shape[1], activation='relu'),
            layers.Softmax(),
        ])
        if freeze:
            for layer in self.encoder:
                layer.trainable = False

    def call(self, x):
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return decoded

    def classify(self, x):
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        return classified





