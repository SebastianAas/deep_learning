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
    def __init__(self, latent_dim, image_shape, num_labels, loss_decoder, loss_classifier, optimizer):
        super(Autoencoder, self).__init__()
        self.loss_decoder = loss_decoder
        self.loss_classifier = loss_classifier
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.optimizer = optimizer
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
            layers.Dense(num_labels, activation='relu'),
            layers.Softmax(),
        ])

    def freeze_encoder_weight(self):
        for layer in self.encoder:
            layer.trainable = False

    def unfreeze_encoder_weight(self):
        for layer in self.encoder:
            layer.trainable = True

    def call(self, x):
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return decoded, classified

    def encode(self, x):
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        return classified

    def classify(self, x):
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        return classified

@tf.function
def train_decoder(images, model):
    with tf.GradientTape() as tape:
        decoder_predictions, classification_predictions = model(images, training=True)
        loss_decoder = model.loss_decoder(decoder_predictions, images)
    gradients = tape.gradient(loss_decoder, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def train_classifier(images, labels, model):
    with tf.GradientTape() as tape:
        decoder_predictions, classification_predictions = model(images, training=True)
        loss_classifier = model.loss_classifier(classification_predictions, labels)
    gradients = tape.gradient(loss_classifier, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train(EPOCHS, data, model):
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch

        for images, labels in data:
            train_decoder(images, model)
            train_classifier(images, labels, model)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {model.train_loss.result()}, '
            f'Accuracy: {model.train_accuracy.result() * 100}, '
        )




def get_autoencoder_models( latent_dim, image_shape, num_labels, optimizer, loss_decoder, loss_classifier):
    input = layers.Input(shape=image_shape)
    x = layers.Flatten()(input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    encoder = layers.Dense(latent_dim, activation='relu')(x)

    decoder = layers.Dense(image_shape[0] * image_shape[1], activation='sigmoid')(encoder)
    decoder = layers.Reshape(image_shape)(decoder)

    classifier = layers.Dense(num_labels, activation='relu')(encoder),

    encoder_model = Model(input, encoder)
    encoder_model.compile(
        optimizer=optimizer,
        loss=loss_decoder,
        metrics=['accuracy']
    )

    decoder_model = Model(input, decoder)
    decoder_model.compile(
        optimizer=optimizer,
        loss=loss_decoder,
        metrics=['accuracy']
    )

    classifier_model = Model(input, classifier)
    classifier_model.compile(
        optimizer=optimizer,
        loss=loss_classifier,
        metrics=["accuracy"]
    )

    return encoder_model, decoder_model, classifier_model
