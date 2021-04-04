import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime

from tensorflow.keras import layers, losses

import visualize
from autoencoder import Autoencoder, get_autoencoder_models
import config
from config_parser import ConfigParser
from neural_network import NeuralNetwork, get_neural_network
from preprocess import get_data, preprocess_data, remove_label, unlabel_data

(d1, d2), ds_info = get_data(config.name_of_dataset, config.num_samples, config.train_test_split)

image_shape = ds_info.features["image"].shape
num_labels = ds_info.features["label"].num_classes

d1_processed = preprocess_data(d1, shuffle=config.shuffle)
d1_train = unlabel_data(d1_processed)
d2_test = unlabel_data(preprocess_data(d2))

fc_size = config.fc_size * tf.data.experimental.cardinality(d2).numpy()

fc = d2.take(fc_size)
remaining = d2.skip(fc_size)

fc_train = preprocess_data(fc)
remaining_test = preprocess_data(remaining)

print(ds_info)

print("Len d1: ", tf.data.experimental.cardinality(d1).numpy())
print("Len d2: ", tf.data.experimental.cardinality(d2).numpy())
print("Len fc: ", tf.data.experimental.cardinality(fc).numpy())
print("Len d2 - fc: ", tf.data.experimental.cardinality(remaining).numpy())

encoder_model, decoder_model, classifier_model = get_autoencoder_models(
    optimizer=ConfigParser.get_optimizer(config.autoencoder_optimizer, config.autoencoder_lr),
    latent_dim=config.latent_vector_size,
    image_shape=image_shape,
    num_labels=num_labels,
    loss_decoder=ConfigParser.get_loss(config.autoencoder_loss_decoder),
    loss_classifier=ConfigParser.get_loss(config.autoencoder_loss_classifier)
)

nn = get_neural_network(image_shape=image_shape, num_labels=num_labels)

nn.compile(
    optimizer=ConfigParser.get_optimizer(config.classifier_optimizer, config.classifier_lr),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

labels = np.concatenate([x[1].numpy() for x in d1_processed]).ravel().tolist()[0:config.tsne_samples]
predictions = encoder_model.predict(d1_train, config.batch_size)[0:config.tsne_samples]
visualize.tsne_plot(predictions, labels, "before_training", config.name_of_dataset, verbose=config.show_tsne)

history_decoder = decoder_model.fit(
    d1_train,
    epochs=config.autoencoder_epochs,
    batch_size=32,
    validation_batch_size=config.batch_size,
    validation_data=d2_test)

visualize.show_reconstructions(d2_test, decoder_model, config.number_of_reconstructions)

predictions = encoder_model.predict(d1_train, config.batch_size)[0:config.tsne_samples]
visualize.tsne_plot(predictions, labels, "after_training_decoder", config.name_of_dataset, verbose=config.show_tsne)

if config.freeze_weights:
    encoder_model.trainable = False

history_auto_classifier = classifier_model.fit(
    fc_train,
    epochs=config.autoencoder_epochs,
    batch_size=config.batch_size,
    validation_batch_size=32,
    validation_data=remaining_test)
test_loss, test_acc = classifier_model.evaluate(remaining_test, verbose=2)

print('\nTest accuracy autoencoder:', test_acc)

history_classifier = nn.fit(
    fc_train,
    epochs=config.classifier_epochs,
    batch_size=config.batch_size,
    validation_batch_size=config.batch_size,
    validation_data=remaining_test)

test_loss, test_acc = nn.evaluate(remaining_test, verbose=2)

print('\nTest accuracy supervised:', test_acc)

print("\nTEST SCORES: ")
test_loss, test_acc = classifier_model.evaluate(d1_processed)
print('\nTest accuracy classifier from autoencoder on D1:', test_acc)
test_loss, test_acc = nn.evaluate(d1_processed)
print('\nTest accuracy supervised on D1:', test_acc)

visualize.plot_loss([("decoder", history_decoder)], "autoencoder_loss")
visualize.plot_loss(
    [("autoencoder_classifier", history_auto_classifier), ("supervised classifier", history_classifier)],
    "{}/classifiers_loss".format(config.name_of_dataset))

visualize.plot_accuracy([("decoder", history_decoder)], "autoencoder_accuarcy")
visualize.plot_accuracy(
    [("autoencoder_classifier", history_auto_classifier), ("supervised classifier", history_classifier)],
    "{}/classifiers_accuracy".format(config.name_of_dataset))

predictions = encoder_model.predict(d1_train, config.batch_size)[0:config.tsne_samples]
visualize.tsne_plot(predictions, labels, "after_training_classifier", config.name_of_dataset, verbose=config.show_tsne)
