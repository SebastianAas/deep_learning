import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses

import visualize
from autoencoder import Autoencoder
import config
from neural_network import NeuralNetwork
from preprocess import get_data, to_numpy, preprocess_data, remove_label, unlabel_data

(ds_train, ds_test), ds_info = get_data(config.name_of_dataset, config.train_test_split);

print(len(ds_train))

x_train = to_numpy(unlabel_data(preprocess_data(ds_train)))
x_test = to_numpy(unlabel_data(preprocess_data(ds_test)))


autoencoder = Autoencoder(latent_dim=config.latent_vector_size, image_shape=(28, 28), num_labels=10)
nn = NeuralNetwork()

autoencoder.compile(optimizer=config.optimizer, loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=15,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.call(x_test).numpy()

visualize.show_reconstructions(config.number_of_reconstructions, actual=x_test, decoded_images=decoded_imgs)
