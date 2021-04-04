import matplotlib
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

import config


def show_reconstructions(data, model, num_reconstructions):
    global decoded_images, actual
    for i in data.take(1):
        decoded_images = model.call(i[0])
        actual = i[0]

    plt.figure(figsize=(12, 4))
    for i in range(num_reconstructions):
        # display original
        ax = plt.subplot(2, num_reconstructions, i + 1)
        plt.imshow(actual[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, num_reconstructions, i + 1 + num_reconstructions)
        plt.imshow(decoded_images[i])
        plt.title("reconstruct")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("plots/{}/reconstruction.png".format(config.name_of_dataset))


def plot_loss(histories, file_name):
    plt.figure()
    for (model_name, history) in histories:
        history_dict = history.history
        history_dict.keys()
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(loss) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, label='Training loss ' + model_name)
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, "--", label='Validation loss ' + model_name)

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig("plots/{}.png".format(file_name))


def plot_accuracy(histories, file_name):
    plt.figure()
    for (model_name, history) in histories:
        history_dict = history.history
        history_dict.keys()
        loss = history_dict['accuracy']
        val_loss = history_dict['val_accuracy']

        epochs = range(1, len(loss) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, label='Training accuracy ' + model_name)
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, "--", label='Validation accuracy ' + model_name)

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("plots/{}.png".format(file_name))


def tsne_plot(data, labels, name, name_of_dataset, verbose=False):
    tsne = TSNE(n_components=2, perplexity=50)

    tsne_data = tsne.fit_transform(data)

    plt.figure(figsize=(16, 10))
    plt.title('tSNE plot {}'.format(name))
    plt.scatter(tsne_data[:, 0],tsne_data[:, 1], c=labels, s=200, cmap=plt.get_cmap("hsv"))
    plt.savefig("plots/{}/tsne/tsne_{}.png".format(name_of_dataset, name))
    if verbose:
        plt.show()

