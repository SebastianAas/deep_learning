import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def get_data(name_of_dataset, split):
    (ds_train, ds_test), ds_info = tfds.load(
        name_of_dataset,
        split=['train[0:{}%]'.format(int(split * 100)), 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (ds_train, ds_test), ds_info


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def remove_label(image, label):
    return image


def preprocess_data(data, shuffle=False, batch=False):
    data = data.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    if shuffle:
        data = data.shuffle(data.splits['train'].num_examples)
    if batch:
        data = data.batch(32)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def unlabel_data(data):
    return data.map(
        remove_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def to_numpy(x):
    return np.array(list(x.as_numpy_iterator()))
