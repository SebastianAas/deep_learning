import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

import config


def get_data(name_of_dataset, num_samples, split):
    (ds_train, ds_test), ds_info = tfds.load(
        name_of_dataset,
        split=['train[0:{}]'.format(int(split * num_samples)), 'train[{}:{}]'.format(int(split * num_samples) + 1, num_samples)],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (ds_train, ds_test), ds_info


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def remove_label(image, label):
    return image, image


def unlabel_data(data):
    return data.map(
        remove_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def preprocess_data(data, shuffle=False, batch=True):
    data = data.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    if shuffle:
        data = data.shuffle(tf.data.experimental.cardinality(data).numpy())
    if batch:
        data = data.batch(32, drop_remainder=True)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data

