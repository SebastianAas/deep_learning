import tensorflow as tf


class ConfigParser:

    @staticmethod
    def get_optimizer(optimizer, learning_rate):
        if optimizer == 'adam':
            return tf.keras.optimizers.Adam(lr=learning_rate)
        elif optimizer == "adagrad":
            return tf.keras.optimizers.Adagrad(lr=learning_rate)
        else:
            return tf.keras.optimizers.RMSprop(lr=learning_rate)

    @staticmethod
    def get_loss(loss):
        if loss == 'sparse_cross_entropy':
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif loss == 'binary_cross_entropy':
            return tf.keras.losses.BinaryCrossentropy()
        else:
            return tf.keras.losses.MeanSquaredError()
