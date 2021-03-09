
# Data generator
dataset_size = 100
sequence_length = 1
num_bits = 8

# Neural Network
verbose = True
learning_rate = 0.01
epochs = 100
batch_size = 32
loss = 'mse'
layer_config = [
    {
        'type': 'input',
        'size': num_bits
    },
    {
        'type': 'recurrent',
        'size': 100,
        'activation': 'relu',
        'learning_rate': 0.001,
        'weight_range': (-0.1, 0.1)
    },
    {
        'type': 'dense',
        'size': num_bits,
        'activation': 'relu',
        'learning_rate': 0.001
    }
]
