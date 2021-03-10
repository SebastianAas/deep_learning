
# Data generator
dataset_size = 100
sequence_length = 8
num_bits = 10

# Neural Network
verbose = False
learning_rate = 0.0001
epochs = 20
batch_size = 4
loss = 'mse'
layer_config = [
    {
        'type': 'input',
        'size': num_bits
    },
    {
        'type': 'recurrent',
        'size': 64,
        'activation': 'tanh',
        'learning_rate': 0.0001
    },
    {
        'type': 'recurrent',
        'size': 64,
        'activation': 'tanh',
        'learning_rate': 0.0001
    },
    {
        'type': 'recurrent',
        'size': 64,
        'activation': 'tanh',
        'learning_rate': 0.0001
    },
    {
        'type': 'dense',
        'size': num_bits,
        'activation': 'tanh',
        'learning_rate': 0.0001
    }
]
