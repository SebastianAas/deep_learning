
# Data generator
dataset_size = 200
sequence_length = 8
num_bits = 10

# Neural Network
verbose = True
learning_rate = 0.0001
epochs = 10
batch_size = 32
loss = 'mse'
layer_config = [
    {
        'type': 'input',
        'size': num_bits
    },
    {
        'type': 'recurrent',
        'size': 64,
        'activation': 'relu',
        'learning_rate': 0.0001,
        'weight_range': (-0.1, 0.1)
    },
    {
        'type': 'dense',
        'size': num_bits,
        'activation': 'relu',
        'learning_rate': 0.0001
    }
]
