
# Data generator
dataset_size = 128
sequence_length = 15
num_bits = 10

# Neural Network
learning_rate = 0.001
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
        'size': 20,
        'activation': 'relu',
        'learning_rate': 0.001
    },
    {
        'type': 'dense',
        'size': 20,
        'activation': 'relu',
        'learning_rate': 0.001
    },
    {
        'type': 'dense',
        'size': num_bits,
        'activation': 'linear',
        'learning_rate': 0.001
    }
]
