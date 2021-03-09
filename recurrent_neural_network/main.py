from data_generator import generate_dataset, split_dataset
from neural_network import NeuralNetwork
from config_files.config import *
from config_parser import *

inputs, targets = generate_dataset(dataset_size, sequence_length, num_bits)
train_inputs, val_inputs, test_inputs = split_dataset(inputs, 0.15, 0.15)
train_targets, val_targets, test_targets = split_dataset(targets, 0.15, 0.15)

layers = get_layers(layer_config)

nn = NeuralNetwork(
    layers=layers,
    loss=get_loss_function(loss),
    learning_rate=learning_rate,
)

nn.fit(
    inputs=inputs,
    targets=targets,
    validation_inputs=val_inputs,
    validation_targets=val_targets,
    epochs=epochs,
    batch_size=batch_size,
    verbose=verbose
)
nn.evaluate(test_inputs, test_targets)
