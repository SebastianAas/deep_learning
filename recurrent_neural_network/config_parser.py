from activations import *
from loss import *
from layers import *

def get_loss_function(str):
    if str == "mse":
        return MSE()
    else:
        raise NotImplementedError


def get_activation_function(config):
    if "activation" in config:
        act = config["activation"]
    else:
        return Linear()
    if act == "relu":
        return Relu()
    elif act == "sigmoid":
        return Sigmoid()
    elif act == "tanh":
        return Tanh()
    else:
        return Linear()


def get_layers(config):
    default_learning_rate = 0.001
    default_weight_range = (-0.1, 0.1)
    config_layers = []
    for i in range(len(config)):
        type = config[i]["type"]
        input = config[i]["size"]
        if type == "input":
            config_layers.append(Input(input))
            continue
        input = config[i - 1]["size"]
        output = config[i]["size"]
        if type == "recurrent":
            activation = get_activation_function(config[i])
            learning_rate = get_value(config[i], "learning_rate", default_learning_rate)
            weight_range = get_value(config[i], "weight_range", default_weight_range)
            config_layers.append(
                RNN(input_size=input, output_size=output, activation=activation, learning_rate=learning_rate, weight_range=weight_range))
        else:
            activation = get_activation_function(config[i])
            learning_rate = get_value(config[i], "learning_rate", default_learning_rate)
            weight_range = get_value(config[i], "weight_range", default_weight_range)
            config_layers.append(
                Dense(input_size=input, output_size=output, activation=activation, learning_rate=learning_rate, weight_range=weight_range))
    return config_layers

def get_value(config, key, default):
    if key in config:
        return config[key]
    else:
        return default
