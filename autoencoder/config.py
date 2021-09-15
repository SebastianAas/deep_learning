# Name of dataset
# Possible datasets: mnist, mnist_fashion, cats_vs_dogs, cars196
name_of_dataset = "fashion_mnist"

learning_rate = 0.001
loss_function = "mse"
optimizer = "adam"
latent_vector_size = 64
freeze_weights = True

epochs = 10
train_test_split = 0.7
number_of_reconstructions = 10
tSNE_plots = False

neural_network =[
    {
        'type': 'input',
        'size': num_bits
    },
    {
        'type': 'dense',
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
