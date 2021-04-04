# Name of dataset
# Possible datasets: mnist, fashion_mnist, kmnist, emnist
name_of_dataset = "emnist"
num_samples = 40000
train_test_split = 0.8
batch = True
batch_size = 32
shuffle = False
fc_size = 0.6

autoencoder_lr = 0.001
autoencoder_loss_decoder = "binary_cross_entropy"
autoencoder_loss_classifier = "sparse_cross_entropy"
autoencoder_optimizer = "adam"
latent_vector_size = 8
freeze_weights = True

classifier_lr = 0.001
classifier_loss = "sparse_cross_entropy"
classifier_optimizer = 'adam'

autoencoder_epochs = 2
classifier_epochs = 2
number_of_reconstructions = 10
tSNE_plots = False
tsne_samples = 200
show_tsne = True

