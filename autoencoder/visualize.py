import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def show_reconstructions(num_reconstructions, actual, decoded_images):
    plt.figure(figsize=(20, 4))
    for i in range(num_reconstructions):
        # display original
        ax = plt.subplot(2, num_reconstructions, i + 1)
        plt.imshow(actual[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, num_reconstructions, i + 1 + num_reconstructions)
        plt.imshow(decoded_images[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def tsne_plot(data):
    plt.figure(figsize=(16, 10))
    plt.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        data=data,
        legend="full",
        alpha=0.3
    )
