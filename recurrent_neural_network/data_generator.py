import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def generate_dataset(size, sequence_length, num_bits):
    patterns = [-2, -1, 1, 2]
    inputs = []
    targets = []
    for _ in range(size):
        pattern = np.random.choice(patterns)
        sequence = np.zeros((sequence_length, num_bits))
        target = np.zeros((sequence_length, num_bits))
        bit_pattern = np.random.randint(2, size=num_bits)
        for i in range(sequence_length):
            sequence[i] = bit_pattern
            bit_pattern = np.roll(bit_pattern, pattern)
            target[i] = bit_pattern
        inputs.append(sequence)
        targets.append(target)
    return inputs, targets


def split_dataset(data, val_size, test_size):
    df = np.array(data)
    train, validate, test = np.split(df,
                                     [int(1 - (val_size + test_size) * len(df)), int(1 - test_size * len(df))])
    return train, validate, test


def visualize_data(data):
    colors = ['black', 'white']
    for i in range(0, min(len(data) - 1, 10)):
        plt.imshow(data[i], cmap='Oranges')
        plt.savefig("examples/sequence-{}.png".format(i))


def batch_iterator(batch_size, inputs, targets):
    size = len(inputs)
    splits = np.arange(0, size, batch_size)
    for start in splits:
        end = start + batch_size
        batch_inputs = np.array(inputs)[start:end].transpose((1, 0, 2))
        batch_targets = np.array(targets)[start:end].transpose((1, 0, 2))
        if batch_inputs.shape[1] == batch_size:
            yield batch_inputs, batch_targets
        else:
            continue


if __name__ == '__main__':
    inputs, targets = generate_dataset(size=6, sequence_length=10, num_bits=10)
    train, val, test = split_dataset(inputs, 0.2, 0.2)
    train_t, val_t, test_t = split_dataset(targets, 0.2, 0.2)
    visualize_data(inputs)
    for (a, b) in batch_iterator(4, train, train_t):
        print("A: ", a)
        print("B: ", b)
