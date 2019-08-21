import matplotlib.pyplot as plt
import autograd.numpy as np


def plot_values(matrices, max_cols=15):
    """Plots a grid of histograms from an array of matrices"""
    fig = plt.figure(figsize=(30, 10))
    n = len(matrices)
    cols = max_cols

    while n % cols != 0:
        cols -= 1

    fig_rows = n / cols

    for i, m in enumerate(matrices):
        _ = fig.add_subplot(fig_rows, cols, i + 1)
        _.hist(m.flatten())

    plt.show()


if __name__ == "__main__":
    test = [np.random.rand(4, 2) for _ in range(6)]
    plot_values(test)
