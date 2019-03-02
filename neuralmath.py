"""
neuralmath.py
Pre-defines all derivatives and gradients
'd_' is shorthand for 'derivative of'
"""

import numpy as np


class Activations:
    """Class that handles activation funcs of the neural network"""

    def sigmoid(x):
        return (1 / (1 + np.exp(-1 * x)))

    def d_sigmoid(x):
        return np.exp(-1 * x) / ((1 / (1 + np.exp(-1 * x))) ** 2)

    def relu(x):
        return np.maximum(0, x)

    def d_relu(x):
        return (x > 0) * 1

    def tanh(x):
        return np.tanh(x)

    def d_tanh(x):
        return 1 - (np.tanh(x) ** 2)


if __name__ == "__main__":
    math = Activations()
    print(Activations().__getattribute__("relu"))
    a = np.random.rand(2, 2)
    print(math.d_relu(a))
