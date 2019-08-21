"""
neuralmath.py
Pre-defines all derivatives and gradients
'd_' is shorthand for 'derivative of'
"""

import autograd.numpy as np


class Activations:
    """Class that handles activation funcs of the neural network"""

    def sigmoid(x, deriv=False):
        if not deriv:
            return (1 / (1 + np.exp(-1 * x)))
        else:
            return np.exp(-1 * x) / ((1 / (1 + np.exp(-1 * x))) ** 2)

    def relu(x, deriv=False):
        if not deriv:
            return np.maximum(0, x)
        else:
            return (x > 0) * 1

    def tanh(x, deriv=False):
        if not deriv:
            return np.tanh(x)
        else:
            return 1 - (np.tanh(x) ** 2)


if __name__ == "__main__":
    math = Activations()
    a = np.random.rand(2, 2)
    print(Activations.d_relu(a))
