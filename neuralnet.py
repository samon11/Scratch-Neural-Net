"""
neuralnet.py
An implementation of a Deep Feed Forward
Neural Network framework in python.

@author: Michael Samon
"""

import autograd.numpy as np
from autograd import grad

# custom-nn imports
from utils import NeuralError
from neuralmath import Activations


class Network:
    """
    Handles matrix shaping and initialization.
    """
    def __init__(self):
        self.training = False
        self.compiled = False
        self.loss = 'mse'
        self.gradients = []

        # list containing layer objects in order of graph definition
        self.layers = []

        # weights of layers in sequential order
        self.layer_weights = []

        # 

        # list containing the successive shapes of network layers
        self.shapes = []

        # network biases to be initialized
        self.biases = None

    def get_matrix_shapes(self):
        """
        Computes a list of all layer shapes with the input matrix
        shape at the start index and the output matrix shape
        at the end index.
        """

        if len(self.layers) < 1:
            raise NeuralError(
                "Not enough layers to initialize graph, found",
                data=len(self.layers))

        for i, layer in enumerate(self.layers):
            # input matrix
            if i == 0:
                if layer.INPUT_SHAPE is None:
                    raise NeuralError(
                        "input_shape not specified for first layer got None"
                        )

                shape = (layer.INPUT_SHAPE[0] + 1, layer.width)  # +1 for bias
                self.shapes.append(shape)

            else:
                # set columns of preceding matrix as row
                # size of current matrix
                shape = (self.shapes[i-1][1], layer.width)
                self.shapes.append(shape)

    def add(self, layer, activation=None):
        """Add layer to the network graph."""

        # check that a layer object was passed in
        try:
            getattr(layer, "width")
        except AttributeError:
            raise NeuralError(
                "'layer' argument is not a layer object instead got",
                data=type(layer))

        self.layers.append(layer)

        # index 0 represents the input matrix
        layer.layer_index = len(self.layers) - 1

    def compile_graph(self):
        """Initialize layer matrices and their weights."""
        self.get_matrix_shapes()
        for layer in self.layers:
            shape = self.shapes[layer.layer_index]
            random_matrix = np.random.rand(shape[0], shape[1])
            layer.weights = random_matrix
            self.layer_weights.append(layer.weights)

        self.compiled = True

    def forward_pass(self, x, weights):
        """
        Input a batch of data and return the output of the
        compiled graph.
        """

        if not self.compiled:
            raise NeuralError(
                "Network graph not compiled. Must run 'compile_graph()' first."
                )

        z_s = []
        a_s = []

        # concatenate the bias to the input data
        if self.biases is None:
            init_biases = np.ones((x.shape[0], 1))
            indata = np.concatenate((x, init_biases), axis=1)
        else:
            indata = np.concatenate((x, self.biases), axis=1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                z = np.dot(indata, weights[i])
                a = layer.activation(z)

                z_s.append(z)
                a_s.append(a)
                continue

            z = np.dot(a, weights[i])
            a = layer.activation(z)

            z_s.append(z)  # values pre-activation
            a_s.append(a)  # values post-activation

        return z_s, a_s

    def compute_loss(self, x, y, weights):
        """Get loss from one forward pass of the input"""
        if self.loss == 'mse':
            z, a = self.forward_pass(x, weights)
            y_pred = a[-1]

            loss = np.mean((y - y_pred)**2)
            print("Network Loss:", loss._value)
            return loss

    gradients_op = grad(compute_loss, 3)

    def back_prop(self, x, y, lr=0.01):
        """Back-propogate and make a weight update"""

        gradients = self.gradients_op(x, y, self.layer_weights)
        epoch_gradients = []
        for i, weights in enumerate(self.layer_weights):
            self.layer_weights[i] -= gradients[i] * lr
            epoch_gradients.append(gradients[i] * lr)

        self.gradients.append(epoch_gradients)

    def train(self, x, y, epochs=10, lr=0.01):
        """Run `epochs` amount of gradient updates"""

        for _ in range(epochs):
            self.back_prop(x, y, lr=lr)


class Dense:
    """Class that defines Dense network layer attributes."""

    def __init__(self, width, activation=None, input_shape=None):
        self.INPUT_SHAPE = input_shape

        self.width = width

        if activation is not None:
            self.activation = getattr(Activations, activation)
        else:
            # inline do nothing function for None activation
            self.activation = lambda x: x

        # layer index in the graph
        self.layer_index = 0

        # tuple containing the layer's weight matrix shape
        self.shape = None

        # weights matrix
        self.weights = None


if __name__ == "__main__":
    network = Network()
    network.add(Dense(4, input_shape=(3,), activation="tanh"))
    network.add(Dense(3, activation="tanh"))
    network.add(Dense(1, activation="relu"))
    network.compile_graph()  # initialize variables into memory

    input_data = np.random.rand(32, 3)
    y = np.random.rand(32, 1)

    network.train(input_data, y, epochs=50)
