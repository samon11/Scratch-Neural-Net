"""
neuralnet.py
An implementation of a Deep Feed Forward
Neural Network framework in python.

@author: Michael Samon
"""

import numpy as np

# custom-nn imports
from utils import NeuralError


class Network:
    """
    Handles matrix shaping and initialization.
    """
    def __init__(self):
        self.training = False
        self.compiled = False

        # list containing layer objects in order of graph definition
        self.layers = []

        # list of supported activations
        self.activations = ['tanh', 'sigmoid', 'relu']

        # list containing the successive shapes of network layers
        self.shapes = []

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

                shape = (layer.INPUT_SHAPE[0], layer.width)
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

            #self.weight_matrices.append(random_matrix)
        self.compiled = True

    def forward_pass(self, x):
        """
        Input a batch of data and return the output of the
        compiled graph.
        """

        if not self.compiled:
            raise NeuralError(
                "Network graph not compiled. Must run 'compile_graph()' first."
                )
        
        output = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                output = np.dot(x, layer.weights) + layer.b
    
            output = np.dot(output, layer.weights) + layer.b

        return output


class Dense:
    """Class that defines Dense network layer attributes."""

    def __init__(self, width, activation=None, input_shape=None):
        self.INPUT_SHAPE = input_shape

        self.width = width
        self.activation = activation

        # layer index in the graph
        self.layer_index = 0

        # tuple containing the layer's weight matrix shape
        self.shape = None

        # weights matrix
        self.weights = None

        # set default bias value
        self.b = 0


if __name__ == "__main__":
    network = Network()
    network.add(Dense(6, input_shape=(6,)))
    network.add(Dense(16))
    network.add(Dense(1))
    network.compile_graph()

    input_data = np.random.rand(32, 6)
    preds = network.forward_pass(input_data)
    print(preds)
    print(preds.shape)
