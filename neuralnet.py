"""
neuralnet.py
An implementation of a Deep Feed Forward
Neural Network in python.

@author: Michael Samon
"""

import numpy as np

# custom-nn imports
from utils import NeuralError


class Layer:
    """
    Handles matrix shaping and initialization.
    """
    def __init__(self, input_shape=None):
        self.INPUT_SHAPE = input_shape
        self.training = False

        # list containing layer widths in order of graph definition
        self.layers = []

    def get_matrix_shapes(self):
        """
        Return a list of all layer shapes with the input matrix
        shape at the start index and the output matrix shape
        at the end index.
        """

        # input_shape must have been set by this point
        if self.INPUT_SHAPE is None:
            raise NeuralError("'input_shape' was found to be None")

        shapes = []

        if len(self.layers) < 1:
            raise NeuralError(
                "Not enough layers to initialize graph",
                data=self.layers)

        for i, width in enumerate(self.layers):
            if i == 0:  # input matrix
                shape = (self.INPUT_SHAPE[0], width)
                shapes.append(shape)
            else:
                # set columns of preceding matrix as row
                # size of current matrix
                shape = (shapes[i-1][1], width)
                shapes.append(shape)

        return shapes


if __name__ == "__main__":
    network = Layer()
    network.layers = [6, 6, 1]
    print(network.get_matrix_shapes())
