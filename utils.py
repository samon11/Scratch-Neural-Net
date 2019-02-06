"""
utils.py
Error definitions and utility functions.

@author: Michael Samon
"""


class NeuralError(Exception):
    """ Neural Network class errors. """
    def __init__(self, message, data=None):
        self.message = message
        self.data = data

    def __str__(self):
        error_data = "::" + repr(self.data) if self.data is not None else ""
        error_message = self.message + error_data
        return error_message
