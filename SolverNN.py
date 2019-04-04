"""
SolverNN.py

Module for the sudoku solver class.

The solving method is a feedforward neural network. Node weights are represented in a list of numpy
arrays. Each element in the list represents the weights for a layer. A set of weights is an n by m
numpy array where n is the number of nodes in the previous (upstream) layer and m is the number of
nodes in the current layer. Hence, each column represents the weights for a given neuron in the
layer.

"""

import numpy as np

class SolverNN:
    
    def __init__(self, n_inputs, n_outputs, n_nodes):

        # TODO init by loading parameters from file

        # Input validation
        if not isinstance(n_inputs, int) or not isinstance(n_outputs, int):
            raise TypeError("Parameters n_inputs and n_outputs must be integers.")
        if not isinstance(n_nodes, (int, list)):
            raise TypeError("Parameter n_nodes must be an integer or a list of integers.")

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = len(n_nodes)
        self.n_nodes = [n_nodes] if isinstance(n_nodes, int) else n_nodes

        # Randomly initialise weights
        self.weights = [np.random.rand(n_inputs, n_nodes[0])]
        for i in range(1, len(n_nodes)):
            self.weights.append( np.random.rand(n_nodes[i-1], n_nodes[i]) )
        self.weights.append( np.random.rand(n_nodes[-1], n_outputs) )

    def train(self, puzzle, solution):
        """
        TODO description
        
        """

        # TODO input validation

        output = self._feedforward(puzzle)
        # output_err = self._error(output, solution)
        # self._backpropagation(output_err)

        return output
        
    def _feedforward(self, puzzle):
        """Performs feedforward with the neural network to provide output layer values."""
        # TODO consider adding bias nodes later
        
        current_layer = puzzle
        for layer in range(0, self.n_layers):
            current_layer = self._activation( np.dot( current_layer, self.weights[layer] ))

        return current_layer        

    def _activation(self, layer):
        """Neuron activation function. Currently using sigmoid."""        
        return 1 / (1 + np.exp(-layer))

    def _error(self, output, solution):
        """Returns a measure of the output layer error for a given input-output combination."""
        pass

    def _backpropagation(self, output_err):
        pass

    