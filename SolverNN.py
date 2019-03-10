import numpy as np

class SolverNN:
    
    def __init__(self, n_inputs, n_outputs, n_nodes):
        
        # Integer for n_nodes - single input layer
        if isinstance(n_nodes, int):
            self.weights = [
                np.random.rand(n_inputs, n_nodes),
                np.random.rand(n_nodes, n_outputs)
            ]

        # List for n_nodes - multiple input layers
        elif isinstance(n_nodes, list):
            self.weights = [np.random.rand(n_inputs, n_nodes[0])]
            for i in range(1, len(n_nodes)):
                self.weights.append( np.random.rand(n_nodes[i-1], n_nodes[i]) )
            self.weights.append( np.random.rand(n_nodes[-1], n_outputs) )

        # Input validation
        else:
            raise TypeError("Parameter n_nodes must be an integer or a list of integers.")
        
        if not isinstance(n_inputs, int) or not isinstance(n_outputs, int):
            raise TypeError("Parameters n_inputs and n_outputs must be integers.")

    def _feedforward(self):
        """Performs feedforward with the neural network to provide output layer values."""
        pass

    def _error(self):
        """Returns a measure of the output layer error for a given input-output combination."""
        pass

    