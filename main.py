import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from SolverNN import SolverNN

# Parameters
n_samples = 1
test_set_size = 0.3
sudoku_dim = 9*9
n_nodes = [100]

# Load sudoku puzzles and convert them into numpy arrays within dataframe
data = pd.read_csv("sudoku.csv", skiprows=0, nrows=n_samples, dtype=str)
data = data.applymap(lambda x: np.array(list(map(int, x))))

# Select train and test data
data_train, data_test = train_test_split(data, test_size=test_set_size)

# Initialise solver
solver = SolverNN(sudoku_dim, sudoku_dim, n_nodes)

# Train

# Save solver state

# Apply solver to test puzzles

# Print / save result

