from problem_generation.Generating_Problems import Problem
from utils.Matrix import Matrix
import numpy as np


class Knapsack(Problem):
    """Knapsack problem generator."""

    def __init__(self, maxWeight, weights, values, seed: int=42, A: int=1, B: int=1):
        # check if the weights and values are of the same length (every object needs both)
        if len(weights) != len(values):
            raise ValueError("The number of weights and values should be the same")
        if maxWeight < 0:
            raise ValueError("The maximum weight should be non-negative")

        super().__init__(seed=seed)
        self.maxWeight = maxWeight
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.num_items = len(weights)
        self.A = A
        self.B = B
        self.position_translater = None

        # Pre-compute vector n
        self.vec_n = np.arange(1, self.maxWeight + 1)

        # Initialize the Hamiltonian matrix
        self.matrix = self.knapsack_to_matrix()

    def knapsack_to_matrix(self):
        """Constructs the Hamiltonian matrix for the knapsack problem."""

        # Total number of variables: items + weights
        num_variables = self.num_items + self.maxWeight

        # Create the matrix class
        self.matrixClass = Matrix(num_variables + 1)
        self.matrix = self.matrixClass.matrix

        x_indices = range(1, self.num_items + 1)
        y_indices = range(self.num_items + 1, num_variables + 1)

        for i in y_indices:
            self.matrixClass.add_diag_element(i, -self.A) # A * (1 - 2 * a^T y)= A - 2 * A = -A
            for j in y_indices:
                if i != j:
                    self.matrixClass.add_off_element(i, j, 2 * self.A) # A * (1 + y^T (a^T a) y) = A + A = 2 * A
                else:
                    self.matrixClass.add_diag_element(j, 2* self.A) # A * (1 + y^T (a^T a) y) = A + A = 2 * A

        # A * y^T (n n^T) y
        for i in y_indices:
            for j in y_indices:
                if i != j:
                    vec_n_i = self.vec_n[i - (self.num_items + 1)]
                    vec_n_j = self.vec_n[j - (self.num_items + 1)]
                    self.matrixClass.add_off_element(i, j, self.A * vec_n_i * vec_n_j)
                else:
                    vec_n_j = self.vec_n[j - (self.num_items + 1)]
                    self.matrixClass.add_diag_element(j, self.A * vec_n_j * vec_n_j)

        # -2 * y^T (n w^T) x
        for i in y_indices:
            for j in x_indices:
                self.matrixClass.add_off_element(i, j, -2 * self.vec_n[i - self.num_items - 1] * self.weights[j - 1])

        # A * x^T (w w^T) x
        for i in x_indices:
            for j in x_indices:
                if i != j:
                    self.matrixClass.add_off_element(i, j, 2 * self.A * self.weights[i - 1] * self.weights[j - 1])
                else:
                    self.matrixClass.add_diag_element(i, self.A * self.weights[i - 1] * self.weights[j - 1])

        # Construct H_B
        for i, x in enumerate(x_indices):
            self.matrixClass.add_diag_element(x, -self.B * self.values[i])

        self.position_translater = [0] + list(x_indices)

        return self.matrix