from problem_generation.Generating_Problems import Problem
from utils.Matrix import Matrix
import numpy as np


class Knapsack(Problem):
    """Knapsack problem generator."""

    def __init__(self, maxWeight, weights, values, seed: int=42, A: int=1, B: int=1):
        # check if the weights and values are the same length (every object needs both)
        if len(weights) != len(values):
            raise ValueError("The number of weights and values should be the same")

        super().__init__(seed=seed)
        self.maxWeight = maxWeight
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.num_items = len(weights)
        self.A = A
        self.B = B
        # todo: how to set the position_translater?
        self.position_translater = None

        # Pre-compute vectors for ease of matrix operations
        self.vec_a = np.ones(self.maxWeight)
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

        # x_indices are 1 to num_items
        # y_indices are num_items + 1 to num_items + maxWeight
        x_indices = range(1, self.num_items + 1)
        y_indices = range(self.num_items + 1, num_variables + 1)
        smallest_y = self.num_items + 1

        # A(1-2ay+f_Y(a)) diagonal is zero because A-2A+A=0
        # A*f_y(n)
        for i in y_indices:
            self.matrixClass.add_diag_element(i, self.A * self.vec_n[i - smallest_y] ** 2)
            for j in y_indices:
                if i != j:
                    self.matrixClass.add_off_element(
                        i, j, 2 * self.A + self.A * self.vec_n[i - smallest_y] * self.vec_n[j - smallest_y]
                    )

        # -2n(yw)x
        for i in y_indices:
            for alpha, x in enumerate(x_indices):
                self.matrixClass.add_off_element(
                        i, x, -2 * self.A * self.vec_n[i - smallest_y] * self.weights[alpha]
                )

        # A(f_x(w))
        for i in x_indices:
            self.matrixClass.add_diag_element(i, self.A * self.weights[i - 1] ** 2)
            for j in x_indices:
                if i != j:
                    self.matrixClass.add_off_element(i, j, 2 * self.A * self.weights[i - 1] * self.weights[j - 1])

        # Construct H_B
        for alpha, x in enumerate(x_indices):
            self.matrixClass.add_diag_element(x, -self.B * self.values[alpha])

        # I think that is not sufficient because it doesnt give any information what x and y is in the matrix
        self.position_translater = [0] + list(x_indices) + list(y_indices)

        return self.matrix