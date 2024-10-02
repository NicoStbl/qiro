from problem_generation.Generating_Problems import Problem
from utils.Matrix import Matrix
import numpy as np


class Knapsack(Problem):
    """Knapsack problem generator."""

    def __init__(self, maxWeight, weights, values, seed: int=42, A: int=1, B: int=1):
        # check if the weights and values are of the same length (every object needs both)
        if len(weights) != len(values):
            raise ValueError("The number of weights and values should be the same")

        super().__init__(seed=seed)
        self.maxWeight = maxWeight
        self.weights, self.values = self.prune_knapsack(maxWeight, weights, values)
        self.num_items = len(self.weights)
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
                self.matrixClass.add_off_element(j, j, -2 * self.vec_n[i - self.num_items - 1] * self.weights[j - 1])

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

    def prune_knapsack(self, max_weight, weights, values):
        """Prune the knapsack problem by removing items that are too heavy."""
        indices_to_remove = np.where(weights > max_weight)[0]
        weights = np.delete(weights, indices_to_remove)
        values = np.delete(values, indices_to_remove)
        return weights, values


    def evaluate_solution(self, solution):
        """
        Evaluates a solution to the knapsack problem.

        :param solution: A binary array indicating which items are selected (1 for selected, 0 for not selected)
        :return: Total value of the selected items if weight constraint is satisfied, else 0 or a penalty.
        """

        # Check if solution length matches the number of items
        if len(solution) != self.num_items:
            raise ValueError("Solution length must be equal to the number of items.")

        # Calculate the total weight and value of the selected items
        total_weight = np.dot(solution, self.weights)
        total_value = np.dot(solution, self.values)

        # If the total weight exceeds the knapsack capacity, return a penalty (e.g., 0)
        if total_weight > self.maxWeight:
            return 0  # Or any penalty value, such as -(total_weight - self.maxWeight) if you want to penalize excess

        # Otherwise, return the total value of the selected items
        return total_value