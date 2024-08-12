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

        # A(1-2ay+f_Y(a)) diagonal is zero because A-2A+A=0
        # => not fully, right? Because some elements relate to off and diag elements
        # A*f_y(n)
        for i in y_indices:
            self.matrixClass.add_diag_element(i, -self.A) # A(1-2ay)=A-2A=-A
            for j in y_indices:
                if i != j:
                    self.matrixClass.add_off_element(i, j, 2 * self.A) # A(1+yaay)=A+A=2A

        # A*ynny
        for i in y_indices:
            for j in y_indices:
                if i != j:
                    self.matrixClass.add_off_element(i, j, self.A)

        # -2y(nw)x
        # y perspective
        for i in y_indices:
            const_to_added = 0
            for j in x_indices:
                const_to_added += self.vec_n[i-self.num_items-1] * self.weights[j - 1]
            self.matrixClass.add_diag_element(i, -2 * const_to_added)

        # x perspective
        for j in x_indices:
            const_to_added = 0
            for i in y_indices:
                const_to_added += self.vec_n[i-self.num_items-1] * self.weights[j - 1]
            self.matrixClass.add_diag_element(j, -2 * const_to_added)

        '''
        for i in x_indices:
            self.matrixClass.add_diag_element(i, -2 * self.weights[i - 1])
        '''


        # A(f_x(w))
        for i in x_indices:
            for j in x_indices:
                if i != j:
                    self.matrixClass.add_off_element(i, j, 2 * self.A * self.weights[i - 1] * self.weights[j - 1])

        # Construct H_B
        for alpha, x in enumerate(x_indices):
            self.matrixClass.add_diag_element(x, -self.B * self.values[alpha])

        # I think that is not sufficient because it doesnt give any information what x and y is in the matrix
        # only x relevant for now as it represents the items and y only if a certain weight is matched
        # y might be used to check if solution is valid in the end
        # todo: check after QIRO implementation if this approach makes sense
        self.position_translater = [0] + list(x_indices)

        return self.matrix