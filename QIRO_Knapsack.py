from QIRO import QIRO
from expval_calculation.SingleLayerQAOAKnapsack import SingleLayerQAOAExpectationValuesKnapsack
from expval_calculation.SingleLayerQAOA import SingleLayerQAOAExpectationValues
from expval_calculation.StateVecQAOA import StateVecQAOAExpectationValues
from problem_generation.Generate_Knapsack import Knapsack
import numpy as np


class QIRO_Knapsack(QIRO):
    """
    :param expectation_values_input: The expectation values that shall be optimized
    :param nc_input: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, nc_input, expectation_values_input):
        super().__init__(nc=nc_input, expectation_values=expectation_values_input)
        self.solution = None

    def execute(self):
        step_nr = 0
        self.solution = []

        while self.expectation_values.problem.maxWeight >= np.min(self.expectation_values.problem.weights):
            step_nr += 1

            max_expect_val_location, max_expect_val_sign, max_expect_val = self.expectation_values.optimize()

            if len(max_expect_val_location) == 1:
                print(f"single var {max_expect_val_location}. Sign: {max_expect_val_sign}")
                self.update_single_correlation(max_expect_val_location, max_expect_val_sign)
            else:
                print(f"Correlation {max_expect_val_location}. Sign: {max_expect_val_sign}.")
                self.update_double_correlation(max_expect_val_location, max_expect_val_sign)

            print("Optimized expectation values: ", max_expect_val_location, " Step: ", step_nr)

            # stop, if no elements left
            if self.expectation_values.problem.weights.size == 0 or self.expectation_values.problem.maxWeight == 0:
                break

        optimimal_value = lambda self: sum(sub_arr[2] for sub_arr in self.solution)
        print("Optimization finished. Solution: [Index, Weight, Value]", self.solution, " with total value: ", optimimal_value(self))

    ################################################################################
    # Helper functions.                                                            #
    ################################################################################

    def update_single_correlation(self, max_expect_val_location, max_expect_val_sign):
        """
        Updates Hamiltonian according to one-point correlations

        neg correlation: var. more likely to be 0
        pos correlation: var. more likely to be 1
        approx. 0 correlation: including var. doesn't affect the optimality of the solution
        """

        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        index = max_expect_val_location[0] - 1
        new_weight = self.expectation_values.problem.maxWeight

        if max_expect_val_sign > 0 and weights[index] <= new_weight:
            print("Include item to solution: ", index, " with weight: ", weights[index], " and value: ", values[index])

            self.solution.append([index, weights[index], values[index]])
            new_weight = new_weight - weights[index]

        print("Deleting item: ", index, " with weight: ", weights[index], " and value: ", values[index])

        weights = np.delete(weights, index)
        values = np.delete(values, index)

        print("New weight: ", new_weight)
        self._reinitialize_problem_and_expectation_values(new_weight, weights, values)

    def update_double_correlation(self, max_expect_val_location, max_expect_val_sign):
        """
        Updates Hamiltonian according to two-point correlations

        neg correlation: including one item makes it less likely to include the other regarding optimality
        pos correlation: including one item makes it more likely to include the other regarding optimality
        approx. 0 correlation: including one item doesn't affect the optimality of including the other
        """

        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        index_1 = max_expect_val_location[0] - 1
        index_2 = max_expect_val_location[1] - 1
        combined_weight = weights[index_1] + weights[index_2]
        new_weight = self.expectation_values.problem.maxWeight

        if max_expect_val_sign > 0 and combined_weight <= new_weight:
            print(f"Including both items {index_1} and {index_2} to the solution.")

            new_weight = new_weight - combined_weight
            self.solution.append([index_1, weights[index_1], values[index_1]])
            self.solution.append([index_2, weights[index_2], values[index_2]])

        print(f"Removing items: {index_1} and {index_2}.")

        weights = np.delete(weights, [index_1, index_2])
        values = np.delete(values, [index_1, index_2])

        print("New weight: ", new_weight)
        self._reinitialize_problem_and_expectation_values(new_weight, weights, values)

    def _reinitialize_problem_and_expectation_values(self, new_weight, weights, values):
        """Reinitializes the problem and expectation values based on the updated weights and values."""

        self.problem = Knapsack(new_weight, weights, values, self.problem.A, self.problem.B)

        if self.expectation_values.type == "SingleLayerQAOAExpectationValueKnapsack":
            self.expectation_values = SingleLayerQAOAExpectationValuesKnapsack(self.problem)
        elif self.expectation_values.type == "SingleLayerQAOAExpectationValue":
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem, self.expectation_values.gamma,
                                                                       self.expectation_values.beta)
        elif self.expectation_values.type == "StateVecQAOAExpectationValues":
            self.expectation_values = StateVecQAOAExpectationValues(self.problem, self.expectation_values.p)