from QIRO import QIRO
from expval_calculation.SingleLayerQAOAKnapsack import SingleLayerQAOAExpectationValuesKnapsack
from expval_calculation.SingleLayerQAOA import SingleLayerQAOAExpectationValues
from expval_calculation.StateVecQAOA import StateVecQAOAExpectationValues
from problem_generation.Generate_Knapsack import Knapsack
import numpy as np


class QIRO_Knapsack(QIRO):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, nc_input, expectation_values_input):
        super().__init__(nc=nc_input, expectation_values=expectation_values_input)

    def execute(self):
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0

        while self.expectation_values.problem.maxWeight > 0: # as long as no item is deleted anymore?
            step_nr += 1
            print(self.expectation_values.problem.weights)
            print(self.expectation_values.problem.values)
            max_expect_val_location, max_expect_val_sign, max_expect_val = self.expectation_values.optimize()
            if len(max_expect_val_location) == 1:
                self.update_single_correlation(max_expect_val_location)
            else:
                self.update_double_correlation(max_expect_val_location)
            print("Optimized expectation values: ", max_expect_val_location, " Step: ", step_nr)
            # often it is the case that we have two point correlations
            # in that case, we face infinity loops

    def update_single_correlation(self, max_expect_val_location):
        """Updates Hamiltonian according to one-point correlations"""
        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        if len(max_expect_val_location) == 1:
            index = max_expect_val_location[0] - 1

            weight = weights[index]
            print("Deleted item: ", index, " with weight: ", weight, " and value: ", values[index])
            weights = np.delete(weights, index)
            values = np.delete(values, index)
            self.problem = Knapsack(self.expectation_values.problem.maxWeight - weight, weights, values, self.expectation_values.problem.A, self.expectation_values.problem.B)

            if self.expectation_values.type == "SingleLayerQAOAExpectationValueKnapsack":
                self.expectation_values = SingleLayerQAOAExpectationValuesKnapsack(self.problem, self.expectation_values.gamma, self.expectation_values.beta)
            elif self.expectation_values.type == "SingleLayerQAOAExpectationValue":
                self.expectation_values = SingleLayerQAOAExpectationValues(self.problem, self.expectation_values.gamma, self.expectation_values.beta)
            elif self.expectation_values.type == "StateVecQAOAExpectationValues":
                self.expectation_values = StateVecQAOAExpectationValues(self.problem, self.expectation_values.p)
            # elif self.expectation_values.type == "QtensorQAOAExpectationValuesMIS":
            #     self.expectation_values = QtensorQAOAExpectationValuesMIS(
            #         self.problem, self.expectation_values.p
            #     )

    def update_double_correlation(self, max_expect_val_location, max_expect_val_sign):
        """Updates Hamiltonian according to two-point correlations"""

        if len(max_expect_val_location) == 2:
            print("hello")

            if max_expect_val_sign == 1:
                pass
                # remove both variables

            if self.expectation_values.type == "SingleLayerQAOAExpectationValueKnapsack":
                self.expectation_values = SingleLayerQAOAExpectationValuesKnapsack(self.problem)
            elif self.expectation_values.type == "SingleLayerQAOAExpectationValue":
                self.expectation_values = SingleLayerQAOAExpectationValues(self.problem, self.expectation_values.gamma, self.expectation_values.beta)
            elif self.expectation_values.type == "StateVecQAOAExpectationValues":
                self.expectation_values = StateVecQAOAExpectationValues(self.problem, self.expectation_values.p)
            # elif self.expectation_values.type == "QtensorQAOAExpectationValuesMIS":
            #     self.expectation_values = QtensorQAOAExpectationValuesMIS(
            #         self.problem, self.expectation_values.p
            #     )

