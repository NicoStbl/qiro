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
        self.solution = None

    def execute(self):
        step_nr = 0
        self.solution = []

        while self.expectation_values.problem.maxWeight >= np.min(self.expectation_values.problem.weights):
            # do we need another loop here? I think for the initial version, it is not necessary

            # as long as there is a weight that is smaller than or equal to the maxWeight
            # => a item that still can be added
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
            if self.expectation_values.problem.weights.size == 0:
                break

        print("Optimization finished. Solution: [Index, Weight, Value]", self.solution)

    ################################################################################
    # Helper functions.                                                            #
    ################################################################################

    def update_single_correlation(self, max_expect_val_location, max_expect_val_sign):
        """Updates Hamiltonian according to one-point correlations"""

        # neg correlation: var. more likely to be 1
        # pos correlation: var. more likely to be 0
        # approx. 0 correlation: var. has an equal probability of being included or not
        # => we delete approx. 0 correlation because they don't contribute to optimality

        # => we want to delete the variable that is more likely to be 0
        # and include the variable that is more likely to be 1
        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        index = max_expect_val_location[0] - 1

        new_weight = self.expectation_values.problem.maxWeight
        print("Max expect val sign: ", max_expect_val_sign)

        # what is the right direction > or <?
        if max_expect_val_sign < 0:
            # delete item and include to solution
            print("Deleted item and include to solution: ", index, " with weight: ", weights[index], " and value: ", values[index])
            print(self.expectation_values.problem.maxWeight)
            if (weights[index] <= self.expectation_values.problem.maxWeight):
                # add item to solution
                self.solution.append([index, weights[index], values[index]])
                new_weight = self.expectation_values.problem.maxWeight - weights[index]
        else:
            # delete item but dont add to solution
            index = max_expect_val_location[0] - 1
            print(self.expectation_values.problem.maxWeight)


            print("Deleted item and exclude from solution: ", index, " with weight: ", weights[index], " and value: ", values[index])

        weights = np.delete(weights, index)
        values = np.delete(values, index)

        self.problem = Knapsack(new_weight, weights, values, self.expectation_values.problem.A, self.expectation_values.problem.B)
        self._reinitialize_problem_and_expectation_values(weights, values)

    def update_double_correlation(self, max_expect_val_location, max_expect_val_sign):
        """Updates Hamiltonian according to two-point correlations"""

        # todo clean the code here -> it looks quite messy

        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        index_1 = max_expect_val_location[0] - 1
        index_2 = max_expect_val_location[1] - 1

        combined_weight = weights[index_1] + weights[index_2]
        index_to_remove = None
        print("Max expect val sign: ", max_expect_val_sign)

        if max_expect_val_sign < 0 and combined_weight <= self.expectation_values.problem.maxWeight:
            # Add both items if possible
            print(f"Adding both items {index_1} and {index_2} due to positive correlation.")
            # Adjust the problem's maxWeight
            self.problem.maxWeight -= combined_weight
            self.solution.append([index_1, weights[index_1], values[index_1]])
            self.solution.append([index_2, weights[index_2], values[index_2]])
            print(f"Added items {index_1} and {index_2} to the solution.")
            '''
            else:
                # Decide based on some criterion
                # E.g., select item with the highest value-to-weight ratio
                # todo decide based on one point correlation
                # => the item with the more promising one is added to the solution
                # how to get the one point correlation here? Is that even possible?
                # i dont like the following differentiation
                
                # first, just delete both for simplification

                if values[index_1] / weights[index_1] > values[index_2] / weights[index_2] and weights[index_1] <= self.expectation_values.problem.maxWeight:
                    index_to_remove = index_1
                elif weights[index_2] <= self.expectation_values.problem.maxWeight:
                    index_to_remove = index_2

                if index_to_remove == None: # is that really a good differentiation? Maybe, I should rather consider to introduce a binary variable stating if an item is selected
                    pass

                print(f"Adding item {index_to_remove} and removing the other due to positive correlation.")
                if weights[index_to_remove] <= self.expectation_values.problem.maxWeight:
                    # Update maxWeight
                    self.problem.maxWeight -= weights[index_to_remove]
                else:
                    # too large to add
                    # => remove and dont add to solution => continue procedure
                    pass
                '''
        else:
            '''
            # comparison between the two items using the value-to-weight ratio
            if values[index_1] / weights[index_1] < values[index_2] / weights[index_2]:
                index_to_remove = index_1
            else:
                index_to_remove = index_2
            '''

            print(f"Removing items: {index_1} and {index_2} due to negative correlation.")

        weights = np.delete(weights, [index_1, index_2])
        values = np.delete(values, [index_1, index_2])
        # Reinitialize problem and expectation values
        self.problem = Knapsack(self.problem.maxWeight, weights, values, self.problem.A, self.problem.B)
        self._reinitialize_problem_and_expectation_values(weights, values)

    def _reinitialize_problem_and_expectation_values(self, weights, values):
        """Reinitializes the problem and expectation values based on the updated weights and values."""

        self.problem = Knapsack(self.problem.maxWeight, weights, values, self.problem.A, self.problem.B)

        if self.expectation_values.type == "SingleLayerQAOAExpectationValueKnapsack":
            self.expectation_values = SingleLayerQAOAExpectationValuesKnapsack(self.problem)
        elif self.expectation_values.type == "SingleLayerQAOAExpectationValue":
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem, self.expectation_values.gamma,
                                                                       self.expectation_values.beta)
        elif self.expectation_values.type == "StateVecQAOAExpectationValues":
            self.expectation_values = StateVecQAOAExpectationValues(self.problem, self.expectation_values.p)