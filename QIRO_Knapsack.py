from QIRO import QIRO
from expval_calculation.SingleLayerQAOAKnapsack import SingleLayerQAOAExpectationValuesKnapsack
from expval_calculation.SingleLayerQAOA import SingleLayerQAOAExpectationValues
from expval_calculation.StateVecQAOA import StateVecQAOAExpectationValues
from expval_calculation.StateVecQAOAKnapsack import StateVecQAOAExpectationValuesKnapsack
from problem_generation.Generate_Knapsack import Knapsack
import numpy as np
import matplotlib.pyplot as plt


class QIRO_Knapsack(QIRO):
    """
    :param expectation_values_input: The expectation values that shall be optimized
    :param nc_input: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, nc_input, expectation_values_input, variation="MINQ"):
        super().__init__(nc=nc_input, expectation_values=expectation_values_input)
        self.solution = None
        self.variation = variation

        if self.variation not in ['QIRO', 'MINQ', 'MAXQ', 'MMQ']:
            raise ValueError("Invalid variation type")



    def execute(self):
        step_nr = 0
        self.solution = []

        while self.expectation_values.problem.weights.size != 0 and self.expectation_values.problem.maxWeight > 0 \
                and self.expectation_values.problem.maxWeight >= np.min(self.expectation_values.problem.weights):
            step_nr += 1

            self.expectation_values.optimize()


            if self.variation == "MINQ" or self.variation == "MAXQ" or self.variation == "MMQ":
                for key in self.expectation_values.expect_val_dict.copy().keys():
                    if len(key) == 2:
                        del self.expectation_values.expect_val_dict[key]

            if self.variation == 'QIRO' or self.variation == 'MINQ' or self.variation == 'MMQ':
                # Ties are broken randomly
                sorted_correlation_dict = sorted(self.expectation_values.expect_val_dict.items(),
                                                 key=lambda item: (item[1], np.random.rand()), reverse=True)

            if self.variation == 'MAXQ':
                # Ties are broken randomly
                sorted_correlation_dict = sorted(self.expectation_values.expect_val_dict.items(),
                                                 key=lambda item: (item[1], np.random.rand()), reverse=False)

            max_expect_val_location, max_expect_val = sorted_correlation_dict[0]

            if self.variation in ['QIRO', 'MMQ']:
                max_expect_val_sign = np.sign(max_expect_val).astype(int) # returns 1 if positive and -1 if negative
            elif self.variation == 'MINQ':
                max_expect_val_sign = +1
            elif self.variation == 'MAXQ':
                max_expect_val_sign = -1
            else:
                raise ValueError("Invalid variation type")

            print("Sorted correlation dict: ", sorted_correlation_dict)

            if len(max_expect_val_location) == 1:
                print(f"single var {max_expect_val_location}. Sign: {max_expect_val_sign}")
                self._update_single_correlation(list(max_expect_val_location), max_expect_val_sign, max_expect_val)

            else:
                print(f"Correlation {max_expect_val_location}. Sign: {max_expect_val_sign}.")
                self._update_double_correlation(list(max_expect_val_location), max_expect_val_sign)

            print("Optimized expectation values: ", max_expect_val_location, " Step: ", step_nr)

        calculated_value = lambda self: sum(sub_arr[2] for sub_arr in self.solution)
        calculated_weight = lambda self: sum(sub_arr[1] for sub_arr in self.solution)

        print(
            "Optimization finished. Solution: [Index, Weight, Value]", self.solution,
            " with total value: ", calculated_value(self),
            " and total weight: ", calculated_weight(self)
        )


    ################################################################################
    # Helper functions.                                                            #
    ################################################################################

    def _update_single_correlation(self, max_expect_val_location, max_expect_val_sign, max_expect_val):
        """
        Updates Hamiltonian according to one-point correlations

        neg. correlation: var. more likely to be 0
        pos. correlation: var. more likely to be 1
        approx. 0 correlation: we don't include the var. because it doesn't affect the optimality of the solution
        """

        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        index = max_expect_val_location[0] - 1
        new_weight = self.expectation_values.problem.maxWeight

        print("Current Index: ", index)

        if max_expect_val_sign > 0 or max_expect_val > 0:
            print("Include item to solution: ", index, " with weight: ", weights[index], " and value: ", values[index])

            self.solution.append([index, weights[index], values[index]])
            new_weight = new_weight - weights[index]

        print(
            "Deleting item: ", index,
            " with weight: ", weights[index],
            " and value: ", values[index]
        )

        weights = np.delete(weights, index)
        values = np.delete(values, index)

        print("New weight: ", new_weight)

        self._reinitialize_problem_and_expectation_values(new_weight, weights, values)


    def _update_double_correlation(self, max_expect_val_location, max_expect_val_sign):
        """
        Updates Hamiltonian according to two-point correlations

        neg. correlation: including one item makes it less likely to include the other regarding optimality
        pos. correlation: including one item makes it more likely to include the other regarding optimality
        approx. 0 correlation: including one item doesn't affect the optimality of including the other
        """

        weights = self.expectation_values.problem.weights
        values = self.expectation_values.problem.values

        index_1 = max_expect_val_location[0] - 1
        index_2 = max_expect_val_location[1] - 1

        # add larger correlation item to the solution
        if max_expect_val_sign >= 0:
            corr_1 = self.expectation_values.expect_val_dict[frozenset({index_1 + 1})]
            corr_2 = self.expectation_values.expect_val_dict[frozenset({index_2 + 1})]
            new_weight = 0
            index_adjustment = 0

            if corr_1 > corr_2 and weights[index_1] <= self.problem.maxWeight:
                print(f"Adding item {index_1} to the solution.")
                new_weight += weights[index_1]
                self.solution.append([index_1, weights[index_1], values[index_1]])
                weights = np.delete(weights, [index_1])
                values = np.delete(values, [index_1])
                index_adjustment = 1

            if corr_2 >= corr_1 and new_weight + weights[index_2 - index_adjustment] <= self.problem.maxWeight:
                print(f"Adding item {index_2} to the solution.")
                self.solution.append([index_2, weights[index_2 - index_adjustment], values[index_2 - index_adjustment]])
                new_weight += weights[index_2 - index_adjustment]
                weights = np.delete(weights, [index_2 - index_adjustment])
                values = np.delete(values, [index_2 - index_adjustment])

            self._reinitialize_problem_and_expectation_values(self.problem.maxWeight-new_weight, weights, values)

        # delete lower correlation item
        else:
            corr_1 = self.expectation_values.expect_val_dict[frozenset({index_1 + 1})]
            corr_2 = self.expectation_values.expect_val_dict[frozenset({index_2 + 1})]
            deleted_weight = 0
            index_adjustment = 0

            if corr_1 < corr_2:
                # delete corr 1 item
                print(f"Exclude item {index_1} from problem.")
                deleted_weight += weights[index_1]
                index_adjustment = 1
                weights = np.delete(weights, [index_1])
                values = np.delete(values, [index_1])

            if corr_2 <= corr_1:
                # delete corr 2 item
                print(f"Exclude item {index_2} from problem.")
                deleted_weight += weights[index_2 - index_adjustment]
                weights = np.delete(weights, [index_2 - index_adjustment])
                values = np.delete(values, [index_2 - index_adjustment])

            self._reinitialize_problem_and_expectation_values(self.problem.maxWeight - deleted_weight, weights, values)


    def _reinitialize_problem_and_expectation_values(self, new_weight, weights, values):
        """Reinitializes the problem and expectation values based on the updated weights and values."""

        self.expectation_values.problem = Knapsack(new_weight, weights, values, self.problem.A, self.problem.B)

        if self.expectation_values.type == "SingleLayerQAOAExpectationValueKnapsack":
            self.expectation_values = SingleLayerQAOAExpectationValuesKnapsack(self.expectation_values.problem)
        elif self.expectation_values.type == "StateVecQAOAExpectationValuesKnapsack":
            self.expectation_values = StateVecQAOAExpectationValuesKnapsack(self.expectation_values.problem, self.expectation_values.p)


    ################################################################################
    # Other Approaches. Deprecated!                                                #
    ################################################################################

    def execute_greedy(self):
        self.solution = []
        maxWeight = self.problem.maxWeight

        # Optimize and retrieve expectation values
        self.expectation_values.optimize()
        expectation_values = self.expectation_values.expect_val_dict

        # Sort the expectation values by absolute value (with random tie-breaking)
        sorted_expectation_values = dict(
            sorted(expectation_values.items(), key=lambda item: (abs(item[1]), np.random.rand()), reverse=True)
        )

        # Add the first largest elements (keys) to the solution list
        for key in sorted_expectation_values:
            keys = list(key)
            # only consider one point correlations
            if len(keys) == 1 and self.problem.weights[keys[0]-1] <= maxWeight:
                maxWeight -= self.problem.weights[keys[0]-1]
                index = keys[0] - 1
                self.solution.append([index, self.problem.weights[index], self.problem.values[index]])

        calculated_value = lambda self: sum(sub_arr[2] for sub_arr in self.solution)
        calculated_weight = lambda self: sum(sub_arr[1] for sub_arr in self.solution)

        print(
            "Optimization finished. Solution: [Index, Weight, Value]", self.solution,
            " with total value: ", calculated_value(self),
            " and total weight: ", calculated_weight(self)
        )


    def execute_greedy_with_pruning(self):
        self.solution = []
        maxWeight = self.problem.maxWeight

        # Optimize and retrieve expectation values
        self.expectation_values.optimize()
        expectation_values = self.expectation_values.expect_val_dict

        # Sort the expectation values by absolute value (with random tie-breaking)
        sorted_expectation_values = dict(
            sorted(expectation_values.items(), key=lambda item: (abs(item[1]), np.random.rand()), reverse=True)
        )

        # Pruning threshold
        pruning_threshold = 0.05

        for key in sorted_expectation_values:
            keys = list(key)
            if len(keys) == 1:
                item_weight = self.problem.weights[keys[0] - 1]
                item_value = self.problem.values[keys[0] - 1]

                # Check if the item fits within the remaining capacity
                if item_weight <= maxWeight:
                    # Predict the remaining capacity after adding this item
                    remaining_capacity = maxWeight - item_weight

                    # Determine the maximum value that can be added with the remaining capacity
                    max_future_value = 0
                    for future_key in sorted_expectation_values:
                        future_keys = list(future_key)
                        if len(future_keys) == 1 and future_key != key:
                            future_item_weight = self.problem.weights[future_keys[0] - 1]
                            future_item_value = self.problem.values[future_keys[0] - 1]
                            if future_item_weight <= remaining_capacity:
                                max_future_value += future_item_value

                    # Prune the item if including it would result in inefficient use of capacity
                    if remaining_capacity > 0 and (remaining_capacity / maxWeight) < pruning_threshold:
                        continue

                    # Include the item if it's not pruned
                    maxWeight -= item_weight
                    self.solution.append([keys[0] - 1, item_weight, item_value])

        # Calculate total value and weight of the selected items
        calculated_value = sum(sub_arr[2] for sub_arr in self.solution)
        calculated_weight = sum(sub_arr[1] for sub_arr in self.solution)

        print(
            "Optimization finished. Solution: [Index, Weight, Value]", self.solution,
            " with total value: ", calculated_value,
            " and total weight: ", calculated_weight
        )


    # greedy with correlation to weight ratio
    # dumb approach
    def execute_greedy_with_weight_ratio(self):
        self.solution = []
        maxWeight = self.problem.maxWeight

        # Optimize and retrieve expectation values
        self.expectation_values.optimize()
        expectation_values = self.expectation_values.expect_val_dict

        # Sort the expectation values by the ratio of absolute value to weight (with random tie-breaking)
        sorted_expectation_values = dict(
            sorted(expectation_values.items(),
                   key=lambda item: (abs(item[1]) / self.problem.weights[list(item[0])[0] - 1], np.random.rand()),
                   reverse=True)
        )

        # Add the first largest elements (keys) to the solution list
        for key in sorted_expectation_values:
            keys = list(key)
            # Only consider one-point correlations
            if len(keys) == 1 and self.problem.weights[keys[0] - 1] <= maxWeight:
                maxWeight -= self.problem.weights[keys[0] - 1]
                index = keys[0] - 1
                self.solution.append([index, self.problem.weights[index], self.problem.values[index]])

        calculated_value = lambda self: sum(sub_arr[2] for sub_arr in self.solution)
        calculated_weight = lambda self: sum(sub_arr[1] for sub_arr in self.solution)

        print(
            "Optimization finished. Solution: [Index, Weight, Value]", self.solution,
            " with total value: ", calculated_value(self),
            " and total weight: ", calculated_weight(self)
        )
        return calculated_value(self), calculated_weight(self), self.solution