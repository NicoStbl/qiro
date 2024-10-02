import numpy as np
from scipy.optimize import fsolve

from expval_calculation.SingleLayerQAOA import SingleLayerQAOAExpectationValues
from problem_generation.Generating_Problems import Problem


class SingleLayerQAOAExpectationValuesKnapsack(SingleLayerQAOAExpectationValues):
    """
    :param problem: input problem
    This class computes the p=1 QAOA expectation values, as given
    by the formulae in Ozaeta et al.
    """

    # Initialization method
    def __init__(self, problem: Problem, gamma: float = 0.0, beta: float = 0.0):
        # Call the parent class constructor
        super().__init__(problem, gamma, beta)

        # Set the type of the expectation value
        self.type = "SingleLayerQAOAExpectationValueKnapsack"

    def calc_expect_val(self) -> (list, int, float):
        """Calculate all one- and two-point correlation expectation values and return the one with highest absolute value."""

        # get rid of slack variables (Y)

        # initialize dictionary for saving the correlations
        self.expect_val_dict = {}

        # this first part takes care of the case where all correlations are 0.
        Z = np.sin(2 * self.beta) * self._calc_single_terms(gamma=self.gamma, index=1)
        if np.abs(Z) > 0.:
            rounding_list = [
                [[self.problem.position_translater[1]], np.sign(Z), np.abs(Z)]
            ]
            max_expect_val = np.abs(Z)
        else:
            rounding_list = [[[self.problem.position_translater[1]], 1, 0.]]
            max_expect_val = 0.

        self.expect_val_dict[frozenset({1})] = Z

        # iterating through single-body terms
        for index in range(1, len(self.problem.position_translater)):
            Z = np.sin(2 * self.beta) * self._calc_single_terms(
                gamma=self.gamma, index=index
            )
            self.expect_val_dict[frozenset({index})] = Z
            if np.abs(Z) > max_expect_val:
                rounding_list = [
                    [[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)]
                ]
                max_expect_val = np.abs(Z)
            elif np.abs(Z) == max_expect_val:
                rounding_list.append(
                    [[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)]
                )

        # iterating through two-body terms; on
        for index_large in range(1, len(self.problem.position_translater)):
            for index_small in range(1, index_large):
                # we only compute correlations if the coupling coefficient is not 0. between variables index_large and index_small
                if self.problem.matrix[index_large, index_small] != 0:
                    b_part_term, c_part_term = self._calc_coupling_terms(
                        gamma=self.gamma,
                        index_large=index_large,
                        index_small=index_small,
                    )
                    ZZ = (
                        2*np.sin(4 * self.beta) * b_part_term
                        - ((np.sin(2 * self.beta)) ** 2) * c_part_term
                    )
                    self.expect_val_dict[frozenset({index_large, index_small})] = ZZ
                    if np.abs(ZZ) > max_expect_val:
                        rounding_list = [
                            [
                                [
                                    self.problem.position_translater[index_large],
                                    self.problem.position_translater[index_small],
                                ],
                                np.sign(ZZ),
                                np.abs(ZZ),
                            ]
                        ]
                        max_expect_val = np.abs(ZZ)
                    elif np.abs(ZZ) == max_expect_val:
                        rounding_list.append(
                            [
                                [
                                    self.problem.position_translater[index_large],
                                    self.problem.position_translater[index_small],
                                ],
                                np.sign(ZZ),
                                np.abs(ZZ),
                            ]
                        )

        # random tie-breaking of the largest correlation.
        random_index = np.random.randint(len(rounding_list))
        rounding_element = rounding_list[random_index]
        max_expect_val_location = rounding_element[0]
        max_expect_val_sign = rounding_element[1]
        max_expect_val = rounding_element[2]

        return max_expect_val_location, int(max_expect_val_sign), max_expect_val