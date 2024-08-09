from QIRO import QIRO


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
        self.expectation_values.optimize()
