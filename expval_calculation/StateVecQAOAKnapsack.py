from expval_calculation.StateVecQAOA import StateVecQAOAExpectationValues
import pennylane as qml

class StateVecQAOAExpectationValuesKnapsack(StateVecQAOAExpectationValues):

    def __init__(self, problem, p, device="default.qubit", num_opts=1, num_opt_steps=50):
        super().__init__(problem, p, device, num_opts, num_opt_steps)

        self.type = "StateVecQAOAExpectationValuesKnapsack"

    def _get_expval_operator_dict(self):
        """
        Returns a list of PauliZ operators for each qubit.
        """

        expval_operator_dict = {}
        for i in range(1, len(self.problem.position_translater)):
            for j in range(1, len(self.problem.position_translater)):
                if self.problem.matrix[i, j] != 0:
                    if i == j:
                        expval_operator_dict[frozenset({i})] = qml.PauliZ(i - 1)
                    else:
                        expval_operator_dict[frozenset({i, j})] = qml.PauliZ(i - 1) @ qml.PauliZ(j - 1)
        return expval_operator_dict