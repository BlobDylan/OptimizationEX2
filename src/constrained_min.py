import numpy as np
from abc import ABC, abstractmethod
from tests.examples import Function


class InteriorPoint(ABC):
    def __init__(
        self,
        func: Function,
        ineq_constraints,
        eq_constraints_mat,
        eq_constraints_rhs,
        x0,
    ):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.x0 = x0
        self.current_iteration = 0
        self.current_x = x0.copy()
        self.current_fx = self.f.objective(x0)
        self.history = [(x0.copy(), self.current_fx)]
        self.success = False
        self.output_message = ""

    def backtracking(self, x, d, fx, gx, c1=0.01, rho=0.5, max_backtrack=50):
        alpha = 1.0
        for _ in range(max_backtrack):
            new_x = x + alpha * d
            new_fx = self.f.objective(new_x)
            if new_fx <= fx + c1 * alpha * np.dot(gx, d):
                return alpha
            alpha *= rho
        return alpha

    @abstractmethod
    def step(self):
        pass

    def minimize(self):
        while self.step():
            pass
        return self.current_x, self.current_fx, self.success, self.history
