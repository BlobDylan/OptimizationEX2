import numpy as np
from tests.examples import Function


class BarrierFunction(Function):

    def __init__(self, func: Function, ineq_constraints: list[Function], t: float):
        super().__init__(hessian_needed=func.hessian_needed)
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.t = t

    def objective(self, x) -> float:
        penalty = sum(-np.log(-c.objective(x)) for c in self.ineq_constraints)
        return self.func.objective(x) + (1 / self.t) * penalty

    def gradient(self, x) -> np.ndarray:
        func_grad = self.func.gradient(x)
        penalty_grad = sum(
            -1 / c.objective(x) * c.gradient(x) for c in self.ineq_constraints
        )
        return func_grad + (1 / self.t) * penalty_grad

    def hessian(self, x) -> np.ndarray:
        func_hess = self.func.hessian(x)
        first_term_hess = sum(
            (1 / (c.objective(x) ** 2)) * np.outer(c.gradient(x), c.gradient(x))
            for c in self.ineq_constraints
        )
        second_term_hess = sum(
            (-1 / c.objective(x)) * c.hessian(x) for c in self.ineq_constraints
        )
        return func_hess + (1 / self.t) * (first_term_hess + second_term_hess)
