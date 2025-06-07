import numpy as np
from tests.examples import Function


class InteriorPoint:
    def __init__(
        self,
        func: Function,
        ineq_constraints: list[Function],
        eq_constraints_mat: np.ndarray,
        eq_constraints_rhs: np.ndarray,
        x0: np.ndarray,
        mu: float = 10,
        t: int = 1,
        lambda_threshold: float = 1e-8,
        outer_loop_epsilon: float = 1e-10,
        outer_loop_max_iter: int = 1000,
    ):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs

        self.current_x = x0.copy()
        self.current_fx = self.func.objective(x0)

        self.mu = mu
        self.t = t
        self.lambda_threshold = lambda_threshold
        self.outer_loop_epsilon = outer_loop_epsilon
        self.outer_loop_max_iter = outer_loop_max_iter

        self.history = [(self.current_x, self.current_fx)]
        self.success = False
        self.output_message = ""

    def backtracking(self, x, d, fx, gx, c1=0.01, rho=0.5, max_backtrack=50):
        alpha = 1.0
        for _ in range(max_backtrack):
            new_x = x + alpha * d
            new_fx = self.func.objective(new_x)
            if new_fx <= fx + c1 * alpha * np.dot(gx, d):
                return alpha
            alpha *= rho
        return alpha

    def inner_loop(self, max_iter: int = 1000) -> bool:
        barrier = BarrierFunction(self.func, self.ineq_constraints, self.t)
        current_itteration = 0

        while current_itteration < max_iter:
            barrier_hessian = barrier.hessian(self.current_x)
            barrier_gradient = barrier.gradient(self.current_x)
            left_hand_side_matrix = np.block(
                [
                    [barrier_hessian, self.eq_constraints_mat.T],
                    [
                        self.eq_constraints_mat,
                        np.zeros(
                            (
                                self.eq_constraints_mat.shape[0],
                                self.eq_constraints_mat.shape[0],
                            )
                        ),
                    ],
                ]
            )
            right_hand_side_vector = np.concatenate(
                [
                    -barrier_gradient,
                    np.zeros(self.eq_constraints_mat.shape[0]),
                ]
            )

            try:
                step_direction = np.linalg.solve(
                    left_hand_side_matrix, right_hand_side_vector
                )
            except np.linalg.LinAlgError as e:
                self.output_message = f"Linear algebra error: {e}"
                return False
            step_direction = step_direction[: len(self.current_x)]

            lambda_current = np.sqrt(
                step_direction.T @ barrier_hessian @ step_direction
            )

            if 0.5 * (lambda_current**2) < self.lambda_threshold:
                return True

            step_size = self.backtracking(
                self.current_x, step_direction, self.current_fx, barrier_gradient
            )

            self.current_x += step_size * step_direction
            self.current_fx = barrier.objective(self.current_x)
            self.history.append((self.current_x, self.current_fx))
            current_itteration += 1

        self.output_message = "Maximum iterations reached without convergence."
        return False

    def minimize(self):
        outer_loop_iter = 0

        while outer_loop_iter < self.outer_loop_max_iter:
            if not self.inner_loop():
                return False

            if self.ineq_constraints.shape[0] >= self.outer_loop_epsilon * self.t:
                self.t *= self.mu
            else:
                self.success = True
                return True

            outer_loop_iter += 1

        self.output_message = (
            "Maximum outer loop iterations reached without convergence."
        )
        return False


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
        return func_grad + penalty_grad

    def hessian(self, x) -> np.ndarray:
        func_hess = self.func.hessian(x)
        first_term_hess = sum(
            (1 / (self.t * c.objective(x) ** 2))
            * np.outer(c.gradient(x), c.gradient(x))
            for c in self.ineq_constraints
        )
        second_term_hess = sum(
            (-1 / c.objective(x)) * c.hessian(x) for c in self.ineq_constraints
        )

        return func_hess + first_term_hess + second_term_hess
