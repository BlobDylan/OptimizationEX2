import numpy as np
from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self, hessian_needed=True):
        self.hessian_needed = hessian_needed

    @abstractmethod
    def objective(self, x) -> float:
        pass

    @abstractmethod
    def gradient(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def hessian(self, x) -> np.ndarray:
        pass


class QuadraticFunctionWithConstraints(Function):
    def __init__(self):
        super().__init__(hessian_needed=True)
        self.equality_constraints_mat = np.array([[1, 1, 1]])
        self.equality_constraints_rhs = np.array([1])
        self.inequality_constraints = [
            InequalityConstraint(
                f=lambda x: -x[0],
                g=lambda _: np.array([-1, 0, 0]),
                h=lambda _: np.zeros((3, 3)),
            ),
            InequalityConstraint(
                f=lambda x: -x[1],
                g=lambda _: np.array([0, -1, 0]),
                h=lambda _: np.zeros((3, 3)),
            ),
            InequalityConstraint(
                f=lambda x: -x[2],
                g=lambda _: np.array([0, 0, -1]),
                h=lambda _: np.zeros((3, 3)),
            ),
        ]

    def objective(self, x):
        return x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2

    def gradient(self, x):
        return np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])

    def hessian(self, x):
        return np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])


class LinearFunctionWithConstraints(Function):
    def __init__(self):
        super().__init__(hessian_needed=True)
        self.equality_constraints_mat = np.array([])
        self.equality_constraints_rhs = np.array([])
        self.inequality_constraints = [
            InequalityConstraint(
                f=lambda x: -x[0] - x[1] + 1,
                g=lambda _: np.array([-1, -1]),
                h=lambda _: np.zeros((2, 2)),
            ),
            InequalityConstraint(
                f=lambda x: x[1] - 1,
                g=lambda _: np.array([0, 1]),
                h=lambda _: np.zeros((2, 2)),
            ),
            InequalityConstraint(
                f=lambda x: x[0] - 2,
                g=lambda _: np.array([1, 0]),
                h=lambda _: np.zeros((2, 2)),
            ),
            InequalityConstraint(
                f=lambda x: -x[1],
                g=lambda _: np.array([0, -1]),
                h=lambda _: np.zeros((2, 2)),
            ),
        ]

    def objective(self, x):
        return -(x[0] + x[1])

    def gradient(self, x):
        return np.array([-1, -1])

    def hessian(self, x):
        return np.array([[0, 0], [0, 0]])


class InequalityConstraint(Function):
    def __init__(self, f, g, h):
        super().__init__(hessian_needed=True)
        self.f = f
        self.g = g
        self.h = h

    def objective(self, x):
        return self.f(x)

    def gradient(self, x):
        return self.g(x)

    def hessian(self, x):
        return self.h(x)
