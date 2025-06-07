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


class QuadraticFunction(Function):
    def __init__(self, Q, b=np.zeros(2), c=0.0):
        super().__init__(hessian_needed=True)
        self.A = 2 * Q
        self.b = b
        self.c = c

    def objective(self, x) -> float:
        return 0.5 * x.T @ self.A @ x + self.b @ x + self.c

    def gradient(self, x) -> np.ndarray:
        return self.A @ x + self.b

    def hessian(self, x) -> np.ndarray:
        return self.A


class LinearFunction(Function):
    def __init__(self, a):
        super().__init__(hessian_needed=False)
        self.a = a

    def objective(self, x) -> float:
        return self.a @ x

    def gradient(self, x) -> np.ndarray:
        return self.a

    def hessian(self, x) -> np.ndarray:
        return np.zeros((len(self.a), len(self.a)))
