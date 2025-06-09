import unittest
import numpy as np
import os

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("WARNING: Matplotlib not available, skipping plots")
    MATPLOTLIB_AVAILABLE = False
from src.constrained_min import InteriorPoint
from tests.examples import (
    QuadraticFunctionWithConstraints,
    LinearFunctionWithConstraints,
)


class TestConstrainedMin(unittest.TestCase):
    def setUp(self):
        os.makedirs("output", exist_ok=True)

    def test_qp(self):

        qp = QuadraticFunctionWithConstraints()
        x0 = np.array([0.1, 0.2, 0.7])

        solver = InteriorPoint(
            func=qp,
            ineq_constraints=qp.inequality_constraints,
            eq_constraints_mat=qp.equality_constraints_mat,
            eq_constraints_rhs=qp.equality_constraints_rhs,
            x0=x0,
        )
        solver.minimize()

        final_x = solver.current_x
        final_obj = qp.objective(final_x)
        outer_hist = [point for point, _ in solver.outer_history]

        print("\nQP Results:")
        print(f"Final x: {final_x}")
        print(f"Objective: {final_obj:.6f}")
        print(
            f"Constraints: x+y+z = {np.sum(final_x):.6f}, "
            f"x≥0: {final_x[0]>=0}, y≥0: {final_x[1]>=0}, z≥0: {final_x[2]>=0}"
        )

        if MATPLOTLIB_AVAILABLE:

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.2, color="gray"
            )

            path = np.array(outer_hist)
            ax.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                "bo-",
                markersize=4,
                label="Central Path",
            )
            ax.scatter(*final_x, color="red", s=100, label="Solution")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("QP: Central Path in 3D")
            ax.legend()
            plt.savefig(os.path.join("output", "qp_central_path.png"), dpi=300)
            plt.close()

            objectives = [obj for _, obj in solver.outer_history]
            plt.figure(figsize=(10, 6))
            plt.plot(objectives, "bo-")
            plt.xlabel("Outer Iteration")
            plt.ylabel("Objective Value")
            plt.title("QP: Objective Value vs. Iteration")
            plt.grid(True)
            plt.savefig(os.path.join("output", "qp_objective_history.png"), dpi=300)
            plt.close()

    def test_lp(self):
        lp = LinearFunctionWithConstraints()
        x0 = np.array([0.5, 0.75])

        solver = InteriorPoint(
            func=lp,
            ineq_constraints=lp.inequality_constraints,
            eq_constraints_mat=lp.equality_constraints_mat,
            eq_constraints_rhs=lp.equality_constraints_rhs,
            x0=x0,
        )
        solver.minimize()

        final_x = solver.current_x
        final_obj = -lp.objective(final_x)
        outer_hist = [point for point, _ in solver.outer_history]

        print("\nLP Results:")
        print(f"Final x: {final_x}")
        print(f"Objective (max): {final_obj:.6f}")
        print("Constraints:")
        print(f"  y ≥ -x+1: {final_x[1] >= -final_x[0] + 1}")
        print(f"  y ≤ 1: {final_x[1] <= 1}")
        print(f"  x ≤ 2: {final_x[0] <= 2}")
        print(f"  y ≥ 0: {final_x[1] >= 0}")

        if MATPLOTLIB_AVAILABLE:

            plt.figure(figsize=(10, 8))

            vertices = np.array([[1, 0], [2, 0], [2, 1], [1, 1]])
            plt.fill(
                vertices[:, 0],
                vertices[:, 1],
                "gray",
                alpha=0.2,
                label="Feasible Region",
            )

            path = np.array(outer_hist)
            plt.plot(path[:, 0], path[:, 1], "bo-", markersize=4, label="Central Path")
            plt.scatter(*final_x, color="red", s=100, label="Solution")

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("LP: Central Path in 2D")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("output", "lp_central_path.png"), dpi=300)
            plt.close()

            objectives = [-obj for _, obj in solver.outer_history]
            plt.figure(figsize=(10, 6))
            plt.plot(objectives, "bo-")
            plt.xlabel("Outer Iteration")
            plt.ylabel("Objective Value (x+y)")
            plt.title("LP: Objective Value vs. Iteration")
            plt.grid(True)
            plt.savefig(os.path.join("output", "lp_objective_history.png"), dpi=300)
            plt.close()


if __name__ == "__main__":
    unittest.main()
