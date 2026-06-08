"""Fast tests for the optimizer logic.

The real physics formulas are expensive and do not have simple closed-form
answers.  These tests temporarily replace them with simple formulas whose best
answers are known, so a failure points to the optimizer setup itself.
"""

import contextlib
import unittest

import jax.numpy as jnp
import numpy as np

from optimal_JAX import optimal_constrainted_norm_p as optimizer


@contextlib.contextmanager
def replace_physics_with_simple_formulas(objective_formula, volume_formula):
    old_objective_value_jax = optimizer.objective_value_jax
    old_volume_jax = optimizer.volume_jax
    optimizer.objective_value_jax = objective_formula
    optimizer.volume_jax = volume_formula
    try:
        yield
    finally:
        optimizer.objective_value_jax = old_objective_value_jax
        optimizer.volume_jax = old_volume_jax


class OptimizeShapeTests(unittest.TestCase):
    def test_optimizer_leaves_slack_when_best_shape_is_below_volume_limit(self):
        best_shape = jnp.array([0.35, 0.8, 0.1], dtype=jnp.float64)

        def simple_objective(
            shape,
            objective_name,
            A=optimizer.DEFAULT_A,
            N=optimizer.DEFAULT_N,
        ):
            del objective_name, A, N
            return -jnp.sum((shape - best_shape) ** 2)

        def simple_volume(shape, point_count=optimizer.DEFAULT_VOLUME_POINTS):
            del point_count
            return shape[0] + 0.25 * shape[1] + 0.05 * shape[2] ** 2

        start_shape = np.array([0.12, 0.45, -0.15], dtype=float)
        target_volume = 1.25

        with replace_physics_with_simple_formulas(simple_objective, simple_volume):
            run = optimizer.optimize_shape(
                start_shape=start_shape,
                target_volume=target_volume,
                maxiter=40,
                volume_points=16,
            )

        self.assertTrue(run["result"].success, run["result"].message)
        self.assertGreater(run["final_objective"], run["initial_objective"])
        self.assertGreater(run["final_volume_margin"], 0.5)
        np.testing.assert_allclose(
            run["final_shape"],
            np.asarray(best_shape),
            atol=1e-6,
        )

    def test_optimizer_moves_to_known_boundary_optimum_when_volume_limit_is_active(self):
        epsilon_lower_bound = optimizer.PARAMETER_BOUNDS["epsilon"][0]

        def simple_objective(
            shape,
            objective_name,
            A=optimizer.DEFAULT_A,
            N=optimizer.DEFAULT_N,
        ):
            del objective_name, A, N
            return shape[0] + 2.0 * shape[1] - 0.1 * shape[2] ** 2

        def simple_volume(shape, point_count=optimizer.DEFAULT_VOLUME_POINTS):
            del point_count
            return shape[0] + shape[1] + shape[2] ** 2

        start_shape = np.array([0.2, 0.2, 0.1], dtype=float)
        target_volume = 0.8
        expected_shape = np.array(
            [epsilon_lower_bound, target_volume - epsilon_lower_bound, 0.0],
            dtype=float,
        )

        with replace_physics_with_simple_formulas(simple_objective, simple_volume):
            run = optimizer.optimize_shape(
                start_shape=start_shape,
                target_volume=target_volume,
                maxiter=40,
                volume_points=16,
            )

        self.assertTrue(run["result"].success, run["result"].message)
        self.assertGreater(run["final_objective"], run["initial_objective"])
        self.assertLessEqual(run["final_volume"], target_volume + 1e-7)
        np.testing.assert_allclose(run["final_shape"], expected_shape, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
