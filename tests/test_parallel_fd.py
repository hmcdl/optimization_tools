"""Tests for parallel finite-difference helper."""

from __future__ import annotations

import copy
import pickle
import unittest

import numpy as np

from optimization_tools.abstract_object import CachableObject
from optimization_tools.abstract_solver import CachableSolver
from optimization_tools.config import OptimizationConfig
from optimization_tools.opt_conditions import OptConditions
from optimization_tools.optimizers.gradient_optimizer import (
    GradientOptimizer,
    OptimizationTaskWithNormalization,
)
from optimization_tools.parallel_fd import (
    ParallelFiniteDifferences,
    _dedupe_points,
    _invalidate_task_eval_cache,
    clip_to_bounds,
    collect_fd_stencil_points,
    model_at_x_norm,
)
from scipy.optimize import Bounds


def rosen_mass(x1: float, x2: float) -> float:
    return sum(100.0 * (x2 - x1**2) ** 2 + (1 - x1) ** 2 for _ in [0])


class SimpleVector(CachableObject):
    def __init__(self, x1: float, x2: float) -> None:
        super().__init__()
        self.x1 = x1
        self.x2 = x2


class RosenSolver(CachableSolver):
    eval_count = 0

    def non_cached_calculation(self, calc_task: SimpleVector, unique_id: str):
        RosenSolver.eval_count += 1
        mass = rosen_mass(calc_task.x1, calc_task.x2)
        return {
            "ineq1": 1 - calc_task.x1 - 2 * calc_task.x2,
            "ineq2": 1 - calc_task.x1**2 - calc_task.x2,
            "mass": mass,
            "objective": mass,
        }

    def configure(self, configure_dict):
        return None


class EvalCacheConstraint:
    def __init__(self, task, parameter: str, limit: float) -> None:
        self.task = task
        self.parameter = parameter
        self.limit = limit

    def __call__(self, x):
        results = self.task.ensure_evaluated(x)
        result = float(results[self.parameter])
        if self.limit != 0:
            return result / self.limit - 1
        return result


class EvalCacheTask(OptimizationTaskWithNormalization):
    """Task-level eval cache like WingSystemOptimizationTaskWithNormalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._eval_cache_signature = None
        self._eval_cache_results = None

    def invalidate_eval_cache(self) -> None:
        self._eval_cache_signature = None
        self._eval_cache_results = None

    def ensure_evaluated(self, x) -> dict:
        x_denorm = [
            float(x[i]) * self.denorm_coefficients[i]
            for i in range(len(self.lower_bounds))
        ]
        inner_model = copy.deepcopy(self.model)
        self.x_to_model(inner_model, x_denorm, self.conversion_map)
        signature = inner_model.signature()
        if signature != self._eval_cache_signature:
            self._eval_cache_results = self.solver.solve(
                inner_model,
                self.unique_id,
                res_type=None,
            )
            self._eval_cache_signature = signature
        assert self._eval_cache_results is not None
        return self._eval_cache_results

    def objective(self, x):
        results = self.ensure_evaluated(x)
        return float(results["objective"]) / self.cost_function_normalization

    def _update_bounds_and_constraints(self):
        super()._update_bounds_and_constraints()
        self.cons = []
        for constraint in self.opt_conditions.constraints:
            self.cons.append({
                "type": "ineq",
                "fun": EvalCacheConstraint(
                    self,
                    constraint,
                    self.opt_conditions.constraints[constraint],
                ),
            })


class TestParallelFD(unittest.TestCase):
    def setUp(self):
        RosenSolver.eval_count = 0
        self.config = OptimizationConfig(num_proc=1, max_iter=3, logging_dir=".", calculation_dir=".")
        self.model = SimpleVector(0.4, 0.1)
        self.model.cache_signature_fields = ("x1", "x2")
        self.solver = RosenSolver(self.config)
        opt_vars = {"x1": {"min": 0.0, "max": 1.0}, "x2": {"min": -0.5, "max": 2.0}}
        constraints = {"ineq1": 0.0, "ineq2": 0.0}
        self.task = OptimizationTaskWithNormalization(
            self.model,
            "rosen_parallel_fd",
            OptConditions(opt_vars, constraints),
            self.solver,
            self.config,
        )
        self.task.cost_function_normalization = 1.0
        bounds = Bounds(
            [b * coeff for b, coeff in zip(self.task.lower_bounds, self.task.normalization_coefficients)],
            [b * coeff for b, coeff in zip(self.task.upper_bounds, self.task.normalization_coefficients)],
        )
        self.bounds = bounds
        self.x0 = np.array(
            [v * self.task.normalization_coefficients[i] for i, v in enumerate(self.task.get_x())],
            dtype=float,
        )

    def test_solver_picklable(self):
        payload = pickle.dumps(self.solver)
        restored = pickle.loads(payload)
        self.assertIsInstance(restored, RosenSolver)

    def test_stencil_has_multiple_points(self):
        points = collect_fd_stencil_points(self.x0, 0.01, self.bounds)
        self.assertGreater(len(points), 1)

    def test_parallel_fd_gradient_runs(self):
        fd = ParallelFiniteDifferences(self.task, self.config, 0.01, self.bounds)
        grad = fd.objective_jac(self.x0)
        self.assertEqual(grad.shape, (2,))

    def test_prefill_populates_cache(self):
        fd = ParallelFiniteDifferences(self.task, self.config, 0.01, self.bounds)
        fd.prefill(self.x0)
        self.assertGreater(len(fd._fd_cache_map), 0)

    def test_prefill_calls_log_fd_design_point_hook(self):
        logged: list[tuple] = []

        def log_fd_design_point(x_norm, results):
            logged.append((np.asarray(x_norm, dtype=float).copy(), dict(results)))

        self.task.log_fd_design_point = log_fd_design_point
        fd = ParallelFiniteDifferences(self.task, self.config, 0.01, self.bounds)
        fd.prefill(self.x0)
        self.assertGreater(len(logged), 0)
        for _x_norm, results in logged:
            self.assertIn("objective", results)

    def test_prefill_commit_after_flag_is_ignored(self):
        commits: list[str] = []

        def commit_sensitivity_iteration():
            commits.append("commit")

        self.task.commit_sensitivity_iteration = commit_sensitivity_iteration
        fd = ParallelFiniteDifferences(self.task, self.config, 0.01, self.bounds)
        fd.prefill(self.x0, commit_after=False)
        x1 = self.x0.copy()
        x1[0] = min(float(self.bounds.ub[0]), x1[0] * 1.01)
        fd.prefill(x1, commit_after=True)
        self.assertEqual(commits, [])

    def test_stencil_with_out_of_bounds_x0(self):
        x_bad = self.x0.copy()
        x_bad[0] = float(self.bounds.ub[0]) + 1.0
        points = collect_fd_stencil_points(x_bad, 0.01, self.bounds)
        self.assertGreater(len(points), 1)

    def test_prefill_with_out_of_bounds_x0(self):
        x_bad = self.x0.copy()
        x_bad[0] = float(self.bounds.ub[0]) + 1.0
        clipped = clip_to_bounds(x_bad, self.bounds)
        self.assertLessEqual(clipped[0], self.bounds.ub[0])
        fd = ParallelFiniteDifferences(self.task, self.config, 0.01, self.bounds)
        fd.prefill(x_bad)
        self.assertGreater(len(fd._fd_cache_map), 0)


class TestParallelFDEvalCache(unittest.TestCase):
    def setUp(self):
        RosenSolver.eval_count = 0
        self.model = SimpleVector(0.4, 0.1)
        self.model.cache_signature_fields = ("x1", "x2")
        opt_vars = {"x1": {"min": 0.0, "max": 1.0}, "x2": {"min": -0.5, "max": 2.0}}
        constraints = {"ineq1": 0.0, "ineq2": 0.0}
        self.bounds = Bounds(
            lb=np.array([-1.0, -0.25]),
            ub=np.array([1.0, 1.0]),
        )
        self.x0 = np.array([0.4 * 2.0, 0.1 * 1.5], dtype=float)

    def _make_task(
        self,
        num_proc: int,
        parallel_fd_workers: bool | None = None,
    ) -> EvalCacheTask:
        if parallel_fd_workers is None:
            parallel_fd_workers = num_proc > 1
        config = OptimizationConfig(
            num_proc=num_proc,
            parallel_fd_workers=parallel_fd_workers,
            max_iter=3,
            logging_dir=".",
            calculation_dir=".",
        )
        solver = RosenSolver(config)
        task = EvalCacheTask(
            self.model,
            "rosen_eval_cache",
            OptConditions(
                {"x1": {"min": 0.0, "max": 1.0}, "x2": {"min": -0.5, "max": 2.0}},
                {"ineq1": 0.0, "ineq2": 0.0},
            ),
            solver,
            config,
        )
        task.cost_function_normalization = 1.0
        return task

    def test_prefill_invalidates_stale_eval_cache(self):
        task = self._make_task(num_proc=1)
        task.objective(self.x0)
        fd = ParallelFiniteDifferences(task, task.config, 0.01, self.bounds)
        fd.prefill(self.x0)
        center_model = model_at_x_norm(fd._ctx, self.x0)
        signature = center_model.signature()
        original_value = fd._fd_cache_map[signature]["ineq1"]
        stale_value = original_value + 0.25
        fd._fd_cache_map[signature] = {
            **fd._fd_cache_map[signature],
            "ineq1": stale_value,
        }

        fd._prefill_memo_key = None
        fd.prefill(self.x0)
        self.assertIsNone(task._eval_cache_results)

        constraint_fun = fd._constraint_fun(0)
        self.assertAlmostEqual(constraint_fun(self.x0), original_value)

    def test_constraint_jac_uses_cache_map_after_prefill(self):
        task = self._make_task(num_proc=1)
        task.objective(self.x0)
        fd = ParallelFiniteDifferences(task, task.config, 0.01, self.bounds)
        fd.setup()
        try:
            jac_fun = fd.make_constraint_jac(0)
            jac = jac_fun(self.x0)
        finally:
            fd.close()
        self.assertEqual(jac.shape, (2,))
        self.assertTrue(np.all(np.isfinite(jac)))

    def test_prefill_workers_do_not_write_main_solver_cache(self):
        task = self._make_task(num_proc=2, parallel_fd_workers=True)
        task.objective(self.x0)
        main_cache_size = len(task.solver.cache_map)
        fd = ParallelFiniteDifferences(task, task.config, 0.01, self.bounds)
        fd.setup()
        try:
            fd.prefill(self.x0)
        finally:
            fd.close()
        self.assertGreater(len(fd._fd_cache_map), 1)
        self.assertEqual(len(task.solver.cache_map), main_cache_size)

    def test_serial_and_parallel_constraint_jac_match(self):
        serial_task = self._make_task(num_proc=1, parallel_fd_workers=False)
        parallel_task = self._make_task(num_proc=2, parallel_fd_workers=True)

        serial_fd = ParallelFiniteDifferences(serial_task, serial_task.config, 0.01, self.bounds)
        parallel_fd = ParallelFiniteDifferences(parallel_task, parallel_task.config, 0.01, self.bounds)
        serial_fd.setup()
        parallel_fd.setup()
        try:
            serial_task.objective(self.x0)
            parallel_task.objective(self.x0)
            serial_jac = serial_fd.make_constraint_jac(0)(self.x0)
            parallel_jac = parallel_fd.make_constraint_jac(0)(self.x0)
        finally:
            serial_fd.close()
            parallel_fd.close()

        np.testing.assert_allclose(serial_jac, parallel_jac, rtol=1e-5, atol=1e-5)

    def test_invalidate_helper_calls_task_method(self):
        task = self._make_task(num_proc=1)
        task.objective(self.x0)
        self.assertIsNotNone(task._eval_cache_results)
        _invalidate_task_eval_cache(task)
        self.assertIsNone(task._eval_cache_results)
        self.assertIsNone(task._eval_cache_signature)

    def test_prefill_stencil_includes_center_point(self):
        rel_step = 0.01
        stencil = collect_fd_stencil_points(self.x0, rel_step, self.bounds)
        combined = _dedupe_points([self.x0] + stencil)
        self.assertTrue(
            any(np.allclose(point, self.x0) for point in combined),
            "FD prefill must include the center design point",
        )

    def test_constraint_fun_reads_fd_cache_not_eval_cache(self):
        task = self._make_task(num_proc=1)
        fd = ParallelFiniteDifferences(task, task.config, 0.01, self.bounds)
        fd.prefill(self.x0)
        center_model = model_at_x_norm(fd._ctx, self.x0)
        signature = center_model.signature()
        cached_value = fd._fd_cache_map[signature]["ineq1"] + 0.4
        fd._fd_cache_map[signature] = {
            **fd._fd_cache_map[signature],
            "ineq1": cached_value,
        }
        task.objective(self.x0)
        self.assertNotEqual(task._eval_cache_results["ineq1"], cached_value)

        constraint_fun = fd._constraint_fun(0)
        self.assertAlmostEqual(constraint_fun(self.x0), cached_value)

    def test_prefill_main_thread_matches_solver_cache(self):
        task = self._make_task(num_proc=1)
        task.objective(self.x0)
        main_cache_size = len(task.solver.cache_map)
        fd = ParallelFiniteDifferences(task, task.config, 0.01, self.bounds)
        fd.prefill(self.x0)
        self.assertGreater(len(fd._fd_cache_map), 1)
        self.assertGreater(len(task.solver.cache_map), main_cache_size)
        for signature, results in fd._fd_cache_map.items():
            self.assertIn(signature, task.solver.cache_map)
            self.assertIs(results, task.solver.cache_map[signature])

    def test_anchor_center_reuses_main_cache(self):
        task = self._make_task(num_proc=1)
        fd = ParallelFiniteDifferences(task, task.config, 0.01, self.bounds)
        task.objective(self.x0)
        center_model = model_at_x_norm(fd._ctx, self.x0)
        center_signature = center_model.signature()
        solve_calls = RosenSolver.eval_count
        fd._anchor_center_from_main_cache(self.x0)
        self.assertEqual(RosenSolver.eval_count, solve_calls)
        self.assertIs(
            fd._fd_cache_map[center_signature],
            task.solver.cache_map[center_signature],
        )


if __name__ == "__main__":
    unittest.main()
