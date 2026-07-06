"""Parallel finite-difference gradients with cache prefill for expensive solvers."""

from __future__ import annotations

import copy
import logging
import multiprocessing
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize._numdiff import approx_derivative

logger = logging.getLogger(__name__)

_PROCESS_SOLVER: Any = None
_PROCESS_CTX: Any = None


@dataclass
class FDEvaluationContext:
    model: Any
    conversion_map: Dict[int, str]
    denorm_coefficients: List[float]
    unique_id: str
    cost_function_normalization: float
    x_to_model: Callable[..., None]


def model_at_x_norm(ctx: FDEvaluationContext, x_norm: np.ndarray) -> Any:
    x_denorm = [
        float(x_norm[i]) * ctx.denorm_coefficients[i]
        for i in range(len(ctx.denorm_coefficients))
    ]
    make_eval_copy = getattr(ctx.model, "make_eval_copy", None)
    if callable(make_eval_copy):
        inner_model = make_eval_copy()
    else:
        inner_model = copy.deepcopy(ctx.model)
    ctx.x_to_model(inner_model, x_denorm, ctx.conversion_map)
    return inner_model


def _normalize_level2_changed_vars(solve_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    changed = solve_kwargs.get("level2_changed_vars")
    if changed is None:
        return solve_kwargs
    normalized = dict(solve_kwargs)
    normalized["level2_changed_vars"] = set(changed)
    return normalized


def _fd_solve_context(opt_task: Any, x_norm: np.ndarray | None) -> Tuple[Dict[str, Any], Any]:
    solve_kwargs: Dict[str, Any] = {}
    if x_norm is not None:
        kwargs_fn = getattr(opt_task, "fd_solve_kwargs_for_x_norm", None)
        if callable(kwargs_fn):
            solve_kwargs = dict(kwargs_fn(np.asarray(x_norm, dtype=float)))
    baseline_payload = None
    baseline_fn = getattr(opt_task, "level2_baseline_payload_for_fd", None)
    if callable(baseline_fn):
        baseline_payload = baseline_fn()
    return _normalize_level2_changed_vars(solve_kwargs), baseline_payload


def _apply_level2_baseline_payload(solver: Any, baseline_payload: Any) -> None:
    if baseline_payload is None:
        return
    apply_fn = getattr(solver, "apply_level2_baseline_payload", None)
    if callable(apply_fn):
        apply_fn(baseline_payload)


def _solver_solve_for_fd(
    opt_task: Any,
    solver: Any,
    model: Any,
    x_norm: np.ndarray | None,
) -> Dict[str, Any]:
    solve_kwargs, baseline_payload = _fd_solve_context(opt_task, x_norm)
    _apply_level2_baseline_payload(solver, baseline_payload)
    unique_id = getattr(opt_task, "unique_id", "")
    return solver.solve(model, unique_id, res_type=None, **solve_kwargs)


def _make_fd_job(opt_task: Any, x_norm: np.ndarray) -> Tuple[List[float], Dict[str, Any], Any]:
    solve_kwargs, baseline_payload = _fd_solve_context(opt_task, x_norm)
    return (np.asarray(x_norm, dtype=float).tolist(), solve_kwargs, baseline_payload)


def _maybe_capture_level2_baseline(opt_task: Any, x_norm: np.ndarray) -> None:
    capture_fn = getattr(opt_task, "maybe_capture_level2_baseline_for_x_norm", None)
    if callable(capture_fn):
        capture_fn(np.asarray(x_norm, dtype=float))


def _process_worker_init(
    payloads_bytes: bytes,
    slot_counter: Any,
    slot_lock: Any,
    worker_count: int,
) -> None:
    global _PROCESS_SOLVER, _PROCESS_CTX
    with slot_lock:
        slot = slot_counter.value % worker_count
        slot_counter.value += 1
    payloads = pickle.loads(payloads_bytes)
    _PROCESS_SOLVER, _PROCESS_CTX = payloads[slot]


def _process_eval_job(job: Tuple[Any, ...]) -> Tuple[Any, Dict[str, Any]]:
    if _PROCESS_SOLVER is None or _PROCESS_CTX is None:
        raise RuntimeError("FD process worker is not initialized")
    x_norm_list = job[0]
    solve_kwargs = job[1] if len(job) > 1 else {}
    baseline_payload = job[2] if len(job) > 2 else None
    x_norm = np.asarray(x_norm_list, dtype=float)
    model = model_at_x_norm(_PROCESS_CTX, x_norm)
    signature = model.signature()
    _apply_level2_baseline_payload(_PROCESS_SOLVER, baseline_payload)
    results = _PROCESS_SOLVER.solve(
        model,
        _PROCESS_CTX.unique_id,
        res_type=None,
        **_normalize_level2_changed_vars(solve_kwargs),
    )
    return signature, results


def _approx_grad(fun, x_arr: np.ndarray, rel_step: float, f0: float, bounds) -> np.ndarray:
    result = approx_derivative(
        fun,
        x_arr,
        method="3-point",
        rel_step=rel_step,
        f0=f0,
        bounds=_bounds_tuple(bounds),
    )
    if isinstance(result, tuple):
        grad = result[0]
    else:
        grad = result
    return np.atleast_1d(np.asarray(grad, dtype=float))


def _approx_jac(
    fun,
    x_arr: np.ndarray,
    rel_step: float,
    f0: np.ndarray,
    bounds,
) -> np.ndarray:
    result = approx_derivative(
        fun,
        x_arr,
        method="3-point",
        rel_step=rel_step,
        f0=f0,
        bounds=_bounds_tuple(bounds),
    )
    if isinstance(result, tuple):
        jac = result[0]
    else:
        jac = result
    return np.atleast_2d(np.asarray(jac, dtype=float))


def _bounds_tuple(bounds: Bounds | Tuple[np.ndarray, np.ndarray] | None) -> Tuple[np.ndarray, np.ndarray] | None:
    if bounds is None:
        return None
    if isinstance(bounds, Bounds):
        return np.asarray(bounds.lb, dtype=float), np.asarray(bounds.ub, dtype=float)
    lb, ub = bounds
    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def clip_to_bounds(
    x: np.ndarray,
    bounds: Bounds | Tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Project normalized x into scipy Bounds (handles out-of-range initial designs)."""
    bt = _bounds_tuple(bounds)
    if bt is None:
        return np.asarray(x, dtype=float)
    lb, ub = bt
    clipped = np.clip(np.asarray(x, dtype=float), lb, ub)
    if not np.allclose(clipped, x):
        logger.debug("clipped FD point into bounds: %s -> %s", x, clipped)
    return clipped


def _invalidate_task_eval_cache(opt_task: Any) -> None:
    invalidate = getattr(opt_task, "invalidate_eval_cache", None)
    if callable(invalidate):
        invalidate()


def _log_fd_design_point(
    opt_task: Any,
    x_norm: np.ndarray,
    results: Dict[str, Any],
) -> None:
    log_fn = getattr(opt_task, "log_fd_design_point", None)
    if callable(log_fn):
        log_fn(x_norm, results)


def _dedupe_points(points: List[np.ndarray]) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    seen: set[Tuple[float, ...]] = set()
    for point in points:
        key = tuple(np.round(point, 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return unique


def collect_fd_stencil_points(
    x0: np.ndarray,
    rel_step: float,
    bounds: Bounds | Tuple[np.ndarray, np.ndarray] | None,
) -> List[np.ndarray]:
    """Collect normalized x points that scipy 3-point FD would evaluate."""
    collected: List[np.ndarray] = []

    def collecting_fun(x: np.ndarray) -> float:
        collected.append(np.asarray(x, dtype=float).copy())
        return 0.0

    x0_arr = clip_to_bounds(np.asarray(x0, dtype=float), bounds)
    approx_derivative(
        collecting_fun,
        x0_arr,
        method="3-point",
        rel_step=rel_step,
        f0=0.0,
        bounds=_bounds_tuple(bounds),
    )
    return _dedupe_points(collected)


class ParallelFiniteDifferences:
    """Parallel 3-point FD with cache prefill; one long-lived solver clone per process."""

    def __init__(
        self,
        opt_task: Any,
        config: Any,
        rel_step: float,
        bounds: Bounds,
    ) -> None:
        self.opt_task = opt_task
        self.config = config
        self.rel_step = rel_step
        self.bounds = bounds
        self._executor: ProcessPoolExecutor | None = None
        self._mp_manager: multiprocessing.managers.SyncManager | None = None
        self._prefill_lock = threading.Lock()
        self._prefill_memo_key: Tuple[float, ...] | None = None
        self._fd_cache_map: Dict[Any, Dict[str, Any]] = {}
        self._jac_center_key: Tuple[float, ...] | None = None
        self._objective_grad: np.ndarray | None = None
        self._constraint_jac: np.ndarray | None = None
        self._last_prefill_points: List[np.ndarray] = []
        self._ctx = self._build_context()

    def _build_context(self) -> FDEvaluationContext:
        return FDEvaluationContext(
            model=self.opt_task.model,
            conversion_map=dict(self.opt_task.conversion_map),
            denorm_coefficients=list(self.opt_task.denorm_coefficients),
            unique_id=self.opt_task.unique_id,
            cost_function_normalization=float(
                self.opt_task.cost_function_normalization or 1.0
            ),
            x_to_model=self.opt_task.x_to_model,
        )

    def _use_fd_workers(self) -> bool:
        return bool(getattr(self.config, "parallel_fd_workers", False))

    def setup(self) -> None:
        """Create a persistent process pool for the whole SLSQP run."""
        if self._use_fd_workers():
            self._ensure_process_pool()
        else:
            logger.info(
                "parallel FD: main-thread stencil prefill (parallel_fd_workers=False)"
            )

    def _ensure_process_pool(self) -> None:
        if not self._use_fd_workers():
            return
        if self.config.num_proc <= 1:
            return
        if self._executor is not None:
            return
        worker_count = int(self.config.num_proc)
        self._mp_manager = multiprocessing.Manager()
        slot_counter = self._mp_manager.Value("i", 0)
        slot_lock = self._mp_manager.Lock()
        payloads = [
            (
                self.opt_task.solver.clone_for_parallel_eval(str(index)),
                self._ctx,
            )
            for index in range(worker_count)
        ]
        payloads_bytes = pickle.dumps(payloads)
        self._executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_process_worker_init,
            initargs=(payloads_bytes, slot_counter, slot_lock, worker_count),
        )
        logger.info(
            "parallel FD: persistent ProcessPool with %s worker process(es)",
            worker_count,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._mp_manager is not None:
            self._mp_manager.shutdown()
            self._mp_manager = None

    def __enter__(self) -> "ParallelFiniteDifferences":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def prefill(self, x_norm: np.ndarray, *, commit_after: bool = True) -> None:
        del commit_after  # sensitivity commits only on SLSQP callback
        x_arr = clip_to_bounds(np.asarray(x_norm, dtype=float), self.bounds)
        key = tuple(np.round(x_arr, 12))
        with self._prefill_lock:
            if key == self._prefill_memo_key:
                return
            self._prefill_memo_key = key
            self._fd_cache_map.clear()
            self._invalidate_jacobian_memo()
        points = _dedupe_points(
            [x_arr] + collect_fd_stencil_points(x_arr, self.rel_step, self.bounds)
        )
        self._prefill_points(points, x_arr)
        self._last_prefill_points = [np.asarray(point, dtype=float).copy() for point in points]
        self._maybe_validate_prefilled_stencils(x_arr, points)

    def _prefill_points(
        self,
        points: List[np.ndarray],
        center: np.ndarray,
    ) -> None:
        center_arr = np.asarray(center, dtype=float)
        perturbations = [
            point
            for point in points
            if not np.allclose(point, center_arr, rtol=0.0, atol=0.0)
        ]
        self._anchor_center_from_main_cache(center_arr)

        missing: List[np.ndarray] = []
        for point in perturbations:
            model = model_at_x_norm(self._ctx, point)
            signature = model.signature()
            if signature not in self._fd_cache_map:
                missing.append(point)
        if missing:
            if self._use_fd_workers():
                self._ensure_process_pool()
                if self._executor is None:
                    for point in missing:
                        self._eval_and_cache_on_main(point)
                else:
                    jobs = [_make_fd_job(self.opt_task, point) for point in missing]
                    started = time.perf_counter()
                    for point, (signature, results) in zip(
                        missing,
                        self._executor.map(_process_eval_job, jobs),
                    ):
                        self._fd_cache_map[signature] = results
                        _log_fd_design_point(self.opt_task, point, results)
                    logger.info(
                        "parallel FD prefill: %s worker point(s) in %.3fs",
                        len(missing),
                        time.perf_counter() - started,
                    )
            else:
                started = time.perf_counter()
                for point in missing:
                    self._eval_and_cache_on_main(point)
                logger.info(
                    "parallel FD prefill: %s main-thread point(s) in %.3fs",
                    len(missing),
                    time.perf_counter() - started,
                )
        _invalidate_task_eval_cache(self.opt_task)

    def _maybe_validate_prefilled_stencils(
        self,
        center: np.ndarray,
        points: List[np.ndarray],
    ) -> None:
        validate = getattr(self.opt_task, "validate_and_retry_fd_stencils", None)
        if not callable(validate):
            return

        entries: List[Dict[str, Any]] = []
        for point in points:
            point_arr = np.asarray(point, dtype=float)
            model = model_at_x_norm(self._ctx, point_arr)
            results = self._lookup_results(model, point_arr)
            entries.append({
                "x_norm": point_arr.tolist(),
                "results": results,
            })

        updates = validate(np.asarray(center, dtype=float), entries)
        if not updates:
            return

        for update in updates:
            point_arr = np.asarray(update["x_norm"], dtype=float)
            model = model_at_x_norm(self._ctx, point_arr)
            self._fd_cache_map[model.signature()] = update["results"]

        _invalidate_task_eval_cache(self.opt_task)
        self._invalidate_jacobian_memo()

    def _anchor_center_from_main_cache(self, center: np.ndarray) -> None:
        """FD center uses main-thread cache; avoid a second Nastran run after worker prefill."""
        center_arr = np.asarray(center, dtype=float)
        model = model_at_x_norm(self._ctx, center_arr)
        signature = model.signature()
        main_cache = self.opt_task.solver.cache_map
        if signature in main_cache:
            self._fd_cache_map[signature] = main_cache[signature]
        else:
            results = _solver_solve_for_fd(
                self.opt_task,
                self.opt_task.solver,
                model,
                center_arr,
            )
            self._fd_cache_map[signature] = results
            _log_fd_design_point(self.opt_task, center_arr, results)
        _maybe_capture_level2_baseline(self.opt_task, center_arr)

    def _eval_and_cache_on_main(self, x_norm: np.ndarray) -> None:
        """Match serial scipy FD: main-thread solve() with the task unique_id."""
        x_arr = np.asarray(x_norm, dtype=float)
        model = model_at_x_norm(self._ctx, x_arr)
        signature = model.signature()
        if signature in self._fd_cache_map:
            return
        results = _solver_solve_for_fd(
            self.opt_task,
            self.opt_task.solver,
            model,
            x_arr,
        )
        self._fd_cache_map[signature] = results
        _log_fd_design_point(self.opt_task, x_arr, results)

    def _lookup_results(self, model: Any, x_norm: np.ndarray | None = None) -> Dict[str, Any]:
        signature = model.signature()
        if signature in self._fd_cache_map:
            return self._fd_cache_map[signature]
        results = _solver_solve_for_fd(
            self.opt_task,
            self.opt_task.solver,
            model,
            x_norm,
        )
        if x_norm is not None:
            _log_fd_design_point(self.opt_task, x_norm, results)
        return results

    def _invalidate_jacobian_memo(self) -> None:
        self._jac_center_key = None
        self._objective_grad = None
        self._constraint_jac = None

    def _constraint_specs(self) -> List[Tuple[str | None, float | None]]:
        specs: List[Tuple[str | None, float | None]] = []
        for constraint in self.opt_task.cons:
            raw_fun = constraint["fun"]
            specs.append((
                getattr(raw_fun, "parameter", None),
                getattr(raw_fun, "limit", None),
            ))
        return specs

    @staticmethod
    def _normalized_constraint_value(value: float, limit: float | None) -> float:
        if limit not in (None, 0):
            return value / float(limit) - 1.0
        return value

    def _results_at_x_norm(self, x_norm: np.ndarray) -> Dict[str, Any]:
        x_arr = np.asarray(x_norm, dtype=float)
        model = model_at_x_norm(self._ctx, x_arr)
        return self._lookup_results(model, x_arr)

    def _constraint_vector_at(self, x_norm: np.ndarray) -> np.ndarray:
        results = self._results_at_x_norm(x_norm)
        values: List[float] = []
        for parameter, limit in self._constraint_specs():
            if parameter is None:
                raise RuntimeError("Batched constraint Jacobian requires constraint parameters")
            values.append(
                self._normalized_constraint_value(
                    float(results[parameter]),
                    limit,
                )
            )
        return np.asarray(values, dtype=float)

    def _ensure_jacobians_built(self, x_arr: np.ndarray) -> None:
        key = tuple(np.round(x_arr, 12))
        if key == self._jac_center_key:
            return
        self.prefill(x_arr)
        objective_f0 = self._objective_fun(x_arr)
        self._objective_grad = _approx_grad(
            self._objective_fun,
            x_arr,
            self.rel_step,
            objective_f0,
            self.bounds,
        )
        constraint_specs = self._constraint_specs()
        if constraint_specs:
            constraint_f0 = self._constraint_vector_at(x_arr)
            self._constraint_jac = _approx_jac(
                self._constraint_vector_at,
                x_arr,
                self.rel_step,
                constraint_f0,
                self.bounds,
            )
        else:
            self._constraint_jac = np.zeros((0, x_arr.size), dtype=float)
        self._jac_center_key = key
        logger.info(
            "parallel FD jacobians: n=%s m=%s built",
            x_arr.size,
            0 if self._constraint_jac is None else self._constraint_jac.shape[0],
        )

    def _objective_fun(self, x_norm: np.ndarray) -> float:
        x_arr = np.asarray(x_norm, dtype=float)
        model = model_at_x_norm(self._ctx, x_arr)
        value = float(self._lookup_results(model, x_arr)["objective"])
        return value / self._ctx.cost_function_normalization

    def _constraint_fun(self, constraint_index: int) -> Callable[[np.ndarray], float]:
        raw_fun = self.opt_task.cons[constraint_index]["fun"]
        parameter = getattr(raw_fun, "parameter", None)
        limit = getattr(raw_fun, "limit", None)
        if parameter is None:
            return raw_fun

        def fun(x_norm: np.ndarray) -> float:
            x_arr = np.asarray(x_norm, dtype=float)
            model = model_at_x_norm(self._ctx, x_arr)
            value = float(self._lookup_results(model, x_arr)[parameter])
            if limit != 0:
                return value / float(limit) - 1.0
            return value

        return fun

    def objective_jac(self, x_norm: np.ndarray) -> np.ndarray:
        x_arr = clip_to_bounds(np.asarray(x_norm, dtype=float), self.bounds)
        self._ensure_jacobians_built(x_arr)
        assert self._objective_grad is not None
        return self._objective_grad.copy()

    def make_constraint_jac(self, constraint_index: int) -> Callable[[np.ndarray], np.ndarray]:
        def jac(x_norm: np.ndarray) -> np.ndarray:
            x_arr = clip_to_bounds(np.asarray(x_norm, dtype=float), self.bounds)
            self._ensure_jacobians_built(x_arr)
            assert self._constraint_jac is not None
            return self._constraint_jac[constraint_index, :].copy()

        return jac

    def attach_constraint_jacs(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for index, constraint in enumerate(constraints):
            item = dict(constraint)
            item["fun"] = self._constraint_fun(index)
            if item.get("jac") is None:
                item["jac"] = self.make_constraint_jac(index)
            enriched.append(item)
        return enriched
