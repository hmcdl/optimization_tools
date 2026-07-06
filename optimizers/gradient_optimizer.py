"""
Градиентный оптимизатор на основе SLSQP
"""
import copy
import json
import logging
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize, Bounds

# from exceptions import AllLoadsZeroException
from ..config import OptimizationConfig
from optimization_tools.abstract_solver import AbstractSolver
from optimization_tools.constraints_creators import ConstraintForNormalized
from optimization_tools.exceptions import SolverError
from optimization_tools.opt_conditions import OptimizationTaskResults
from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer, AbstractOptimizationTask

from .. import opt_tools_settings
from ..parallel_fd import ParallelFiniteDifferences, clip_to_bounds
from ..utils import constraints_are_satisfied
# print(f"MAX_ITER {get_max_iter()}")

class OptimizationTaskWithNormalization(AbstractOptimizationTask):
    """
    Оптимизационная задача с нормализацией переменных.
    """
    def __init__(
        self, 
        initial_state, 
        unique_id, 
        opt_conditions, 
        solver: AbstractSolver, 
        config: OptimizationConfig,
        optimization_dir: str = None
    ) -> None:
        super().__init__(initial_state, unique_id, opt_conditions, solver, config, optimization_dir)
        self.lower_bounds = []
        self.upper_bounds = []
        self.cost_function_normalization = None
        self.history = []
        self._update_bounds_and_constraints()

    def _update_bounds_and_constraints(self):
        """Обновление границ и ограничений"""
        self.lower_bounds = []
        self.upper_bounds = []
        opt_vars_map = self.opt_conditions.vars
        var_values_with_bounds = {}
        for key in opt_vars_map:
            var_values_with_bounds[key] = {"cur_val": getattr(self._model, key), "bounds": opt_vars_map[key]}

        for i, _ in enumerate(var_values_with_bounds):
            var = self.conversion_map[i]
            self.lower_bounds.append(var_values_with_bounds[var]["bounds"]["min"])
            self.upper_bounds.append(var_values_with_bounds[var]["bounds"]["max"])

        # нормализация, чтобы все было в пределах -1...1
        self.normalization_coefficients = [2/(self.lower_bounds[i] + self.upper_bounds[i]) 
                                           for i in range(len(self.lower_bounds))]
        self.denorm_coefficients = [0.5*(self.lower_bounds[i] + self.upper_bounds[i]) 
                                    for i in range(len(self.lower_bounds))]
        
        self.cons = []
        for constraint in self.opt_conditions.constraints:
            self.cons.append({
                    'type': 'ineq',
                        'fun': ConstraintForNormalized(self,
                            constraint,
                            self.opt_conditions.constraints[constraint],
                            self.denorm_coefficients)
                })

    def update_opt_vars(self):
        """Обновление переменных оптимизации"""
        self._update_bounds_and_constraints()

    def get_vars_dict(self, x: list):
        vars_dict = {}
        conversion_map = self.get_conversion_map()
        x_denorm = [x[i] * self.denorm_coefficients[i] for i in range(len(self.lower_bounds))]
        for i, _ in enumerate(x_denorm):
            vars_dict[conversion_map[i]] = x_denorm[i]

        return vars_dict

    def objective(self, x):
        x = [float(x_component) for x_component in x]
        logger = logging.getLogger(self.local_log_path + "solver_log")
        cur_values_map = {}
        for i,val in enumerate(x):
            cur_values_map[self.conversion_map[i]] = val
        
        x_denorm = [x[i] * self.denorm_coefficients[i] for i in range(len(self.lower_bounds))]
        logger.info(x_denorm)
        inner_model = copy.deepcopy(self.model)
        self.x_to_model(inner_model, x_denorm, self.conversion_map)
        result = self.solver.solve(inner_model, self.unique_id, "objective")
        logger.info(f"objective: {result}")
        return result / self.cost_function_normalization


class GradientOptimizer(AbstractOPtimizer):
    def __init__(
        self, 
        optimized_object: OptimizationTaskWithNormalization, 
        config: OptimizationConfig,
        first_approx_function=None
    ):
        super().__init__(optimized_object, config)
        self.first_approx_function = first_approx_function
        self.history: list[dict] = []

    def _use_parallel_fd(self) -> bool:
        if bool(self.config.extra.get("sensitivity_stencil_validation", False)):
            return True
        if self.config.num_proc <= 1:
            return False
        if self.config.parallel_fd is not None:
            return bool(self.config.parallel_fd)
        return True

    def _find_last_feasible_history_point(self) -> dict | None:
        limits = self.optimized_object.opt_conditions.constraints
        for history_point in reversed(self.history):
            if constraints_are_satisfied(history_point.get("constraints"), limits):
                return history_point
        return None

    def _apply_history_point(self, history_point: dict) -> tuple[dict, dict, float]:
        vars_dict = history_point["vars"]
        vars_list = [
            vars_dict[self.optimized_object.conversion_map[i]]
            for i in range(len(self.optimized_object.conversion_map))
        ]
        self.optimized_object.x_to_model(
            self.optimized_object.model,
            vars_list,
            self.optimized_object.conversion_map,
        )

        result_vars_map = {
            var: getattr(self.optimized_object.model, var)
            for var in self.optimized_object.opt_conditions.vars
        }
        constraint_values = history_point["constraints"]
        final_constraints = {
            constraint_name: constraint_values[constraint_name]
            for constraint_name in self.optimized_object.opt_conditions.constraints
        }
        objective = constraint_values.get("objective")
        if objective is None:
            objective = self.optimized_object.solver.solve(
                self.optimized_object.model,
                self.optimized_object.unique_id,
                "objective",
            )
        return result_vars_map, final_constraints, objective

    def callback(self, x):
        if len(self.history)  == 4:
            debug = 1
        vars_dict = self.optimized_object.get_vars_dict(x)
        vars_list = [vars_dict[var] for var in vars_dict]
        model = copy.deepcopy(self.optimized_object.model)
        self.optimized_object.x_to_model(model=model, x=vars_list, conversion_map=self.optimized_object.conversion_map)
        constraint_values = self.optimized_object.solver.solve(model, self.optimized_object.unique_id + "_callback", None)
        self.history.append({"vars" : vars_dict, "constraints": constraint_values})
        self.optimized_object.history = self.history

    def optimize(self, **kwargs) -> OptimizationTaskResults:
        # Используем self.config.max_iter
        options = {
            'maxiter': self.config.max_iter, 
            'disp': False, 
            "finite_diff_rel_step": 0.01,
            'ftol': 1e-6
        }
        try:
            self.optimized_object.update_opt_vars()
            logger = self.logger
            logger.debug("Gradient optimization started")
            time_start = time.time()
            if self.first_approx_function:
                initial_results = self.optimized_object.solver.solve(
                    self.optimized_object.model, 
                    self.optimized_object.unique_id, 
                    None
                )
                self.first_approx_function(
                    results_map=initial_results, 
                    panel=self.optimized_object.model, 
                    opt_conditions=self.optimized_object.opt_conditions, 
                    logger=self.logger
                )

            x0 = self.optimized_object.get_x()
            bounds = Bounds([self.optimized_object.lower_bounds[i] *
                              self.optimized_object.normalization_coefficients[i]
                                for i in range(len(self.optimized_object.lower_bounds))], 
                        [self.optimized_object.upper_bounds[i] *
                          self.optimized_object.normalization_coefficients[i] 
                          for i in range(len(self.optimized_object.lower_bounds))])
            x0_normalized = clip_to_bounds(
                np.array(
                    [
                        x0[i] * self.optimized_object.normalization_coefficients[i]
                        for i in range(len(self.optimized_object.lower_bounds))
                    ],
                    dtype=float,
                ),
                bounds,
            )
            initial_results = self.optimized_object.solver.solve(
                self.optimized_object.model, 
                self.optimized_object.unique_id, 
                None
            )
            cost_fun_coeff = initial_results["objective"]
            self.optimized_object.cost_function_normalization = cost_fun_coeff if cost_fun_coeff != 0 else 1
            sim_start_time = time.time()
            logger.info(json.dumps(self.optimized_object.opt_conditions.vars))
            logger.info(f"SLSQP started at {sim_start_time}")

            finite_diff_rel_step = options.get("finite_diff_rel_step", 0.01)
            jac = "3-point"
            constraints = self.optimized_object.cons
            callback = self.callback
            parallel_fd = None
            if self._use_parallel_fd():
                parallel_fd = ParallelFiniteDifferences(
                    self.optimized_object,
                    self.config,
                    finite_diff_rel_step,
                    bounds,
                )
                parallel_fd.setup()
                parallel_fd.prefill(x0_normalized, commit_after=False)
                jac = parallel_fd.objective_jac
                constraints = parallel_fd.attach_constraint_jacs(constraints)
                if self.config.prefetch_fd_in_callback:
                    base_callback = self.callback

                    def callback_with_prefetch(x, _base=base_callback, _fd=parallel_fd, _bounds=bounds):
                        result = _base(x)
                        _fd.prefill(
                            clip_to_bounds(np.asarray(x, dtype=float), _bounds),
                            commit_after=False,
                        )
                        return result

                    callback = callback_with_prefetch

            try:
                res = minimize(
                    self.optimized_object.objective,
                    x0_normalized,
                    method="SLSQP",
                    jac=jac,
                    constraints=constraints,
                    bounds=bounds,
                    options=options,
                    callback=callback,
                )
            finally:
                if parallel_fd is not None:
                    parallel_fd.close()
            
            logger.info(f"SLSQP finished at {sim_start_time} with status {res.status}")
            logger.info(f"SLSQP duration {time.time() - sim_start_time}")
            res_x_denormalized = [res.x[i] * self.optimized_object.denorm_coefficients[i]
                                   for i in range(len(self.optimized_object.lower_bounds))]
            self.optimized_object.x_to_model(self.optimized_object.model,
                                              res_x_denormalized, self.optimized_object.conversion_map)
            
            objective = self.optimized_object.objective(res.x) * self.optimized_object.cost_function_normalization
            result_vars_map = {}
            for var in self.optimized_object.opt_conditions.vars:
                result_vars_map[var] = getattr(self.optimized_object.model, var)

            final_constraints = {}
            for i, constraint_name in enumerate(self.optimized_object.opt_conditions.constraints):
                if self.optimized_object.opt_conditions.constraints[constraint_name] != 0:
                    constraint = (self.optimized_object.cons[i]['fun'](res.x) + 1) * \
                        self.optimized_object.opt_conditions.constraints[constraint_name]
                else:
                    constraint = self.optimized_object.cons[i]['fun'](res.x)
                final_constraints[constraint_name] = constraint

            if self.config.avoid_constraints_violations:
                limits = self.optimized_object.opt_conditions.constraints
                if not constraints_are_satisfied(final_constraints, limits):
                    logger.warning(
                        "%s Final point violates constraints, searching history for last feasible point",
                        self.optimized_object.unique_id,
                    )
                    fallback_point = self._find_last_feasible_history_point()
                    if fallback_point is not None:
                        logger.info(
                            "%s Using feasible point from optimization history instead of final SLSQP result",
                            self.optimized_object.unique_id,
                        )
                        result_vars_map, final_constraints, objective = self._apply_history_point(
                            fallback_point
                        )
                    else:
                        logger.warning(
                            "%s avoid_constraints_violations is enabled but no feasible point found in history",
                            self.optimized_object.unique_id,
                        )

            logger.info(f"MAX_ITER {opt_tools_settings.get(self.config.max_iter)}")
            logger.info(f"Optimization duration {time.time() - time_start}")
            logger.info("Result variables:")
            logger.info(json.dumps(result_vars_map, indent=2))
            logger.info("Margin values:")
            logger.info(json.dumps(final_constraints, indent=2))
            logger.info(f"objective = {str(objective)}")
            logger.info(json.dumps(self.history))
            results : OptimizationTaskResults = OptimizationTaskResults(
                0, 0, result_vars_map, final_constraints, objective, self.optimized_object.model,
                opt_conditions=self.optimized_object.opt_conditions)
            results.history = self.history
            return results
        except SolverError:
            logger.critical("%s Optimization failed due to unhandled exception during optimization" %  self.optimized_object.unique_id)
            logger.exception("%s exception: " % self.optimized_object.unique_id)
            return OptimizationTaskResults(
                1, 1, None, None, None, self.optimized_object.model)
        # except AllLoadsZeroException:
        #     logger.warning(f"all loads zero for panel {self.optimized_object.unique_id}")
        #     return OptimizationTaskResults(
        #         1, 1, None, None, None, self.optimized_object.model)
        finally:
            logger.info("LOG FINISH")
            if self.filehandler:
                self.filehandler.close()
            self.optimized_object.solver.free_up_log_file()