"""
Градиентный оптимизатор на основе SLSQP
"""
import copy
import json
import logging
import os
import sys
import time

from scipy.optimize import minimize, Bounds

from exceptions import AllLoadsZeroException
from optimization_tools.abstract_solver import AbstractSolver
from optimization_tools.constraints_creators import ConstraintForNormalized
from optimization_tools.exceptions import SolverError
from optimization_tools.opt_conditions import OptimizationTaskResults
from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer, AbstractOptimizationTask
from .gradient_optimizer import OptimizationTaskWithNormalization

from ..opt_tools_settings import get_max_iter


class COBYLAOptimizer(AbstractOPtimizer):
    """
    Градиентный оптимизатор.
    """
    def __init__(self, optimized_object: OptimizationTaskWithNormalization, first_approx_function=None):
        super().__init__(optimized_object=optimized_object)
        self.first_approx_function = first_approx_function
        self.history: list[dict] = []

    def callback(self, x):
        vars_dict = self.optimized_object.get_vars_dict(x)
        vars_list = [vars_dict[var] for var in vars_dict]
        model = copy.deepcopy(self.optimized_object.model)
        self.optimized_object.x_to_model(model=model, x=vars_list, conversion_map=self.optimized_object.conversion_map)
        constraint_values = self.optimized_object.solver.solve(model, self.optimized_object.unique_id + "_callback", None)
        self.history.append({"vars" : vars_dict, "constraints": constraint_values})


    def optimize(self, **kwargs) -> OptimizationTaskResults:
        try:
            self.optimized_object.update_opt_vars()
            logger = self.logger
            logger.debug("Gradient optimization started")
            time_start = time.time()
            if self.first_approx_function:
                initial_results = self.optimized_object.solver.solve(self.optimized_object.model, self.optimized_object.unique_id, None)
                self.first_approx_function(
            results_map=initial_results, panel=self.optimized_object.model, opt_conditions=self.optimized_object.opt_conditions, logger=self.logger)

            x0 = self.optimized_object.get_x()
            x0_normalized = [x0[i]*self.optimized_object.normalization_coefficients[i]
                              for i in range(len(self.optimized_object.lower_bounds))]

            bounds = Bounds([self.optimized_object.lower_bounds[i] *
                              self.optimized_object.normalization_coefficients[i]
                                for i in range(len(self.optimized_object.lower_bounds))], 
                        [self.optimized_object.upper_bounds[i] *
                          self.optimized_object.normalization_coefficients[i] 
                          for i in range(len(self.optimized_object.lower_bounds))])
            initial_results = self.optimized_object.solver.solve(self.optimized_object.model, self.optimized_object.unique_id, None)
            cost_fun_coeff = initial_results["mass"]
            self.optimized_object.cost_function_normalization = cost_fun_coeff
            sim_start_time = time.time()
            logger.info(json.dumps(self.optimized_object.opt_conditions.vars))
            logger.info(f"SLSQP started at {sim_start_time}")
            settings = {
                'rhobeg': 0.2,
                'rhoend': 0.02,
                'catol': catol,
                'maxiter': 200,
                'disp': True
            }
            res = minimize(self.optimized_object.mass, x0_normalized, method='COBYLA',
                        jac='3-point', constraints= self.optimized_object.cons, bounds=bounds, 
                        options={'maxiter': get_max_iter(), 'disp': False}, callback=self.callback)
            
            logger.info(f"SLSQP finished at {sim_start_time} with status {res.status}")
            logger.info(f"SLSQP duration {time.time() - sim_start_time}")
            res_x_denormalized = [res.x[i] * self.optimized_object.denorm_coefficients[i]
                                   for i in range(len(self.optimized_object.lower_bounds))]
            self.optimized_object.x_to_model(self.optimized_object.model,
                                              res_x_denormalized, self.optimized_object.conversion_map)
            
            mass = self.optimized_object.mass(res.x) * self.optimized_object.cost_function_normalization
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
            logger.info(f"Optimization duration {time.time() - time_start}")
            logger.info("Ruslult variables:")
            logger.info(json.dumps(result_vars_map, indent=2))
            logger.info("Margin values:")
            logger.info(json.dumps(final_constraints, indent=2))
            logger.info(f"mass = {str(mass)}")
            logger.info(json.dumps(self.history))
            results : OptimizationTaskResults = OptimizationTaskResults(
                0, 0, result_vars_map, final_constraints, mass, self.optimized_object.model,
                opt_conditions=self.optimized_object.opt_conditions)

            return results
        except SolverError:
            logger.critical("%s Optimization failed due to unhandled exception during optimization" %  self.optimized_object.unique_id)
            logger.exception("%s exception: " % self.optimized_object.unique_id)
            return OptimizationTaskResults(
                1, 1, None, None, None, self.optimized_object.model)
        except AllLoadsZeroException:
            logger.warning(f"all loads zero for panel {self.optimized_object.unique_id}")
            return OptimizationTaskResults(
                1, 1, None, None, None, self.optimized_object.model)
        finally:
            logger.info("LOG FINISH")
            if self.filehandler:
                self.filehandler.close()
            self.optimized_object.solver.free_up_log_file()