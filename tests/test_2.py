import numpy as np

from optimization_tools.optimization_executors import MultiprocessExecutor
from optimization_tools.optimizers.brute_force_optimizer import BruteForceOptimizer
from optimization_tools.simple_optimization_task import OptimizationTaskWithInnerOptimizer

from ..abstract_object import AbstractObject, CachableObject
from ..abstract_solver import AbstractSolver, CachableSolver
from ..exceptions import SolverError
from ..opt_conditions import OptConditions, OptimizationTaskResults
from ..optimizers.gradient_optimizer import GradientOptimizer, OptimizationTaskWithNormalization
from pathos.multiprocessing import ProcessingPool


def rosen(x):
    """The Rosenbrock function"""
    # return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return sum(var**2 for var in x)

class SimpleVector(CachableObject):
    def __init__(self, x1, x2) -> None:
        super().__init__()
        self.x1 = x1
        self.x2 = x2

class RosenSolver(CachableSolver):
    def non_cached_calculation(self, calc_task: SimpleVector, unique_id: str):
        try:

            return {"ineq1": 1 - calc_task.x1 - 2*calc_task.x2,
                        "ineq2": 1 - calc_task.x1**2 - calc_task.x2,
                        "ineq3": 1 - calc_task.x1**2 + calc_task.x2,
                        "mass": rosen(np.array([calc_task.x1, calc_task.x2])) }
        except Exception as exc:
            raise SolverError from exc
        
def do_test():
    x_obj = SimpleVector(0.5, -0.5)
    all_opt_vars = {"x1": {"min": 0.25, "max": 0.75}}

    constraints = {"ineq1": 0, "ineq2": 0, "ineq3": 0,}
    all_opt_params = OptConditions(all_opt_vars, constraints)


    rosen_solver_obj = RosenSolver()

    inner_optimization_task = OptimizationTaskWithNormalization(initial_state=x_obj,
        unique_id="1", opt_conditions=all_opt_params, 
        solver=rosen_solver_obj)
    inner_optimizer = GradientOptimizer(inner_optimization_task)


    # pool = ProcessingPool(opt_tools_settings.NUM_PROC)
    # executor = MultiprocessExecutor(pool)
    # outer_optimizer.set_executor(executor=executor)

    result: OptimizationTaskResults = inner_optimizer.run_optimization()