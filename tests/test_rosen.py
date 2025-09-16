from concurrent.futures import ThreadPoolExecutor
from time import sleep, time
import numpy as np

from optimization_tools import opt_tools_settings
from optimization_tools.optimization_executors import MultiprocessExecutor, ThreadExecutor
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
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    # return sum(var**2 for var in x)

class SimpleVector(CachableObject):
    def __init__(self, x1, x2) -> None:
        super().__init__()
        self.x1 = x1
        self.x2 = x2

class RosenSolver(CachableSolver):
    def non_cached_calculation(self, calc_task: SimpleVector, unique_id: str):
        try:
            sleep(1)
            return {"ineq1": 1 - calc_task.x1 - 2*calc_task.x2,
                        "ineq2": 1 - calc_task.x1**2 - calc_task.x2,
                        "ineq3": 1 - calc_task.x1**2 + calc_task.x2,
                        "mass": rosen(np.array([calc_task.x1, calc_task.x2])) }
        except Exception as exc:
            raise SolverError from exc
        
def do_test():
    x_obj = SimpleVector(0.4, 0.1)
    all_opt_vars = {"x1": {"min": 0, "max": 1}, "x2": {"min": -0.5, "max": 2}}
    inner_opt_vars = {"x1": {"min": 0, "max": 1}}
    outer_opt_vars = {"x2": {"min": -0.5, "max": 2}}
    constraints = {"ineq1": 0, "ineq2": 0, "ineq3": 0,}
    all_opt_params = OptConditions(all_opt_vars, constraints)
    inner_opt_params = OptConditions(inner_opt_vars, constraints)
    outer_opt_params = OptConditions(outer_opt_vars, constraints)

    rosen_solver_obj = RosenSolver()

    inner_optimization_task = OptimizationTaskWithNormalization(initial_state=x_obj,
        unique_id=None, opt_conditions=all_opt_params, 
        solver=rosen_solver_obj)
    inner_optimizer = GradientOptimizer(inner_optimization_task)

    outer_optimization_task = OptimizationTaskWithInnerOptimizer(initial_state=x_obj,
         unique_id="1", opt_conditions=inner_opt_params, 
        inner_optimizer=inner_optimizer,
        solver=rosen_solver_obj)
    outer_optimizer = BruteForceOptimizer(outer_optimization_task, 4)

    super_outer_optimization_task = OptimizationTaskWithInnerOptimizer(initial_state=x_obj,
         unique_id="2", opt_conditions=outer_opt_params, 
        inner_optimizer=outer_optimizer,
        solver=rosen_solver_obj)
    super_outer_optimizer = BruteForceOptimizer(super_outer_optimization_task, 2)

    pool = ProcessingPool(6)
    executor = MultiprocessExecutor(pool)
    outer_optimizer.set_executor(executor=executor)

    outer_executor = ThreadExecutor(ThreadPoolExecutor(10))
    super_outer_optimizer.set_executor(executor=outer_executor)

    start_time = time()
    result: OptimizationTaskResults = super_outer_optimizer.run_optimization()
    print(time() - start_time)