from concurrent.futures import ThreadPoolExecutor
import json
from time import sleep, time
import numpy as np

import aircraft_panel
from aircraft_panel.conver_panel import ClassicPanel
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

class PanelAnalyticalHashableSolver(CachableSolver):
    def __init__(self) -> None:
        super().__init__()
        # super(LoggableSolver, self).__init__()
        # self.filehandler = None
        # self.logger: logging.Logger = None
        # self.cache_map = {}

    def non_cached_calculation(self, calc_task: ClassicPanel, unique_id: str):
        mass = calc_task.calculate_mass()
        general_buckling = aircraft_panel.conver_panel.general_buckling(calc_task, )
        skin_local_buckling = aircraft_panel.conver_panel.skin_local_buckling(calc_task, logger=self.logger)
        skin_strength = aircraft_panel.conver_panel.skin_strength(calc_task)
        stringer_strength = aircraft_panel.conver_panel.stringer_strength(calc_task)
        result_map = {
            "mass" : mass,
            "general_buckling": general_buckling,
            "skin_local_buckling": skin_local_buckling,
            "skin_strength": skin_strength,
            "stringer_strength": stringer_strength,
        }
        return result_map
    
def do_test():
    with open("initializer.json", 'r') as f:
        panel_data = json.load(f)
    panel = ClassicPanel(initializer_json=panel_data["object"])
    gradient_opt_params = OptConditions(panel_data["opt_params"]["opt_vars"], panel_data["opt_params"]["function_constraints"])
    
    discrete_opt_params = OptConditions({
            "str_pitch": {
                "min": 0.03,
                "max": 0.12
            }}, function_constraints=panel_data["opt_params"]["function_constraints"])
    
    # panel_solver = PanelAnalyticalSolver()
    optimization_task = OptimizationTaskWithNormalization(initial_state=panel, unique_id=panel.panel_code, 
                                                 opt_conditions=gradient_opt_params,
                                                   solver=PanelAnalyticalHashableSolver())
    gradient_optimizer = GradientOptimizer(optimized_object=optimization_task)

    seed = OptimizationTaskWithInnerOptimizer(initial_state=panel,
         unique_id="1", opt_conditions=gradient_opt_params, 
        inner_optimizer=gradient_optimizer,
        solver=PanelAnalyticalHashableSolver())
    seed_optimizer = BruteForceOptimizer(seed, 2)

    discrete_optimization_task = OptimizationTaskWithInnerOptimizer(initial_state=panel, unique_id=panel.panel_code, 
                                                 opt_conditions=discrete_opt_params,
                                                   solver=PanelAnalyticalHashableSolver(),
                                                   inner_optimizer=seed_optimizer)
    
    discrete_optimizer = BruteForceOptimizer(optimized_object=discrete_optimization_task, discreteness=8)

    pool = ProcessingPool(opt_tools_settings.NUM_PROC)
    executor = MultiprocessExecutor(pool)
    seed_optimizer.set_executor(executor=executor)

    outer_executor = ThreadExecutor(ThreadPoolExecutor(10))
    discrete_optimizer.set_executor(executor=outer_executor)

    result = discrete_optimizer.run_optimization()