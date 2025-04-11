from typing import Dict

from optimization_tools.abstract_object import AbstractObject



class OptConditions():
    """Класс для описания условий оптимизационной задачи"""
    # __slots__ = ["vars", "constraints", "opt_type"]
    def __init__(self, opt_vars: Dict, function_constraints: Dict) -> None:
        self.vars: dict = opt_vars # Это словарь типа {"h": {"min": 0.001, "max": 0.02}, }
        self.constraints: dict = function_constraints # Это словарь типа {"general_buckling": 1.0, }
        for opt_var in list(self.vars.keys()):
            if isinstance(self.vars[opt_var], dict) and self.vars[opt_var]["min"] == self.vars[opt_var]["max"]:
                self.vars.pop(opt_var)

class OptimizationTaskResults():
    # __slots__ = ['task_status', 'optimizer_status', 'var_values', 'constr_values']
    """класс для хранения результатов оптимизации"""
    def __init__(self, task_status_, optimizer_status_, var_values_, constr_values_, mass, model: AbstractObject):
        self.task_status: Dict = task_status_
        self.optimizer_status: int = optimizer_status_
        self.var_values: Dict = var_values_
        self.constr_values: Dict = constr_values_
        self.mass: float = mass
        self.model = model


# class AbstractTask():
#     model: AbstractObject
#     opt_conditions: OptConditions
#     optimizer: AbstractOPtimizer
#     task_results: OptimizationTaskResults
#     conversion_map: Dict

#     def __init__(self, model : AbstractObject, opt_conditions = None) -> None:
#         self.model = model
#         self.opt_conditions = opt_conditions
    




