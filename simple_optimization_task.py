
import copy
from optimization_tools.abstract_solver import AbstractSolver
from optimization_tools.constraints_creators import Constraint
from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer, AbstractOptimizationTask
from optimization_tools.optimizers.null_optimizer import NullOptimizer


class OptimizationTaskWithInnerOptimizer(AbstractOptimizationTask):
    """
    
    """
    def __init__(self, initial_state, unique_id, opt_conditions, solver: AbstractSolver,
                 inner_optimizer: AbstractOPtimizer=None) -> None:
        super().__init__(initial_state, unique_id, opt_conditions, solver)
        self.cons = self.set_constraints()
        
        if inner_optimizer is None:
            self._inner_optimizer = NullOptimizer(copy.deepcopy(self))
            self._inner_optimizer.optimized_object._inner_optimizer = None
        else:
            self._inner_optimizer = inner_optimizer
        

    def set_constraints(self):
        cons = []
        for constraint in self.opt_conditions.constraints:
            cons.append({
                    'type': 'ineq',
                        'fun': Constraint(self,
                            constraint,
                            self.opt_conditions.constraints[constraint]
                            )
                })
        return cons
            

    @property
    def inner_optimizer(self):
        return self._inner_optimizer
    

    def get_opt_task_results_for_point(self, x):
        inner_model = copy.deepcopy(self.model)
        self.x_to_model(inner_model, x, self.conversion_map)
        self._inner_optimizer.optimized_object.model = inner_model
        self._inner_optimizer.optimized_object.unique_id = self.unique_id + "__" + str(x)
        inner_optimization_results = self._inner_optimizer.optimize()
        return inner_optimization_results


    def mass(self, x):
        results = self.get_opt_task_results_for_point(x)
        return results.mass



