import copy
from optimization_tools.optimizers.abstract_optimizer import AbstractOptimizationTask



class Constraint:
    def __init__(self, opt_task_object: AbstractOptimizationTask, parameter, limit) -> None:
        self.opt_task_object = opt_task_object
        self.parameter = parameter
        self.limit = limit
        self.logname = None
        self.logger = None
            
    
    def __call__(self, x):
        logger = self.logger
        cur_values_map = {}
        for i,val in enumerate(x):
            cur_values_map[self.opt_task_object.conversion_map[i]] = val
        logger.info(cur_values_map)
        copy_model = copy.deepcopy(self.opt_task_object.model)
        self.opt_task_object.x_to_model(copy_model,
                                         x=x, conversion_map=self.opt_task_object.conversion_map)
        result =  self.opt_task_object.solver.solve(calc_task=copy_model,
                     unique_id=self.opt_task_object.unique_id, res_type=self.parameter)
        logger.info(f"{self.parameter}: {result}")
        if self.limit != 0:
            return result / self.limit - 1
        else:
            return result


class ConstraintForNormalized(Constraint):
    def __init__(self, opt_task_object: AbstractOptimizationTask,
                  parameter, limit, denorm_coefficients) -> None:
        super().__init__(opt_task_object, parameter, limit)
        self.denorm_coefficients = denorm_coefficients

    def __call__(self, x):
        x_denorm = [x[i] * self.denorm_coefficients[i] for i in range(len(self.denorm_coefficients))]
        return super().__call__(x_denorm)