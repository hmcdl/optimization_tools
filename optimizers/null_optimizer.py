"""
Нуль-оптимизатор, выполняет расчет оптимизируемого объекта и выдает 
результат, как обычный оптимизатор. Нужен для оптимизации перебором, 
когда вложенного оптимизатора нет.
"""
import logging
import logging.handlers
from optimization_tools.exceptions import SolverError
from optimization_tools.opt_conditions import OptimizationTaskResults
from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer


class NullOptimizer(AbstractOPtimizer):
    """
    Класс Нуль-оптимизатора
    """
    def optimize(self, **kwargs) -> OptimizationTaskResults:
        try:
            logger = self.logger
            logger.info("null optimizer operation")
            x0 = self.optimized_object.get_x()
            logger.info("LOG FINISH")
            mass = self.optimized_object.solver.solve(calc_task=self.optimized_object.model,
                                                    unique_id=self.optimized_object.unique_id, res_type="mass")
                
            result_vars_map = {}
            for var in self.optimized_object.opt_conditions.vars:
                result_vars_map[var] = getattr(self.optimized_object.model, var)
            
            
            final_constraints = {}
            for i, constraint_name in enumerate(self.optimized_object.opt_conditions.constraints):
                if self.optimized_object.opt_conditions.constraints[constraint_name] != 0:
                    constraint = (self.optimized_object.cons[i]['fun'](x0) + 1) *\
                        self.optimized_object.opt_conditions.constraints[constraint_name]
                else:
                    constraint = self.optimized_object.cons[i]['fun'](x0)
                final_constraints[constraint_name] = constraint
            
            results : OptimizationTaskResults = OptimizationTaskResults(
                0, 0, result_vars_map, final_constraints, mass, self.optimized_object.model)
            return results
        except SolverError:
            return OptimizationTaskResults(
                1, 1, None, None, None, self.optimized_object.model)
        finally:
            if self.filehandler:
                self.filehandler.close()