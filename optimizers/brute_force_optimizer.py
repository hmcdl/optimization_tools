"""
BruteForceOptimizer - Оптимизатор тупым перебором. Поддерживает
возможность запуска вложенных оптимизаторов для перебираемых точек
"""
import copy
import logging
import os
import numpy

from optimization_tools.optimization_executors import ForLoopExecutor
from optimization_tools.opt_conditions import OptimizationTaskResults
from optimization_tools.optimization_executors import AbstractExecutor
from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer
from optimization_tools.utils import iterate
from optimization_tools.simple_optimization_task import OptimizationTaskWithInnerOptimizer

from .. import opt_tools_settings



class BruteForceOptimizer(AbstractOPtimizer):
    """
    Класс оптимизатора методом перебора.
    discreteness - дискретность - Имеет смысл для переменных оптимизации, заданных
    непрерывным отрезком
    executor - механизм запуска расчета отдельных точек. По умолчанию - просто цикл, 
    можно запускать с распараллеливанием на одной машине или на кластере.

    Методы:
    set_executor(self, executor)

    optimize(self, **kwargs) - оптимизация


    """
    def __init__(self, optimized_object: OptimizationTaskWithInnerOptimizer, discreteness):
        super().__init__(optimized_object)
        self.discreteness = discreteness
        self.executor = ForLoopExecutor()
        

    def set_executor(self, executor) -> None:
        self.executor: AbstractExecutor = executor


    def optimize(self, **kwargs) -> OptimizationTaskResults:
        logger = self.logger
        all_vars_ranges = []
        logger = logging.getLogger(self.optimized_object.unique_id)
        global_log_dir = opt_tools_settings.LOGGING_DIR
        this_log_dir = os.path.join(global_log_dir,
                                      self.optimized_object.local_log_path)
        
        os.makedirs(this_log_dir, exist_ok=True)
        this_log_file = os.path.join(this_log_dir, self.optimized_object.unique_id)
        filehandler = logging.FileHandler(filename=this_log_file, encoding='utf-8', mode="w")
        filehandler.setLevel(logging.DEBUG)
        logger.addHandler(filehandler)
        logger.setLevel(logging.DEBUG)
        logger.log(logging.DEBUG, "Hello world")

        # Составляем списки значений переменных
        # Если область значений параметра изначально дискретная, 
        # то кладем его, как есть.
        # Если область непрерывная, то разбиваем на равные промежутки
        # в соответствии с настройкой дискретности оптимизатора
        # В результате получаем список списков
        # [[0.8, 0.9, 1.0, 1.1, 1.2], [0.008, 0.009, 0.01, 0.011, 0.012]]
        # Порядок следования списков соответствует порядку в conversion_map
        for key, _ in self.optimized_object.conversion_map.items():
            var_constraints = self.optimized_object.opt_conditions.vars[self.optimized_object.conversion_map[key]]
            if isinstance(var_constraints, list):
                all_vars_ranges.append(var_constraints)
            else:
                min_val = var_constraints["min"]
                max_val = var_constraints["max"]
                all_vars_ranges.append(numpy.linspace(min_val, max_val, self.discreteness))
        
        # создаем набор точек параметров all_points.
        # [[0.8, 0.008], [0.8, 0.009], ..., [0.09, 0.008], [0.08, 0.009], ... [1.2, 0.012]]
        all_points = []
        # Уебищная функция, заполняет all_points
        iterate(0, all_vars_ranges, all_points, [])
        
        another_type_results: list[OptimizationTaskResults] = []

        inner_optimizers_copies = [ copy.deepcopy(self.optimized_object.inner_optimizer) 
                                   for _ in range(len(all_points))]
        for i, optimizer in enumerate(inner_optimizers_copies):
            inner_model = copy.deepcopy(self.optimized_object.model)
            self.optimized_object.x_to_model(inner_model, all_points[i], self.optimized_object.conversion_map)
            optimizer.optimized_object.model = inner_model
            point_dict_str = ""
            for j, component_name in enumerate(self.optimized_object.conversion_map.values()):
                point_dict_str += component_name + "=" + str(all_points[i][j])
                # .append(f"{key}={all_points[i][j]}")
            optimizer.optimized_object.unique_id = self.optimized_object.unique_id + "__" + point_dict_str
            optimizer.optimized_object.local_log_path = self.optimized_object.local_log_path +\
                  "/" + str(all_points[i])
        
        another_type_results = self.executor(inner_optimizers_copies)

        constraints_satisfied_points: list[OptimizationTaskResults] = []
        for result in another_type_results:
            all_satisfied = True
            point_constraints = result.constr_values
            if point_constraints is None: #Если все оптимизации для данной точки провалились, то result.constr_values будет содержать None
                continue
            for key, val in point_constraints.items():
                # Сравниваем фактического получившегося значение огрнаничения с
                # заданным, чтобы превышение было не более, чем на 1% от заданного ограничения 
                etalon = - abs(0.01 * self.optimized_object.opt_conditions.constraints[key])
                difference = val - self.optimized_object.opt_conditions.constraints[key]
                if val - self.optimized_object.opt_conditions.constraints[key] < - abs(0.01 * self.optimized_object.opt_conditions.constraints[key]):
                    all_satisfied = False
            if all_satisfied:
                constraints_satisfied_points.append(result)

        if len(constraints_satisfied_points) == 0:
            return OptimizationTaskResults({"status_code": 0}, 1, None, None, None, None)
        min_mass_point: OptimizationTaskResults = sorted(constraints_satisfied_points, key=lambda x: x.mass)[0]

        return min_mass_point
