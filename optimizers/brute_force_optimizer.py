"""
BruteForceOptimizer - Оптимизатор тупым перебором. Поддерживает
возможность запуска вложенных оптимизаторов для перебираемых точек
"""
import copy
import json
import logging
import os
import numpy
import time

from optimization_tools.optimization_executors import ForLoopExecutor
from optimization_tools.opt_conditions import OptimizationTaskResults
from optimization_tools.optimization_executors import AbstractExecutor
from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer
from optimization_tools.utils import iterate, constraints_are_satisfied
from optimization_tools.simple_optimization_task import OptimizationTaskWithInnerOptimizer
from optimization_tools.mapping_utils import ParameterMapper  # Новый импорт

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
    def __init__(self, optimized_object: OptimizationTaskWithInnerOptimizer, discreteness, config, seed_map: list = []):
        super().__init__(optimized_object, config=config)
        self.discreteness = discreteness
        self.seed_map: dict = seed_map
        self.executor = ForLoopExecutor(config)
        self.all_points = None
        self.param_mapper = None  # Будет инициализирован в optimize

    def set_executor(self, executor) -> None:
        self.executor: AbstractExecutor = executor

    def _init_mapper(self):
        """Инициализирует маппер для кодирования параметров"""
        mapping_file = os.path.join(
            self.optimized_object.logging_dir,
            self.optimized_object.local_log_path,
            "parameter_mapping.json"
        )
        self.param_mapper = ParameterMapper(mapping_file)

    def _get_params_dict_from_point(self, point: list) -> dict:
        """Создает словарь параметров из точки"""
        params_dict = {}
        for j, component_name in enumerate(self.optimized_object.conversion_map.values()):
            params_dict[component_name] = round(point[j], 6)  # Округляем для согласованности
        return params_dict

    def optimize(self, **kwargs) -> OptimizationTaskResults:
        logger = self.logger
        all_vars_ranges = []
        logger = logging.getLogger(self.optimized_object.unique_id)
        
        # Используем директорию из объекта
        global_log_dir = self.optimized_object.logging_dir
        this_log_dir = os.path.join(global_log_dir,
                                    self.optimized_object.local_log_path)
        
        os.makedirs(this_log_dir, exist_ok=True)
        this_log_file = os.path.join(this_log_dir, self.optimized_object.unique_id)
        # filehandler = logging.FileHandler(filename=this_log_file, encoding='utf-8', mode="w")
        # filehandler.setLevel(logging.DEBUG)
        # logger.addHandler(filehandler)
        # logger.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # filehandler.setFormatter(formatter)
        # logger.log(logging.DEBUG, "BruteForce optimization started")
        time_start = time.time()

        # Инициализируем маппер
        self._init_mapper()

        # Составляем списки значений переменных
        bounds_for_gradient = []
        for key, opt_var_name in self.optimized_object.conversion_map.items():
            var_constraints = self.optimized_object.opt_conditions.vars[self.optimized_object.conversion_map[key]]
            if isinstance(var_constraints, list):
                all_vars_ranges.append(var_constraints)
            else:
                min_val = var_constraints["min"]
                max_val = var_constraints["max"]
                if self.seed_map is None:
                    some_var_values = list(numpy.linspace(min_val, max_val, self.discreteness))
                else:
                    if opt_var_name in self.seed_map.keys():
                        some_var_values = list(numpy.linspace(min_val, max_val, self.seed_map[opt_var_name] + 1))
                if opt_var_name not in self.seed_map.keys():
                    bounds_for_gradient.append([(min_val, max_val)])
                    all_vars_ranges.append([getattr(self.optimized_object.model, opt_var_name)])
                else:
                    overlap_ratio = 0.5
                    some_var_bounds = []
                    if len(some_var_values) > 1:
                        n = len(some_var_values)
                        step = some_var_values[1] - some_var_values[0]
                        overlap = step * overlap_ratio
                        
                        for i in range(n - 1):
                            start = some_var_values[i] - overlap
                            end = some_var_values[i + 1] + overlap
                            
                            if i == 0 or start < some_var_values[i]:
                                start = some_var_values[i]
                            if i == n - 2 or end > some_var_values[-1]:
                                end = some_var_values[-1]
                            
                            some_var_bounds.append([start, end])
                    else:
                        some_var_bounds = [[some_var_values[0], some_var_values[0]]]
                    
                    all_vars_ranges.append(numpy.linspace(min_val, max_val, len(some_var_bounds)))
                    bounds_for_gradient.append(some_var_bounds)

        # создаем набор точек параметров all_points
        if self.all_points is not None:
            all_points = self.all_points
        else:
            all_points = []
            all_bounds = []
            iterate(0, all_vars_ranges, all_points, [])
            iterate(0, bounds_for_gradient, all_bounds, [])
        
        # Логируем количество точек
        logger.info(f"Total points to evaluate: {len(all_points)}")
        
        another_type_results: list[OptimizationTaskResults] = []
        inner_optimizers_copies: list[AbstractOPtimizer] = [copy.deepcopy(self.optimized_object.inner_optimizer) 
                                   for _ in range(len(all_points))]
        
        # Словарь для хранения соответствия кода и параметров (для логирования)
        code_to_point_info = {}
        
        for i, optimizer in enumerate(inner_optimizers_copies):
            if hasattr(optimizer, "executor"):
                optimizer.set_executor(self.optimized_object.inner_optimizer.executor)
            
            for j, opt_var in enumerate(optimizer.optimized_object.opt_conditions.vars.keys()):
                if opt_var in self.optimized_object.opt_conditions.vars:
                    new_bounds_for_this_var = all_bounds[i][j]
                    if opt_var in self.seed_map.keys():
                        all_points[i][j] = (all_bounds[i][j][0] + all_bounds[i][j][1]) / 2
                    optimizer.optimized_object.opt_conditions.vars[opt_var] = {
                        "min": new_bounds_for_this_var[0], 
                        "max": new_bounds_for_this_var[1]
                    }
            
            inner_model = copy.deepcopy(self.optimized_object.model)
            self.optimized_object.x_to_model(inner_model, all_points[i], self.optimized_object.conversion_map)
            optimizer.optimized_object.model = inner_model
            
            # Получаем словарь параметров для текущей точки
            params_dict = self._get_params_dict_from_point(all_points[i])
            
            # Получаем код для этих параметров
            point_code = self.param_mapper.get_or_create_code(params_dict)
            code_to_point_info[point_code] = params_dict
            
            # Используем код как имя папки
            optimizer.optimized_object.optimization_dir = self.optimized_object.optimization_dir
            optimizer.optimized_object.unique_id = f"{self.optimized_object.unique_id}__code_{point_code}"
            
            # Путь к папке с кодом
            optimizer.optimized_object.local_log_path = os.path.join(
                self.optimized_object.local_log_path,
                f"point_{point_code}"
            )
            
            # Логируем соответствие
            logger.debug(f"Point {point_code}: {params_dict}")
        
        # Логируем итоговое соответствие кодов и параметров
        logger.info("=" * 60)
        logger.info("Parameter mapping (code -> parameters):")
        for code, params in self.param_mapper.code_to_params.items():
            logger.info(f"  Code {code}: {params}")
        logger.info("=" * 60)
        
        # Запускаем вычисления
        another_type_results = self.executor(inner_optimizers_copies)

        # Анализируем результаты
        constraints_satisfied_points: list[tuple[int, OptimizationTaskResults]] = []
        
        for i, result in enumerate(another_type_results):
            if result is None or result.constr_values is None:
                continue
                
            if constraints_are_satisfied(result.constr_values, self.optimized_object.opt_conditions.constraints):
                # Получаем код для этой точки
                params_dict = self._get_params_dict_from_point(all_points[i])
                point_code = self.param_mapper.get_or_create_code(params_dict)
                constraints_satisfied_points.append((point_code, result))

        if len(constraints_satisfied_points) == 0:
            logger.warning("No points satisfying constraints found")
            return OptimizationTaskResults({"status_code": 0}, 1, None, None, None, None)
        
        # Находим точку с минимальной массой
        min_objective_point_code, min_objective_point = min(constraints_satisfied_points, key=lambda x: x[1].objective)
        
        # Логируем победителя
        logger.info("=" * 60)
        logger.info("BEST POINT FOUND:")
        logger.info(f"Code: {min_objective_point_code}")
        logger.info(f"Parameters: {self.param_mapper.get_params(min_objective_point_code)}")
        logger.info(f"objective: {min_objective_point.objective}")
        logger.info("=" * 60)
        
        # Сохраняем информацию о победителе в отдельный файл
        winner_info = {
            "winning_code": min_objective_point_code,
            "parameters": self.param_mapper.get_params(min_objective_point_code),
            "objective": min_objective_point.objective,
            "constraints": min_objective_point.constr_values,
            "variables": min_objective_point.var_values
        }
        
        winner_file = os.path.join(this_log_dir, "winner_info.json")
        with open(winner_file, 'w', encoding='utf-8') as f:
            json.dump(winner_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Winner info saved to: {winner_file}")
        
        # Выводим информацию о победителе
        logger.info("Brute force optimization finished")
        logger.info(f"Optimization duration {time.time() - time_start}")
        logger.info(f"Winning point code: {min_objective_point_code}")
        logger.info("Result variables:")
        logger.info(json.dumps(min_objective_point.var_values, indent=2))
        logger.info("Margin values:")
        logger.info(json.dumps(min_objective_point.constr_values, indent=2))
        logger.info(f"objective = {str(min_objective_point.objective)}")
        
        return min_objective_point