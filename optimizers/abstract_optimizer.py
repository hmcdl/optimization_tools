from abc import abstractmethod
import logging
import os

from optimization_tools.abstract_solver import AbstractSolver, LoggableSolver
from optimization_tools.opt_conditions import OptConditions, OptimizationTaskResults
from optimization_tools import opt_tools_settings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constraints_creators import Constraint


class AbstractOptimizationTask:
    """
    Класс, осуществляющий взаимодействие между моделью оптимизируемого объекта
    и оптимизатором. Оптимизатор работает с абстрактными массивами и величинами, 
    поэтому необходимы методы, определяющие связь между переменными в массиве
    и конкретными параметрами модели.

    initial_state - модель в начальном состоянии

    unique_id - При создании экземпляра задается уникальный id (может быть
    переопределен в оптимизаторе, например в BruteForceOptimizer происходит
    перебор различных параметров и таска копируется с присвоением нового айдишника)

    opt_conditions - Условия оптимизации (класс OptConditions)

    solver - экземпляр класса решателя для проведения единичного расчета
    
    optimization_dir - директория для сохранения логов и результатов оптимизации.
                      Если None, используется значение из .env
    """
    
    def __init__(self, initial_state, unique_id, opt_conditions,
                  solver: AbstractSolver | LoggableSolver, 
                  optimization_dir: str = None) -> None:
        self._model = initial_state
        self.unique_id: str = unique_id
        self.opt_conditions: OptConditions = opt_conditions
        self.solver = solver
        self.conversion_map = self.get_conversion_map()
        self.local_log_path = self.unique_id
        
        # Сохраняем директорию оптимизации
        self.optimization_dir = optimization_dir
        self._inner_optimizer = None
    
    @property
    def logging_dir(self) -> str:
        """Получить актуальную директорию для логирования"""
        return opt_tools_settings.get_logging_dir(self.optimization_dir)
    
    @property
    def model(self):
        if self._model is None:
            raise ValueError
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def get_x(self):
        x = []
        for key in self.conversion_map:
                x.append(getattr(self._model, self.conversion_map[key]))
        return x
    
    def get_vars_dict(self, x: list):
        vars_dict = {}
        conversion_map = self.get_conversion_map()
        for i, _ in enumerate(x):
            vars_dict[conversion_map[i]] = x[i]

        return vars_dict

    def get_conversion_map(self):
        opt_vars_map = self.opt_conditions.vars
        var_values_with_bounds = {}
        for key in opt_vars_map:
            var_values_with_bounds[key] = {"cur_val": getattr(self._model, key), "bounds": opt_vars_map[key]}
        opt_var_counter = 0
        conversion_map = {}
        for key in list(var_values_with_bounds.keys()):
            conversion_map[opt_var_counter] = key
            opt_var_counter += 1
        return conversion_map
    
    @staticmethod
    def x_to_model(model, x, conversion_map):
        for i, _ in enumerate(x):
            setattr(model, conversion_map[i], x[i])


class AbstractOPtimizer():
    """
    Абстрактный класс оптимизатора.
    optimized_object - Экземляр класса модели оптимизируемого объекта
    logger - логгер, вызывается по AbstractOptimizationTask.unique_id
    filehandler - инициализируется, если в общих настройках стоит флаг
    """
    def __init__(self, optimized_object: AbstractOptimizationTask):
        self.optimized_object = optimized_object
        self.logger = None
        self.filehandler = None
        self.stream_handler_level = logging.CRITICAL

    def _set_up_logging_for_solver(self):
        self.optimized_object.solver.set_working_dir(
            self.optimized_object.local_log_path,
            self.optimized_object.logging_dir  # Передаём директорию
        )
        self.optimized_object.solver.initialize_log(
            self.optimized_object.local_log_path,
            self.optimized_object.logging_dir  # Передаём директорию
        )
        for constraint in self.optimized_object.cons:
            constraint_object: Constraint = constraint["fun"]
            constraint_object.logger = logging.getLogger(self.optimized_object.local_log_path + "solver_log")
 
        
    def run_optimization(self, **kwargs) -> OptimizationTaskResults:
        try:
            self.logger = logging.getLogger(self.optimized_object.unique_id)
            self.logger.setLevel(logging.DEBUG)
            handlers = self._create_handlers()
            for handler in handlers:
                self.logger.addHandler(handler)
            if kwargs.get("handlers", None):
                for handler in kwargs["handlers"]:
                    self.logger.addHandler(handler)
            
            if not self.optimized_object._inner_optimizer and \
                hasattr(self.optimized_object.solver, "set_working_dir"):
                self.optimized_object.solver.set_working_dir(
                    self.optimized_object.local_log_path,
                    self.optimized_object.logging_dir
                )
                self.optimized_object.solver.initialize_log(
                    self.optimized_object.local_log_path,
                    self.optimized_object.logging_dir
                )
                for constraint in self.optimized_object.cons:
                    constraint_object: Constraint = constraint["fun"]
                    constraint_object.logger = logging.getLogger(self.optimized_object.local_log_path + "solver_log")
            
            return self.optimize(**kwargs)
        finally:
            self.logger.info("LOG FINISH")
            if self.filehandler:
                self.filehandler.close()
            self.optimized_object.solver.free_up_log_file()

    @abstractmethod
    def optimize(self, **kwargs) -> OptimizationTaskResults:
        pass

        
    def _create_handlers(self):
        handlers = []
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if opt_tools_settings.FILEHANDLER:
            # Используем директорию из объекта
            global_log_dir = self.optimized_object.logging_dir
            this_log_dir = os.path.join(global_log_dir,
                                        self.optimized_object.local_log_path)
            
            os.makedirs(this_log_dir, exist_ok=True)
            this_log_file = os.path.join(this_log_dir, self.optimized_object.unique_id + ".log")
            filehandler = logging.FileHandler(filename=this_log_file, encoding='utf-8', mode="w")
            filehandler.setLevel(logging.DEBUG)
            filehandler.setFormatter(formatter)
            handlers.append(filehandler)
            self.filehandler = filehandler
        if opt_tools_settings.STREAMHANDLER:
            streamhandler = logging.StreamHandler()
            streamhandler.setLevel(self.stream_handler_level)
            streamhandler.setFormatter(formatter)
            handlers.append(streamhandler)
        return handlers