# optimization_tools/abstract_solver.py
from abc import abstractmethod
import copy
import logging
import os
from .config import OptimizationConfig

class AbstractSolver:
    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.working_dir = None

    @abstractmethod
    def solve(self, calc_task, unique_id: str, res_type: str | None):
        raise NotImplementedError
    
    @abstractmethod
    def configure(self, configure_dict):
        raise NotImplementedError

    def initialize_log(self, local_log_path): ...
    def free_up_log_file(self): ...


class WorkingDirSolver(AbstractSolver):
    def __init__(self, config: OptimizationConfig) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.working_dir = None
    
    def set_working_dir(self, local_working_dir_path, base_log_dir=None):
        # Используем переданную директорию или из конфига
        global_log_dir = base_log_dir if base_log_dir is not None else self.config.logging_dir
        self.working_dir = os.path.join(global_log_dir, local_working_dir_path)
        os.makedirs(self.working_dir, exist_ok=True)


class LoggableSolver(WorkingDirSolver):
    def __init__(self, config: OptimizationConfig) -> None:
        super().__init__(config)
        self.filehandler = None

    def initialize_log(self, local_log_path, base_log_dir=None):
        self.logger = logging.getLogger(local_log_path + "solver_log")
        self.logger.setLevel(logging.INFO)
        
        global_log_dir = base_log_dir if base_log_dir is not None else self.config.logging_dir
        this_log_dir = os.path.join(global_log_dir, local_log_path)
        os.makedirs(this_log_dir, exist_ok=True)
        this_log_file = os.path.join(this_log_dir, "solver_log.log")
        
        filehandler = logging.FileHandler(filename=this_log_file, encoding='utf-8', mode="w")
        filehandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        filehandler.setFormatter(formatter)
        self.logger.addHandler(filehandler)
        self.filehandler = filehandler

    def free_up_log_file(self):
        if self.filehandler:
            self.filehandler.close()
            self.filehandler = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["filehandler"] = None
        return state


class CachableSolver(LoggableSolver):
    def __init__(self, config: OptimizationConfig) -> None:
        super().__init__(config)
        self.cache_map = {}

    @abstractmethod
    def non_cached_calculation(self, calc_task, unique_id: str):
        raise NotImplementedError

    def clone_for_parallel_eval(self, worker_tag: str = "") -> "CachableSolver":
        clone = copy.deepcopy(self)
        clone.cache_map = {}
        clone.filehandler = None
        clone.on_parallel_clone(worker_tag)
        return clone

    def on_parallel_clone(self, worker_tag: str) -> None:
        """Hook for solver-specific isolation after deepcopy (e.g. unique workdirs)."""

    def solve(self, calc_task, unique_id: str, res_type: str | None) -> dict:
        signature = calc_task.signature() 
        if signature in self.cache_map:
            result_map = self.cache_map[signature]
        else:
            calculations_result = self.non_cached_calculation(calc_task, unique_id)
            self.cache_map[signature] = calculations_result
            result_map = calculations_result
        
        if res_type is not None:
            return result_map[res_type]
        else:
            return result_map