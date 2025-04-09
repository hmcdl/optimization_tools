from abc import abstractmethod
import logging
import os

from . import opt_tools_settings



from .abstract_object import AbstractObject, CachableObject

class AbstractSolver():
    def __init__(self) -> None:
        self.working_dir = None

    @abstractmethod
    def solve(self, calc_task: AbstractObject, unique_id: str, res_type: str):
        raise NotImplementedError
    
    @abstractmethod
    def configure(self, configure_dict):
        raise NotImplementedError

    def initialize_log(self, local_log_path): ...

    def free_up_log_file(self): ...


class WorkingDirSolver(AbstractSolver):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.working_dir = None
    
    def set_working_dir(self, local_working_dir_path):
        global_log_dir = opt_tools_settings.LOGGING_DIR
        self.working_dir = os.path.join(global_log_dir, local_working_dir_path)
        os.makedirs(self.working_dir, exist_ok=True)




class LoggableSolver(WorkingDirSolver):
    def __init__(self) -> None:
        super().__init__()
        self.filehandler = None
        # self.logger = logging.getLogger(__name__)

    def initialize_log(self, local_log_path):
        self.logger = logging.getLogger(local_log_path + "solver_log")
        self.logger.setLevel(logging.INFO)
        global_log_dir = opt_tools_settings.LOGGING_DIR
        this_log_dir = os.path.join(global_log_dir, local_log_path)
        # self.working_dir = this_log_dir
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

class CachableSolver(LoggableSolver):
    def __init__(self) -> None:
        super().__init__()
        self.cache_map = {}

    @abstractmethod
    def non_cached_calculation(self, calc_task: CachableObject, unique_id: str, res_type: str):
        raise NotImplementedError

    def solve(self, calc_task: CachableObject, unique_id: str, res_type: str):
        signature = calc_task.signature() 
        if signature in self.cache_map:
            # self.logger.info("cached result")
            return self.cache_map[signature][res_type]
        else:
            calculations_result = self.non_cached_calculation(calc_task, unique_id, res_type)
            self.cache_map[signature] = calculations_result
            if res_type is not None:
                return calculations_result[res_type]
            else:
                return calculations_result




class AbstractSolverFactory():
    @abstractmethod
    def get_solver_class(self) -> AbstractSolver:
        pass

    
    def configure_solver(self):
        pass

    def get_solver(self) -> type[AbstractSolver]:
        print(3, "solver choosing process")
        solver_creator = self.get_solver_class()
        return solver_creator



