# optimization_tools/config.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import os
import json

@dataclass
class OptimizationConfig:
    """
    Конфигурация для оптимизации.
    Передается явно во все компоненты.
    """
    # Директории
    logging_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'optimization_logs'))
    calculation_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'calculations'))
    
    # Параметры выполнения
    num_proc: int = 1
    # None -> enable when num_proc > 1; set False for cheap nested optimizers.
    parallel_fd: Optional[bool] = None
    # When False, FD stencil points use main-thread solve() only.
    # Wing coupled opt enables workers by default when num_proc > 1.
    parallel_fd_workers: bool = False
    # Extra parallel Nastran in SLSQP callback; usually redundant with jac prefill.
    prefetch_fd_in_callback: bool = False
    # If True, fall back to the last feasible point from optimization history.
    avoid_constraints_violations: bool = False
    single_fem_task_timeout: float = 300.0
    max_iter: int = 100
    seed: dict = None
    
    # Флаги
    debug: bool = False
    rabbit: bool = False
    structural_links_config: Optional[Dict] = None
    filehandler: bool = True
    streamhandler: bool = False
    
    # RabbitMQ
    rpc_q_ip: str = "localhost"
    rpc_q_port: int = 5672
    
    # Дополнительные пути (опциональные)
    lat_gen_path: Optional[str] = None
    panel_solver: Optional[str] = None
    mat_db_path: Optional[str] = None
    nastran_solver_path: Optional[str] = None
    
    # Расширение для пользовательских параметров
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Создаем директории при инициализации"""
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
    
    def get_logging_dir(self, custom_dir: Optional[str] = None) -> str:
        """Получить директорию для логирования"""
        result = custom_dir if custom_dir is not None else self.logging_dir
        os.makedirs(result, exist_ok=True)
        return result
    
    def to_dict(self) -> dict:
        """Сериализовать в словарь"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Создать из словаря"""
        return cls(**data)
    
    def save(self, filepath: str):
        """Сохранить в JSON файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """Загрузить из JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)