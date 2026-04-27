# Значения по умолчанию
_settings = {
    'LOGGING_DIR': None,
    'CALCULATION_DIR': None,
    'LAT_GEN_PATH': None,
    'PANEL_SOLVER': None,
    'MAT_DB_PATH': None,
    'NASTRAN_SOLVER_PATH': None,
    'NUM_PROC': 1,
    'SINGLE_FEM_TASK_TIMEOUT': 300.0,
    'MAX_ITER': 100,
    'DEBUG': False,
    'RABBIT': False,
    'RPC_Q_IP': 'localhost',
    'RPC_Q_PORT': 5672,
    'FILEHANDLER': True,
    'STREAMHANDLER': False,
}


def configure(config_dict=None, **kwargs):
    """
    Настройка параметров через словарь или именованные аргументы.
    
    Примеры:
        # Через словарь
        configure({'LOGGING_DIR': '/path/to/logs', 'MAX_ITER': 200})
        
        # Через именованные аргументы
        configure(LOGGING_DIR='/path/to/logs', MAX_ITER=200)
        
        # Смешанный вариант
        configure({'LOGGING_DIR': '/path/to/logs'}, MAX_ITER=200)
    """
    if config_dict:
        _settings.update(config_dict)
    if kwargs:
        _settings.update(kwargs)


def get(key, default=None):
    """Получить значение настройки по ключу"""
    return _settings.get(key, default)


# Геттеры для часто используемых настроек
def get_logging_dir(custom_dir=None):
    """Получить директорию для логирования"""
    return custom_dir if custom_dir is not None else _settings['LOGGING_DIR']


def get_calculation_dir():
    return _settings['CALCULATION_DIR']


def get_lat_gen_path():
    return _settings['LAT_GEN_PATH']


def get_panel_solver():
    return _settings['PANEL_SOLVER']


def get_mat_db_path():
    return _settings['MAT_DB_PATH']


def get_nastran_solver_path():
    return _settings['NASTRAN_SOLVER_PATH']


def get_num_proc():
    return _settings['NUM_PROC']


def get_single_fem_task_timeout():
    return _settings['SINGLE_FEM_TASK_TIMEOUT']


def get_max_iter():
    return _settings['MAX_ITER']


def get_debug():
    return _settings['DEBUG']


def get_rabbit():
    return _settings['RABBIT']


def get_rpc_q_ip():
    return _settings['RPC_Q_IP']


def get_rpc_q_port():
    return _settings['RPC_Q_PORT']


def get_filehandler():
    return _settings['FILEHANDLER']


def get_streamhandler():
    return _settings['STREAMHANDLER']