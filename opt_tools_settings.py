import ast
import os
from os.path import join, dirname
from dotenv import load_dotenv

# load_dotenv()

# Глобальная переменная для переопределения пути логирования (теперь она необязательна)
DEFAULT_LOGGING_DIR = os.environ.get("LOGGING_DIR")

def get_logging_dir(custom_dir: str = None) -> str:
    """
    Получить директорию для логирования.
    Если передан custom_dir, используется он.
    Иначе возвращается значение из .env или None
    """
    if custom_dir is not None:
        return custom_dir
    return DEFAULT_LOGGING_DIR

CALCULATION_DIR = os.environ.get("CALCULATION_DIR")
LAT_GEN_PATH = os.environ.get("LAT_GEN_PATH")
PANEL_SOLVER = os.environ.get("PANEL_SOLVER")
MAT_DB_PATH = os.environ.get("MAT_DB_PATH")
NASTRAN_SOLVER_PATH = os.environ.get("NASTRAN_SOLVER_PATH")
NUM_PROC = int(os.environ.get("NUM_PROC"))
SINGLE_FEM_TASK_TIMEOUT = float(os.environ.get("SINGLE_FEM_TASK_TIMEOUT"))
MAX_ITER = int(os.environ.get("MAX_ITER"))

DEBUG = int(os.environ.get("DEBUG_FLAG"))
if DEBUG == 1:
    DEBUG = True
else:
    DEBUG = False

RABBIT = int(os.environ.get("RABBIT"))
if RABBIT == 1:
    RABBIT = True
else:
    RABBIT = False

RPC_Q_IP = os.environ.get("RPC_Q_IP")
RPC_Q_PORT = int(os.environ.get("RPC_Q_PORT"))

# Для обратной совместимости
LOGGING_DIR = DEFAULT_LOGGING_DIR

FILEHANDLER = int(os.environ.get("FILEHANDLER"))
if FILEHANDLER == 1:
    FILEHANDLER = True
else:
    FILEHANDLER = False

STREAMHANDLER = int(os.environ.get("STREAMHANDLER"))
if STREAMHANDLER == 1:
    STREAMHANDLER = True
else:
    STREAMHANDLER = False