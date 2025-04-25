import os
from os.path import join, dirname
from dotenv import load_dotenv
# from mat_db import mat_db

load_dotenv()

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

# materials_database = mat_db.MatDB(MAT_DB_PATH)

RPC_Q_IP = os.environ.get("RPC_Q_IP")

RPC_Q_PORT = int(os.environ.get("RPC_Q_PORT"))

LOGGING_DIR = os.environ.get("LOGGING_DIR")

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