import os
from scripts.helpers import parent_dir

# def parent_dir(p):
#     return os.path.dirname(os.path.normpath(p))

MOD_FILE = os.path.abspath(__file__)
DATA_DIR = os.path.join(parent_dir(parent_dir(MOD_FILE)), 'data')