
import sys
from helpers import parent_dir

PACKAGE_DIR = parent_dir(parent_dir(__file__))

sys.path.append(PACKAGE_DIR)
