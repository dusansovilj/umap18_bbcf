import os
# from helpers.aux import parent_dir
# from helpers.aux import parent_dir

def parent_dir(mypath):
    return os.path.dirname(os.path.normpath(mypath))

MOD_FILE = os.path.abspath(__file__)
DATA_DIR = os.path.join(parent_dir(parent_dir(MOD_FILE)), 'data')