'''
'''

import numpy as np
import os
import psutil



TDIST_005_VAL = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]


def comp_95confint(p):
    s = np.std(p)
#     import scipy.stats as st
#     print(st.t.interval(0.95, p.size-1, loc=np.mean(p), scale=st.sem(p))[0])
#     return np.mean(p) - st.t.interval(0.95, p.size-1, loc=np.mean(p), scale=st.sem(p))[0][0]
    mult = TDIST_005_VAL[p.size - 1]
    return mult * s / np.sqrt(p.size)


def ensure_directory(name, name_is_file=False):
    d = os.path.realpath(name)
    if name_is_file:
        d = os.path.split(name)[0]
    if not (os.path.isdir(d) or os.path.exists(d)):
        os.makedirs(d)
        

def parent_dir(mypath):
    return os.path.dirname(os.path.normpath(mypath))


def get_thread_count():
    return psutil.cpu_count()