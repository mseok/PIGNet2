"""Insert to `sys.path` the absolute path of `..`.
"""
import inspect
import os
import sys


def get_here(follow_symlink: bool = True):
    # The script that called this function.
    caller = inspect.stack()[1][1]
    if follow_symlink:
        here = os.path.dirname(os.path.realpath(caller))
    else:
        here = os.path.dirname(os.path.abspath(caller))
    return here


def set_path():
    here = get_here()
    upper = os.path.dirname(here)
    sys.path.insert(1, upper)


set_path()
