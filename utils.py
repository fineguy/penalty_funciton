# -*- coding: utf-8 -*-
"""
Created on Mon Nov 7 16:49:21 2016

@author: timasemenov
"""

import sys
import numpy as np


class Function(object):
    """Combines function, its gradient and its hessian into one object"""

    def __init__(self, func, grad=None, hess=None):
        self.func = func
        self.grad = grad
        self.hess = hess

    def func(self, x):
        return self.func(x)

    def grad(self, x):
        return self.grad(x)

    def hess(self, x):
        return self.hess(x)


def safe_call(func):
    """Call func if func is implemented. Quit program otherwise"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            print("Oops! Something went wrong in {}".format(func.__name__))
            sys.exit(0)
    return wrapper


def print_test(test_func):
    """Print test name before execution"""
    def wrapper(*args, **kwargs):
        print('---------- {} ----------'.format(test_func.__name__))
        test_func(*args, **kwargs)
    return wrapper


def numpy_dict(d):
    """Transform entries in history dictionary to numpy arrays"""
    for key, value in d.items():
        d[key] = np.array(value)
    return d


def describe_iter(ind, x, f, g, p, f_n, g_n, a):
    """Pretty print information about iteration"""
    print("""
    Current iteration:      {:10d}
    Current estimation:     {}
    Function value:         {:10f}
    Gradient norm:          {:10f}
    Penalty value:          {:10f}
    Total function calls:   {:10d}
    Total gradient calls:   {:10d}
    Current alpha:          {:10f}
""".format(ind + 1, np.array_str(x), f, np.linalg.norm(g), p, g_n, g_n, a))
