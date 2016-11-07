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


def numpy_dict(d):
    """Transform entries in history dictionary to numpy arrays"""
    for key, value in d.items():
        d[key] = np.array(value)
    return d
