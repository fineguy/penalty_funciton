# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 02:26:18 2016

@author: timasemenov
"""

import numpy as np
import time


def describe_iter(ind, num, f, g):
    print("""
        Current iteration:      {:10d}
        Function calls:         {:10d}
        Function value:         {:10.6f}
        Gradient norm:          {:10.6f}
    """.format(ind, num, f, g))


def add_hist(hist, f, g, num, t):
    hist.setdefault('f', []).append(f)
    hist.setdefault('norm_g', []).append(g)
    hist.setdefault('n_evals', []).append(num)
    hist.setdefault('elaps_t', []).append(t)


def numpy_hist(hist):
    for key in hist.keys():
        hist[key] = np.array(hist[key])
    return hist


class Function(object):
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


def penalty_optim(f, g_list, P, solver, x0, alpha_gen,
                  eps=1e-4, max_iter=500, max_evals=1000,
                  disp=False, trace=False):
    """
        f:          function to be optimized
        g_list:     conditioning functions
        P:          penalty function
        solver:     solver function for optimizing combined f and P
        x0:         starting point
        alpha_gen:  generator for alphas
    """
    x, hist, start = x0, dict(), time.time()

    def f_func(x):
        return f.func(x) + _penalty_func(g_list, P, x) / alpha

    def f_grad(x):
        return f.grad(x) + _penalty_grad(g_list, P, x) / alpha

    for i, alpha in zip(range(max_iter), alpha_gen):
        x = solver(f_func, f_grad, x, eps)
        if alpha < eps:
            break


def _penalty_func(g_list, P, x):
    return sum(P.func(g.func(x)) for g in g_list)


def _penalty_grad(g_list, P, x):
    return sum(P.grad(g.func(x)) * g.grad(x) for g in g_list)
