# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 02:26:18 2016

@author: timasemenov
"""

import numpy as np
from collections import defaultdict
from time import time
from utils import safe_call, numpy_dict


def describe_iter(ind, x, f, g, t):
    """Pretty print information about iteration"""
    print("""
    Current iteration:      {:10d}
    Current estimation:     {}
    Function value:         {:10.6f}
    Gradient norm:          {:10.6f}
    Time elapsed:           {:10.2f}
""".format(ind, np.array_str(x), f, np.linalg.norm(g), time() - t))


def add_hist(hist, f, g, num, t):
    """Add information to history dictionary"""
    hist['func'].append(f)
    hist['grad'].append(g)
    hist['n_evals'].append(num)
    hist['time'].append(t)


def penalty_optim(f, g_list, P, solver, x, alpha_gen,
                  eps=1e-8, max_iter=500, max_evals=1000,
                  disp=False, trace=False):
    """Solve optimization problem with given constraints

    Parameters
    ----------
    f : Function object
        Function to be optimized

    g_list : list of Function objects
        Constraining functions

    P : Function object
        Penalty function

    solver : callable
        Solver function for unconstrained optimization

    x : ndarray, shape (n_dims,)
        Starting point for optimization

    alpha_gen : iterable
        Generator for decreasing sequence of numbers converging to 0

    Returns
    -------
    res : OptimizeResult
        Last approximation result

    hist : defaultdict, optional
        Dictionary that contains data for each iteration
    """
    n_evals, hist, start_time = 0, defaultdict(list), time()

    @safe_call
    def f_func(x):
        return f.func(x) + _penalty_func(g_list, P, x) / alpha

    @safe_call
    def f_grad(x):
        return f.grad(x) + _penalty_grad(g_list, P, x) / alpha

    for ind, alpha in zip(range(max_iter), alpha_gen):
        res = solver(f_func, f_grad, x, eps, disp)
        x = res.x
        n_evals += res.nfev
        add_hist(hist, res.fun, res.jac, n_evals, time() - start_time)

        if _penalty_func(g_list, P, x) < eps:
            break

        if disp:
            describe_iter(ind, res.x, res.fun, res.jac, start_time)

    if trace:
        return res, numpy_dict(hist)
    else:
        return res


def _penalty_func(g_list, P, x):
    return sum(P.func(g.func(x)) for g in g_list)


def _penalty_grad(g_list, P, x):
    return sum(P.grad(g.func(x)) * g.grad(x) for g in g_list)
