# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 02:26:18 2016

@author: timasemenov
"""

from collections import defaultdict
from utils import safe_call, numpy_dict, describe_iter


def add_hist(hist, f, g, f_n, g_n):
    """Add information to history dictionary"""
    hist['func'].append(f)
    hist['grad'].append(g)
    hist['f_evals'].append(f_n)
    hist['g_evals'].append(g_n)


def penalty_optim(f, g_list, P, solver, x, alpha_gen, eps=1e-8,
                  max_iter=500, max_func_evals=1000, max_grad_evals=1000,
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

    eps : float, default=1e-8
        Stopping criterion for maximum penalty_optim

    max_iter : int, default=500
        Stopping criterion for maximum number of iterations

    max_func_evals : int, default=1000
        Stopping criterion for maximum number of function calls

    max_grad_evals : int, default=1000
        Stopping criterion for maximum number of gradient calls

    disp : boolean, default=False
        Show optimization progress

    trace : boolean, default=False
        Return optimization history

    Returns
    -------
    res : OptimizeResult
        Last approximation result

    hist : defaultdict, optional
        Dictionary that contains data for each iteration
    """
    func_evals, grad_evals = 0, 0
    hist = defaultdict(list)

    @safe_call
    def f_func(x):
        return f.func(x) + _penalty_func(g_list, P, x) / alpha

    @safe_call
    def f_grad(x):
        return f.grad(x) + _penalty_grad(g_list, P, x) / alpha

    for ind, alpha in zip(range(max_iter), alpha_gen):
        res = solver(f_func, f_grad, x, eps, disp)
        x = res.x
        func_evals += res.nfev
        grad_evals += res.njev
        penalty = _penalty_func(g_list, P, x)

        if trace:
            add_hist(hist, res.fun, res.jac, func_evals, grad_evals)

        if penalty < eps:
            print("Iteration {}. Penalty function value is less than epsilon".format(ind + 1))
            break
        if func_evals >= max_func_evals:
            print("Iteration {}. Exceeded the expected number of function evaluations".format(ind + 1))
            break
        if grad_evals >= max_grad_evals:
            print("Iteration {}. Exceeded the expected number of gradient evaluations".format(ind + 1))
            break

        if disp:
            describe_iter(ind, res.x, res.fun, res.jac, penalty)

    if trace:
        return res, numpy_dict(hist)
    else:
        return res


def _penalty_func(g_list, P, x):
    return sum(P.func(g.func(x)) for g in g_list)


def _penalty_grad(g_list, P, x):
    return sum(P.grad(g.func(x)) * g.grad(x) for g in g_list)
