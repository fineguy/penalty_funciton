# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 04:34:19 2016

@author: timasemenov
"""

import numpy as np
from scipy.optimize import minimize

from penalty import penalty_optim
from utils import Function


def alpha_gen(alpha0=1, beta=0.1):
    cur_val = alpha0
    while True:
        yield cur_val
        cur_val *= beta


def solver(func, grad, x, eps, disp):
    return minimize(func, x, method='CG', jac=grad, tol=eps)


def output_result(res):
    print("""
    Finished optimization.
    Final approximation:    {}
    Final function value:   {:10.6f}
    Final gradient norm:    {:10.6f}
""".format(np.array_str(res.x), res.fun, np.linalg.norm(res.jac)))


def test_simple():
    f = Function(
        lambda x: x**2,
        lambda x: x * 2
    )
    g = Function(
        lambda x: x,
        lambda x: 1
    )
    P = Function(
        lambda x: x**2,
        lambda x: x * 2
    )
    x0 = 10

    res = penalty_optim(f, [g], P, solver, x0, alpha_gen(), disp=True, max_iter=5)
    output_result(res)


def test_penalty():
    f = Function(
        lambda x: x[0]**2 + x[0] * 6 + x[1]**2 + x[1] * 9,
        lambda x: np.array([x[0] * 2 + 6, x[1] * 2 + 9])
    )
    g1 = Function(
        lambda x: x[0],
        lambda x: np.array([1, 0])
    )
    g2 = Function(
        lambda x: x[1],
        lambda x: np.array([0, 1])
    )
    P = Function(
        lambda x: x**2,
        lambda x: x * 2
    )
    x0 = np.array([1, 0.5])

    res = penalty_optim(f, [g1, g2], P, solver, x0, alpha_gen(), disp=True)
    output_result(res)


def main():
    np.set_printoptions(precision=6, suppress=True)
    test_simple()
    test_penalty()


if __name__ == '__main__':
    main()
