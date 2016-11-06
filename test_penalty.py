# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 04:34:19 2016

@author: timasemenov
"""

import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

from penalty import Function, penalty_optim


def alpha_gen(alpha0=1, beta=0.1):
    cur_val = alpha0
    while True:
        yield cur_val
        cur_val *= beta


def solver(func, grad, x, eps, disp):
    return minimize(func, x, method='CG', jac=grad, tol=eps)


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

    x_sol = penalty_optim(f, [g], P, solver, x0, alpha_gen(), disp=True, max_iter=5)
    print(x_sol)


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

    x_sol = penalty_optim(f, [g1, g2], P, solver, x0, alpha_gen(), disp=True, max_iter=5)
    print(x_sol)


def main():
    test_simple()
    test_penalty()


if __name__ == '__main__':
    main()
