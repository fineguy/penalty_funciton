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


def alpha_gen(alpha0=1, beta=0.995):
    cur_val = alpha0
    while True:
        yield cur_val
        cur_val *= beta


def solver(func, grad, x, eps):
    res = minimize(func, x, method='CG', jac=grad, tol=eps)
    print(res.fun, res.jac, res.x)
    return res.x


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
        lambda x: 1 / x,
        lambda x: - 1 / x ** 2
    )
    x0 = np.array([1, 0.5])

    penalty_optim(f, [g1, g2], P, solver, x0, alpha_gen(beta=0.1), disp=True, max_iter=5)


def main():
    test_penalty()


if __name__ == '__main__':
    main()
