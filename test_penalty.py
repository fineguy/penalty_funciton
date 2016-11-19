# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 04:34:19 2016

@author: timasemenov
"""

import numpy as np
from scipy.optimize import minimize

from penalty import penalty_optim
from utils import Function, print_test


def alpha_gen(alpha0=1, beta=0.1):
    cur_val = alpha0
    while True:
        yield cur_val
        cur_val *= beta


def solver(func, grad, x, disp):
    return minimize(func, x, method='CG', jac=grad, options={'disp': True})


@print_test
def test_very_simple():
    # expect (1)
    f = Function(
        lambda x: x[0],
        lambda x: 1
    )
    g = Function(
        lambda x: 1 - x[0],
        lambda x: np.array([-1])
    )
    x0 = 10
    return penalty_optim(f, [g], [], solver, x0, alpha_gen(), disp=True)


@print_test
def test_simple():
    # expect (1)
    f = Function(
        lambda x: x[0]**2,
        lambda x: x[0] * 2
    )
    g = Function(
        lambda x: 1 - x[0],
        lambda x: np.array([-1])
    )
    x0 = 10
    return penalty_optim(f, [g], [], solver, x0, alpha_gen(), disp=True)


@print_test
def test_medium():
    # expect (0, 0)
    f = Function(
        lambda x: x[0]**2 + x[0] * 6 + x[1]**2 + x[1] * 9,
        lambda x: 2 * x + np.array([6, 9])
    )
    g1 = Function(
        lambda x: -x[0],
        lambda x: np.array([-1, 0])
    )
    g2 = Function(
        lambda x: -x[1],
        lambda x: np.array([0, -1])
    )
    x0 = np.array([-1, 0.5])
    return penalty_optim(f, [g1, g2], [], solver, x0, alpha_gen(), disp=True)


@print_test
def test_novak():
    # expect (1.5, 1.5)
    f = Function(
        lambda x: (x - 4).dot(x - 4),
        lambda x: 2 * (x - 4)
    )
    g1 = Function(
        lambda x: x[0] - 5,
        lambda x: np.array([1, 0])
    )
    g2 = Function(
        lambda x: x.sum() - 3,
        lambda x: np.array([1, 1])
    )
    x0 = np.zeros(2)
    return penalty_optim(f, [g1], [g2], solver, x0, alpha_gen(), disp=True)


@print_test
def test_what():
    # expect to reach limit in function calls
    f = Function(
        lambda x: x[0]**9,
        lambda x: 9 * x[0]**8
    )
    g = Function(
        lambda x: 100000 - x[0],
        lambda x: np.array([-1])
    )
    x0 = np.array([0])
    return penalty_optim(f, [g], [], solver, x0, alpha_gen(), disp=True)


@print_test
def test_impossible():
    # expect to diverge
    f = Function(
        lambda x: np.exp(-x[0]),
        lambda x: -np.exp(-x[0])
    )
    g = Function(
        lambda x: x[0] * np.exp(-x[0]),
        lambda x: (1 - x[0]) * np.exp(-x[0])
    )
    x0 = np.array([1])
    return penalty_optim(f, [], [g], solver, x0, alpha_gen(), eps=1e-20, disp=True)


def main():
    np.set_printoptions(precision=8, suppress=True)
    test_very_simple()
    test_simple()
    test_medium()
    test_novak()
    test_what()
    test_impossible()


if __name__ == '__main__':
    main()
