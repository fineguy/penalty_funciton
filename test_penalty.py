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


def penalty_func(x):
    pass


def penalty_grad(x):
    pass


def output_result(res):
    print("""
    Finished optimization.
    Final approximation:    {}
    Final function value:   {:10f}
    Final gradient norm:    {:10f}
    -----------------------------------------------
""".format(np.array_str(res.x), res.fun, np.linalg.norm(res.jac)))


def test_simple():
    f = Function(
        lambda x: x**2,
        lambda x: x * 2
    )
    g = Function(
        lambda x: x - 1,
        lambda x: np.array([1])
    )
    P = Function(
        lambda x: x**2,
        lambda x: x * 2
    )
    x0 = np.array([10])

    res = penalty_optim(f, [g], P, solver, x0, alpha_gen(), disp=False)
    output_result(res)


def test_medium():
    f = Function(
        lambda x: x[0]**2 + x[0] * 6 + x[1]**2 + x[1] * 9,
        lambda x: 2 * x + np.array([6, 9])
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
    x0 = np.array([-1, 0.5])

    res = penalty_optim(f, [g1, g2], P, solver, x0, alpha_gen(), disp=True)
    output_result(res)


def test_hard():
    f = Function(
        lambda x: x[0]**2 + x[1]**2 + 2 * x[2]**2 + x[3]**2 - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3],
        lambda x: np.array([
            2 * x[0] - 5,
            2 * x[1] - 5,
            2 * x[2] - 21,
            2 * x[3] + 7
        ])
    )
    g1 = Function(
        lambda x: 8 - x[0]**2 - x[1]**2 - x[2]**2 - x[3]**2 - x[0] + x[1] - x[2] + x[3],
        lambda x: np.array([
            -2 * x[0] - 1,
            -2 * x[1] + 1,
            -2 * x[2] - 1,
            -2 * x[3] + 1
        ])
    )
    g2 = Function(
        lambda x: 10 - x[0]**2 - 2 * x[1]**2 - x[2]**2 - 2 * x[3]**2 + x[0] - x[3],
        lambda x: np.array([
            -2 * x[0] + 1,
            -4 * x[1],
            -2 * x[2],
            -4 * x[3] - 1
        ])
    )
    g3 = Function(
        lambda x: 5 - 2 * x[0]**2 - x[1]**2 - x[2]**2 - 2 * x[0] + x[1] + x[3],
        lambda x: np.array([
            -4 * x[0] - 2,
            -2 * x[1] + 1,
            -2 * x[2],
            1
        ])
    )
    P = Function(
        lambda x: x**2,
        lambda x: x * 2
    )
    x0 = np.zeros(4)

    res = penalty_optim(f, [g1, g2, g3], P, solver, x0, alpha_gen(), disp=True)
    output_result(res)
    # expect (0, 1, 2, -1) with function value -44


def test_novak():
    f = Function(
        lambda x: (x - 4).dot(x - 4),
        lambda x: 2 * (x - 4)
    )
    g = Function(
        lambda x: 5 - x.sum(),
        lambda x: np.array([-1, -1])
    )
    P = Function(
        lambda x: x**2,
        lambda x: 2 * x
    )
    x0 = np.zeros(2)

    res = penalty_optim(f, [g], P, solver, x0, alpha_gen(), disp=True)
    output_result(res)
    # expect (2.5, 2.5) with function value 4.5


def test_what():
    f = Function(
        lambda x: x**9,
        lambda x: 9 * x**8
    )
    g = Function(
        lambda x: x - 100000,
        lambda x: 1
    )
    P = Function(
        lambda x: x**2,
        lambda x: 2 * x
    )
    x0 = 0

    res = penalty_optim(f, [g], P, solver, x0, alpha_gen(), disp=False)
    output_result(res)
    # expect to reach limit in function calls


def main():
    np.set_printoptions(precision=8, suppress=True)
    # test_simple()
    test_medium()
    # test_hard()
    # test_novak()
    # test_what()


if __name__ == '__main__':
    main()
