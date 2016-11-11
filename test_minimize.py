# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 15:21:03 2016

@author: timasemenov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils import describe_iter

EPS = 1e-8
OPT = {
    'disp': True,
    'gtol': EPS,
    'return_all': True
}


def check_result(test_name, f, jac, x0, opt=OPT, disp=True):
    iter_idx = 0

    if disp:
        print('---------- {} ----------'.format(test_name))

        def callback(xk):
            nonlocal iter_idx
            describe_iter(iter_idx, xk, f(xk), jac(xk), -1, -1, -1)
            iter_idx += 1

        return minimize(f, x0, method='CG', jac=jac, options=opt, callback=callback)
    else:
        return minimize(f, x0, method='CG', jac=jac, options=opt)


def test1():
    def f(x):
        return 2 * x[0]**2 + 2 * x[1]**2 + 2 * x[0] * x[1] + 20 * x[0] + 10 * x[1] + 10

    def jac(x):
        return np.array([4 * x[0] + 2 * x[1] + 20, 4 * x[1] + 2 * x[0] + 10])

    x0 = np.zeros(2)

    check_result('test1', f, jac, x0)


def test2():
    def f(x):
        return x[0]**3 + x[1]**2 + 4 * x[0]**2 * x[1] + 3 * x[1] + 4

    def jac(x):
        return np.array([3 * x[0]**2 + 8 * x[0] * x[1], 2 * x[1] + 4 * x[0]**2 + 3])

    x0 = np.zeros(2)

    check_result('test2', f, jac, x0)


def test3():
    def f(x):
        return np.sum(1 / x)

    def jac(x):
        return -1 / np.square(x)

    x0 = np.ones(2)

    check_result('test3', f, jac, x0)


def test_eps_iter():
    def f(x):
        return np.sum(1 / x)

    def jac(x):
        return -1 / np.square(x)

    x0 = np.ones(2)
    eps_arr = np.logspace(-9, 0, num=10)
    iter_arr, eval_arr, fun_arr = [], [], []

    for eps in eps_arr:
        opt = {'gtol': eps}
        res = check_result('test_eps_iter', f, jac, x0, opt, False)
        iter_arr.append(res.nit)
        eval_arr.append(res.nfev)
        fun_arr.append(res.fun)

    plt.figure()
    plt.subplot(211)
    plt.plot(np.log(eps_arr), iter_arr, color='r', label='Количество итераций')
    plt.plot(np.log(eps_arr), eval_arr, color='b', label='Количество вызовов функции')
    plt.legend()
    plt.title('$\\frac{1}{x_1} + \\frac{1}{x_2} + \\frac{1}{x_3}$', fontsize=20, y=1.04)

    plt.subplot(212)
    plt.plot(np.log(eps_arr), fun_arr, color='r', label='Значение функции')
    plt.legend()
    plt.xlabel('$\log{\epsilon}$', fontsize='20')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test_eps_iter()
