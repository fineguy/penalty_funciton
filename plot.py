# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 05:45:37 2016

@author: timasemenov
"""

import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return x


def f2(x):
    return np.square(x)


def g(x):
    return np.square(np.maximum(x, 0))


def plot(x, f, f_str, x_min, y_lim, path):
    y1 = f(x)
    y2 = f(x) + g(1-x)
    y3 = f(x) + g(1-x) * 5
    y4 = f(x) + g(1-x) * 25

    fig, ax = plt.subplots()
    ax.plot(x, y1, color='b', label='${}$'.format(f_str))
    ax.plot(x, y2, color='r', label='${}+(1-x)^2_+$'.format(f_str))
    ax.plot(x, y3, color='g', label='${}+5(1-x)^2_+$'.format(f_str))
    ax.plot(x, y4, color='c', label='${}+25(1-x)^2_+$'.format(f_str))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.annotate('$x^*$', xy=(x_min[0], x_min[0]), xycoords='data',
                 xytext=(10, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 fontsize=20)

    plt.plot([x_min[0], x_min[0]], [0, f(x_min[0])], color='b', linestyle='--')
    plt.plot([x_min[1], x_min[1]], [0, f(x_min[1]) + g(1 - x_min[1])],
             color='r', linestyle='--')
    plt.plot([x_min[2], x_min[2]], [0, f(x_min[2]) + g(1 - x_min[2]) * 5],
             color='g', linestyle='--')
    plt.plot([x_min[3], x_min[3]], [0, f(x_min[3]) + g(1 - x_min[3]) * 25],
             color='c', linestyle='--')

    plt.ylim(*y_lim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=20)
    fig.set_size_inches(10,7.5)
    fig.savefig(path, dpi=400)
    plt.show()


plot(np.linspace(-0.5, 1.5, 100), f1, 'x', [1, 0.5, 0.9, 0.98], [0, 5],
      'first.png')
plot(np.linspace(-0.5, 1.5, 100), f2, 'x^2', [1, 0.5, 5 / 6, 25 / 26], [0, 5],
     'second.png')
