"""
helper functions for optimisation algorithms
"""

import numpy as np


class Log(object):
    def __init__(self, step, coord, value):
        self.step = step
        self.coord = coord
        self.value = value

    def __repr__(self):
        if isinstance(self.value, np.ndarray):
            self.value = self.value.item()
        return f"[{self.step:6d}] x = {self.coord}, val = {self.value:.6f}"


class Result(object):
    def __init__(self):
        self.history = list()
        self.n_iter = None
        self.x = None
        self.error = None

    def update(self, step, coord, value):
        self.history.append(Log(step, coord.copy(), value))
        self.n_iter = len(self.history)
        self.x = coord

    def error(self, ground_truth):
        return np.linalg.norm(self.x, ground_truth)


def is_pos_def(mat):
    """
    Check if the (hessian) matrix is positive definite
    """
    evalue = np.linalg.eigvals(mat)
    return np.all(evalue > 0)


def finite_diff_gradient(func, x, diff):
    """
    Very naive implementation of finite difference gradient of `func`,
    given the coordinate `x` and the finite difference `diff`
    """
    n = len(x)
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(n):
        x_plus_dx = x.copy()
        x_plus_dx[i] += diff
        grad[i] = (func(x_plus_dx) - func(x)) / diff
    return grad


def finite_diff_hessian(func, x, diff):
    """
    Very naive implementation of finite difference hessian matrix of `func`,
    given the coordinate `x` and the finite difference `diff`
    """
    n = len(x)
    hess = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            x_plus_dx_ij = x.copy()
            x_plus_dx_i = x.copy()
            x_plus_dx_j = x.copy()
            x_plus_dx_ij[i] += diff
            x_plus_dx_ij[j] += diff
            x_plus_dx_i[i] += diff
            x_plus_dx_j[j] += diff
            hess[i][j] = (
                func(x_plus_dx_ij) - func(x_plus_dx_i) - func(x_plus_dx_j) + func(x)
            ) / diff**2
    return hess


def line_search(func, param_range, x, p, n=100):
    """
    Simple and naive implementation of line searching the minima of `func`,
    given search `range`, coordinate `x`, gradient `p` at `x` and searching
    granularity `n`
    """
    a = np.linspace(*param_range, n)
    index = 0
    minima = func(x)
    for i in range(len(a)):
        curr = func(x + a[i] * p.flatten())
        if curr < minima:
            index = i
    return a[index]


def init_simplex(x_init, scale=1.0):
    """
    Initialise simplex for Nelder-Mead as mentioned in
    https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    """
    n = len(x_init)
    simplex = np.zeros((n + 1, x_init.shape[0]), dtype=np.float64)
    simplex[0] = x_init
    for i in range(1, simplex.shape[0]):
        simplex[i] = x_init + scale * np.eye(n, dtype=np.float64)[i - 1]
    return [np.array(p, dtype=np.float64) for p in simplex.tolist()]
