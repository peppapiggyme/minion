"""
Various famous optimisation methods implemented naively in python
"""

import numpy as np
from src.minion.utils import *


def newton(func, x_init, check=True, gamma=1., diff=1e-5, tol=1e-6, max_iter=100):
    """
    Newton's method in optimisation
    https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    """
    result = Result()
    x = x_init

    for k in range(max_iter):
        result.update(k, x, func(x))
        grad = finite_diff_gradient(func, x, diff)
        hess = finite_diff_hessian(func, x, diff)
        if check:
            assert is_pos_def(hess)

        t = -gamma * np.linalg.solve(hess, grad)
        if np.linalg.norm(t) < tol:
            break
        x += t

    result.update(k, x, func(x))
    return result, hess


def gradient_descent(func, x_init, gamma=1., diff=1e-5, tol=1e-6, max_iter=100):
    """
    Gradient Descent
    https://en.wikipedia.org/wiki/Gradient_descent
    """
    result = Result()
    x = x_init

    for k in range(max_iter):
        result.update(k, x, func(x))
        grad = finite_diff_gradient(func, x, diff)
        if np.linalg.norm(grad) < tol:
            break
        x -= gamma * grad

    result.update(k, x, func(x))
    return result


def quasi_newton(
    func, x_init, check=True, beta=1., range=(-1, 1), diff=1e-5, tol=1e-6, max_iter=100
):
    """
    Quasi-Newton method (BFGS) 
    https://en.wikipedia.org/wiki/Quasi-Newton_method
    """
    result = Result()
    x = x_init

    # scipy has auto determination of beta
    h0 = beta * np.eye(len(x), dtype=np.float64)  # hess^-1
    h = h0
    grad_next = None

    for k in range(max_iter):
        result.update(k, x, func(x))
        if grad_next is None:
            grad = finite_diff_gradient(func, x, diff)
        else:
            grad = grad_next
        if check:
            assert is_pos_def(np.linalg.inv(h))
        p = -(h @ grad).reshape(-1, 1)
        # decide step size according to exact line search
        if np.linalg.norm(p) < tol:
            break
        alpha = line_search(func, range, x, p, 100)
        # assert lower.x < upper.x
        s = alpha * p
        x += s.flatten()
        grad_next = finite_diff_gradient(func, x, diff)
        y = grad_next - grad
        y = y.reshape(-1, 1)
        # inverse hessian update
        sTy = (s.T @ y).item()  # scalar
        ssT = s @ s.T
        hy = h @ y
        h = h + (hy.T @ y + sTy) / sTy**2 * ssT - (hy @ s.T + s @ hy.T) / sTy

    result.update(k, x, func(x))
    return result, h


def nelder_mead(
    func, x_init, alpha=1., gamma=2., rho=0.5, sigma=0.5, scale=1., tol=1e-6, max_iter=100,
):
    """
    Nelder-Mead method
    https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    """
    result = Result()
    simplex = init_simplex(x_init, scale)
    
    for k in range(max_iter):
        result.update(k, simplex[0], func(simplex[0]))

        # termination determined by the standard deviation of the function values
        vals = np.array([func(p) for p in simplex])
        std = vals.std()
        if std < tol:
            break

        # ordering
        simplex.sort(key=lambda x: func(x))
        # centroid
        x0 = np.array([p for p in simplex[:-1]], dtype=np.float64)
        x0 = x0.mean(axis=0)
        # reflection
        xr = x0 + alpha * (x0 - simplex[-1])  # alpha > 0
        if func(xr) >= func(simplex[0]) and func(xr) < func(simplex[-2]):
            simplex[-1] = xr
            continue
        # expansion
        if func(xr) < func(simplex[0]):
            xe = x0 + gamma * (xr - x0)  # gamma > 1
            if func(xe) < func(xr):
                simplex[-1] = xe
            else:
                simplex[-1] = xr
            continue
        # contraction
        if func(xr) < func(simplex[-1]):
            xc = x0 + rho * (xr - x0)  # 0 < rho <= 0.5
            if func(xc) < func(xr):
                simplex[-1] = xc
                continue
        else:
            xc = x0 + rho * (simplex[-1] - x0)  # 0 < rho <= 0.5
            if func(xc) < func(simplex[-1]):
                simplex[-1] = xc
                continue
        # shrink
        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])

    return result
