"""
Many of them can be found from
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import numpy as np
from functools import partial

def default_function(x):
    """
    - dim(x) = 2
    - minima @ (-pi/2+n*2pi, pi+n*2pi)
    """
    return np.sin(x[0]) + np.cos(x[1])

def himmelblau(x):
    """
    - dim(x) = 2
    - minima @ (3, 2), (-2.805118, 3.131312), 
    (-3.779310, -3.283186), (3.584428, -1.848126)
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def rosenbrock(n, x):
    """
    - dim(x) = n
    - minima @ (1, 1, 1, ...), (-1, 1, 1, ...)
    """
    assert n >= 2 and n <= 7
    val = 0
    for i in range(n - 2):
        val += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return val

rosenbrock2d = partial(rosenbrock, 2)
rosenbrock3d = partial(rosenbrock, 3)
