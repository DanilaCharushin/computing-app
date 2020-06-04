import numpy as np
from typing import List


__all__ = ['least_square']


def least_square(x: List[float], y: List[float], n=1) -> callable:
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    A = []
    b = []
    for i in range(n + 1):
        A.append([])
        b.append(sum(y[k] * x[k] ** i for k in range(len(x))))
        for j in range(n + 1):
            if i == j == 0:
                A[i].append(len(x))
            else:
                A[i].append(sum(x[k] ** (i + j) for k in range(len(x))))
    c = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    # def f(x):
    #     return sum(c[i] * x ** i for i in range(len(c)))
    # return f
    return lambda x: sum(c[i] * x ** i for i in range(len(c)))
