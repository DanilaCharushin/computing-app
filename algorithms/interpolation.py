import numpy as np
import scipy.interpolate as interp
from typing import List

__all__ = ['linear', 'quadratic', 'spline', 'lagrange']


def linear(x: List[float], y: List[float]) -> callable:
    return interp.interp1d(x, y, kind="nearest")


def quadratic(x: List[float], y: List[float]) -> callable:
    return interp.interp1d(x, y, kind="quadratic")


def spline(x: List[float], y: List[float]) -> callable:
    # return interp.CubicSpline(x, y)
    return interp.interp1d(x, y, kind="cubic")


def lagrange(x: List[float], y: List[float]) -> callable:
    return interp.lagrange(x, y)
