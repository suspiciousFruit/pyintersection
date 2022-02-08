import numpy as np
from extmodule import __intersect3d, __intersect6d



def check_manifold_type(a):
    if not isinstance(a, np.ndarray):
        raise Exception("Error: ")

    if a.dtype != 'float64':
        raise Exception("Error: type")

    if len(a.shape) != 2:
        raise Exception("Error: shape")

    if a.shape == (0, 0):
        return


def check_tolerance(atol, rtol):
    if not isinstance(atol, float):
        raise Exception("Error: tolerance float")

    if atol <= 0:
        raise Exception("Error: tolerance")


def intersect3d(a, b, atol=None, rtol=None, cube=None):
    check_manifold_type(a)
    check_manifold_type(b)
    check_tolerance(atol, rtol)
    return __intersect3d(a, b, atol)


def intersect6d(a, b, atol=None, rtol=None, cube=None):
    check_manifold_type(a)
    check_manifold_type(b)
    check_tolerance(atol, rtol)
    return __intersect6d(a, b, atol)
