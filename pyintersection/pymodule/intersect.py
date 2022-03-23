import numpy as np
from extmodule import __intersect3d, __intersect6d, __get_boundary_cube3d


def check_manifold_type(a, dims):
    if not isinstance(a, np.ndarray):
        raise Exception("Error: ")

    if a.dtype != 'float64':
        raise Exception("Error: type")

    if len(a.shape) != 2:
        raise Exception("Error: shape")

    if a.shape[1] != dims:
        raise Exception("Error: shape")


def check_tolerance(atol):
    if not isinstance(atol, float):
        raise Exception("Error: tolerance should has float type")

    if atol <= 0:
        raise Exception("Error: tolerance should be greater than 0")


def intersect3d(a, b, atol=None, tree_depth=2):
    check_manifold_type(a, 5)
    check_manifold_type(b, 5)
    check_tolerance(atol)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((0, 7)), np.zeros((0, 7)), np.zeros((3))
    return __intersect3d(a, b, atol, tree_depth)


def intersect6d(a, b, atol=None, tree_depth=2):
    check_manifold_type(a, 8)
    check_manifold_type(b, 8)
    check_tolerance(atol)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((0, 13)), np.zeros((0, 10)), np.zeros((6))
    return __intersect6d(a, b, atol, tree_depth)

def get_boundary_cube3d(a, b):
    check_manifold_type(a, 5)
    check_manifold_type(b, 5)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros(6)
    return __get_boundary_cube3d(a, b)