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
    """
    intersect3d(apoints, bpoints, atol=None, tree_depth=2)

    Find intersections between two manifolds.

    Parameters
    ----------
    apoints: (N, 5) numpy.ndarray
        First manifold.
    bpoints: (N, 5) numpy.ndarray
        Second manifold.
    atol: float > 0
        Target intersect tolerance.
    tree_depth: int > 0
        Depth of sieve tree.

    Returns
    -------
    rcubes: (N, 7) numpy.ndarray
        Intersection cubes (ndarray): [cid, x_down, x_up, y_down, y_up, z_down, z_up].
    points: (N, 7) numpy.ndarray
        Intersection points (numpy.ndarray): [cid, m, n, t, x, y, z].
    tols: (3,) numpy.ndarray
        Tolerance by all axis: [x_tol, y_tol, z_tol].

    Examples
    --------
    >>> import pyintersection as pyi
    >>> import numpy as np
    >>> apoints = np.array([[0, 0.0, 1, 1, 1], [1, 0.1, 1, 1, 2], [2, 0.2, 1, 1, 3]])
    >>> bpoints = np.array([[0, 0.3, 1, 2, 2], [1, 0.4, 1, 1, 2], [2, 0.5, 1, 0, 2]])
    >>> cubes, points, tols = pyi.intersect3d(apoints, bpoints, 0.5)
    >>> print(cubes)
        [[0.  1.  1.  1.  1.5 1.5 2. ]]
    >>> print(points)
        [[0.  0.  1.  0.1 1.  1.  2. ]
        [0.  1.  1.  0.1 1.  1.  2. ]]
    >>> print(tols)
        [0.  0.5 0.5]
    """
    check_manifold_type(a, 5)
    check_manifold_type(b, 5)
    check_tolerance(atol)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((0, 7)), np.zeros((0, 7)), np.zeros((3))
    return __intersect3d(a, b, atol, tree_depth)


def intersect6d(a, b, atol=None, tree_depth=2):
    """
    intersect6d(apoints, bpoints, atol=None, tree_depth=2)

    Find intersections between two manifolds.

    Parameters
    ----------
    apoints: (N, 8) numpy.ndarray
        First manifold.
    bpoints: (N, 8) numpy.ndarray
        Second manifold.
    atol: float > 0
        Target intersect tolerance.
    tree_depth: int > 0
        Depth of sieve tree.

    Returns
    -------
    rcubes: (N, 13) numpy.ndarray
        Intersection cubes (numpy.ndarray): [cid, x_down, x_up, y_down, y_up, z_down, z_up,
        vx_down, vx_up, vy_down, vy_up, vz_down, vz_up].
    points: (N, 10) numpy.ndarray
        Intersection points (numpy.ndarray): [cid, m, n, t, x, y, z, vx, vy, vz].
    tols: (6,) numpy.ndarray
        Tolerance by all axis: [x_tol, y_tol, z_tol, vx_tol, vy_tol, vz_tol].

    Examples
    --------
    >>> import pyintersection as pyi
    >>> import numpy as np
    >>> cubes, points, tols = pyi.intersect6d(np.array([]), np.array([]), 0.5)
    """
    check_manifold_type(a, 8)
    check_manifold_type(b, 8)
    check_tolerance(atol)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((0, 13)), np.zeros((0, 10)), np.zeros((6))
    return __intersect6d(a, b, atol, tree_depth)

def get_boundary_cube3d(a, b):
    """
    get_boundary_cube3d(apoints, bpoints)

    Find boundary cube for two manifolds.

    Parameters
    ----------
    apoints: (N, 5) numpy.ndarray
        First manifold.
    bpoints: (N, 5) numpy.ndarray
        Second manifold.

    Returns
    -------
    cube: (6,) numpy.ndarray
        Boundary cube (numpy.ndarray): [x_down, x_up, y_down, y_up, z_down, z_up].

    Examples
    --------
    """
    check_manifold_type(a, 5)
    check_manifold_type(b, 5)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros(6)
    return __get_boundary_cube3d(a, b)
