pyintersection.intersect3d(apoints, bpoints, atol=None, tree_depth=2)
----------
Find intersections between two manifolds.

Parameters
----------
apoints : (N,5) ndarray
    First manifold.
bpoints : (N,5) ndarray
    Second manifold.
atol : float
    Min accuracy of intersection search.
tree_depth: int, optional
    Depth of sorting tree.
   

Returns
-------
out : tuple(cubes, points, tols)
    cubes: ndarray of cubes (ndarray): ``[cid, x_down, x_up, y_down, y_up, z_down, z_up]``
        ``cid - index of cube (unique in return ndarray)``
    points: ndarray of points (ndarray): ``[cid, m, n, t, x, y, z]``
        ``cid - cube id which contains this point`` ``m - manifold number (0 - apoints or 1 - bpoints)``
    tols: ndarray shape (3,) of tolerances:  ``[x_tol, y_tol, z_tol]``
           ``A_tol is absolute tolerence by coord A``

Notes
-----
This function return ``cubes`` and ``points``. ``cubes`` are ndarray of ndarrays which represented cubes. These cubes contain the probable intersection point of the manifolds.
``points`` are ndarray of ndarray which represented points. These are the closest points to the intersection of the manifolds.
Be careful, ``tree_depth`` affects the minimum accuracy of the calculation, since each layer of the tree divides each axis in half.

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