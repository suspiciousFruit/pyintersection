pyintersection.intersect6d(apoints, bpoints, atol=None, tree_depth=2)
----------
Find intersections between two 6d manifolds.

Parameters
----------
apoints : (N,8) ndarray
    First manifold.
bpoints : (N,8) ndarray
    Second manifold.
atol : float
    Min accuracy of intersection search.
tree_depth: int, optional
    Depth of sorting tree.
   

Returns
-------
out : tuple(cubes, points, tols)
    cubes: ndarray of cubes (ndarray): ``[cid, x_down, x_up, y_down, y_up, z_down, z_up, vx_down, vx_up, vy_down, vy_up, vz_down, vz_up]``
        ``cid - index of cube (unique in return ndarray)``
    points: ndarray of points (ndarray): ``[cid, m, n, t, x, y, z, vx, vy, vz]``
        ``cid - cube id which contains this point`` ``m - manifold number (0 - apoints or 1 - bpoints)``
    tols: ndarray shape (6,) of tolerances:  ``[x_tol, y_tol, z_tol, vx_tol, vy_tol, vz_tol]``
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
>>> cubes, points, tols = pyi.intersect6d(np.array([]), np.array([]), 0.5)
