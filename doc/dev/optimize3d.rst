pyintersection.optimize3d(apoints, bpoints, target_atol=None, agen=None, bgen=None, max_iter=8, tree_depth=2)
----------
Find intersections between two manifolds.

Parameters
----------
apoints : numpy.ndarray with shape (N, 5)
    First initial manifold.
bpoints : numpy.ndarray with shape (N, 5)
    Second initial manifold.
target_atol : float > 0
    Min accuracy of intersection search.
agen: lambda n, t, atol: numpy.ndarray with shape (N, 5)
    Manifold generator for a-manifold. Generate new points around point with coords n, t.
bgen: lambda n, t, atol: numpy.ndarray with shape (N, 5)
    Manifold generator for b-manifold. Generate new points around point with coords n, t.
max_iter: int > 0
    Max count of optimize iterations.
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

Examples
--------
>>> import pyintersection as pyi
