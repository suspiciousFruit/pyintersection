import numpy as np
import pyintersection as pyi



def get_manifold_number(rpoint):
    return rpoint[1]

def get_rpoint_params(rpoint):
    return rpoint[2], rpoint[3]

def generate_new_points(rpoints, atol=None, agen=None, bgen=None):
    ares, bres = [], []
    for rpoint in rpoints:
        n, t = get_rpoint_params(rpoint)
        manifold = get_manifold_number(rpoint)
        
        if manifold == 0:
            new_points = agen(n, t, atol)
            ares.append(new_points)
        elif manifold == 1:
            new_points = bgen(n, t, atol)
            bres.append(new_points)
        else:
            raise Exception("Wrong manifold number")
          
    counter = 0
    for nparr in ares:
        for unarr in nparr:
            counter += 1
    ares = np.reshape(np.array(ares), (counter, 5))
            
    counter = 0
    for nparr in bres:
        for unarr in nparr:
            counter += 1
    bres = np.reshape(np.array(bres), (counter, 5))
        
    return ares, bres

def optimize3d(a, b, target_atol=None, agen=None, bgen=None, max_iter=8):
    """
    optimize3d(apoints, bpoints, target_atol=None,
            agen=None, bgen=None, max_iter=8)

    Find intersections between two manifolds.

    Parameters
    ----------
    apoints: (N, 5) numpy.ndarray
        First manifold.
    bpoints: (N, 5) numpy.ndarray
        Second manifold.
    target_atol: float > 0
        Target optimize tolerance.
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
    atol = max(pyi.get_boundary_cube3d(a, b))
    for i in range(max_iter):
        rcubes, rpoints, atols = pyi.intersect3d(a, b, atol=atol/2)
        #print(f'Iteration {i}: target_atols={atol/2} real_atols={max(atols)} rpoins={len(rpoints)} cubes={len(rcubes)}')
        if max(atols) <= target_atol:
            break
        atol = max(atols)
        a, b = generate_new_points(rpoints, atol=atol, agen=agen, bgen=bgen)
    return rcubes, rpoints, atols