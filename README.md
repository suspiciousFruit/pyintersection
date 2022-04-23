# pyintersection

Overview
----------------------
The pyintersection package provides a set of functions for finding intersections of two parameter 3d or 6d manifolds. The package aims to simplify orbit calculations.

Documentation
----------------------
- **User documentation** https://github.com/suspiciousFruit/pyintersection/tree/master/doc/usr
- **Developer documentation** https://github.com/suspiciousFruit/pyintersection/tree/master/doc/dev

Hello World
----------------------
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
