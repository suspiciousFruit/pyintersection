import pymodule
import numpy as np
from testutils import *

a = np.array([
	[1, 0.0, 1, 2, 3],
	[2, 0.1, 1, 2, 3],
	[3, 0.2, 6, 2, 3],
	[4, 0.3, 1, 2, 3],
])

b = np.array([
	[1, 0.0, 6, 2, 3],
	[2, 0.1, 1, 2, 3],
	[3, 0.2, 1, 2, 3],
	[4, 0.3, 1, 2, 5],
])

cubes, points = pymodule.intersect3d(a, b, 0.5)
#print(cubes, points)
print (test_point_in_some_cube(cubes, points))