import pyintersection as pyi
import numpy as np


a3d = np.array([
	[1, 0.0, 1, 2, 3],
	[2, 0.1, 1, 2, 3],
	[3, 0.2, 6, 2, 3],
	[4, 0.3, 1, 2, 3],
])

b3d = np.array([
	[1, 0.0, 6, 2, 3],
	[2, 0.1, 1, 2, 3],
	[3, 0.2, 1, 2, 3],
	[4, 0.3, 1, 2, 5],
])

cube = pyi.get_boundary_cube3d(a3d, b3d)

print(np.all(cube, np.array([1.0, 6.0, 2.0, 2.0, 3.0, 5.0])))
print("cube", cube)