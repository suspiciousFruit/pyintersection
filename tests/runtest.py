import pymodule
import numpy as np

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


print(pymodule.intersect3d(a, b))