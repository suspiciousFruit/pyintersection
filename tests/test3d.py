import numpy as np
import pyintersection as pyi
import utils

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

def run():
	print('RUNNING TEST FOR pyintersection.intersect3d')
	tests = utils.Tests(dims=3)
	cubes, points = pyi.intersect3d(a3d, b3d, 0.5)
	tests.run(cubes, points, (a3d, b3d))