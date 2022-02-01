import numpy as np
import pyintersection as pyi
import utils

a6d = np.array([
    [3, 3, 1, 1, 0.9, 3, 3, 3],
    [3, 3, 1, 1, 0.5, 3.1, 2.9, 3],
    [3, 3, 1, 1, 2.1, 3, 3, 3],
    [3, 3, 1, 1, 1.9, 3.1, 3, 2.8],
    [3, 3, 1, 1, 1.1, 3.1, 3, 2.8],
    [3, 3, 1, 1, 0.8, 3.1, 3, 2.8],
    [3, 3, 1, 1, 1.5, 3.1, 3, 2.8],
    [3, 3, 1, 4, 3.1, 3.1, 2.9, 3],
    [3, 3, 0.1, 0.4, 0.7, 3.1, 2.9, 3],
    [3, 3, 6.7, 6.7, 5.1, 2.7, 3.3, 3.5],
    [3, 3, 7, 7, 5, 2.9, 3, 3.1],
    [3, 3, 6.7, 6.7, 5, 2.7, 3.1, 3.5],
    [3, 3, 6.7, 7, 5, 2.9, 3.3, 3.6],
    [3, 3, 1, 1, 0.2, 2.7, 3.3, 3.5]
])

b6d = np.array([
  	[3, 3, 1, 1, 0.9, 3, 3, 3],
	[3, 3, 1, 4, 3, 3.1, 2.8, 3],
	[3, 3, 7, 7, 5, 3.1, 2.8, 3.1],
	[3, 3, 6.7, 6.8, 5, 2.7, 3.1, 3.5]  
])

cubes, points = pyi.intersect6d(a6d, b6d, 0.5)


def run():
    print('RUNNING TEST FOR pyintersection.intersect6d')
    tests = utils.Tests(dims=6)
    cubes, points = pyi.intersect6d(a6d, b6d, 0.5)
    tests.run(cubes, points, (a6d, b6d))



