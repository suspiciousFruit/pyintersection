import numpy as np

def cube_has_point(cube, point):
	p = point[4:7]
	c = cube[1:7]
	for i in range(3):
		if not (c[2 * i] <= p[i] <= c[2 * i + 1]) or point[0] != cube[0]:
			return False
	return True

def cubes_has_point(cubes, point):
	for cube in cubes:
		if cube_has_point(cube, point):
			return True
	return False

# No empty cubes
def test_all_cubes_has_some_points(cubes, points):
	pass

# No fantom points
def test_point_in_some_cube(cubes, points):
	return all([cubes_has_point(cubes, p) for p in points])