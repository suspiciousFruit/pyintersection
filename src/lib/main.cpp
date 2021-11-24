#include <iostream>
#include <vector>
#include "Intersector/point6d.h"
#include "Intersector/Intersector3d.h"

std::vector<point3d> bpoints = {
	{1, 1, 0.03},
	{1, 1, 0.5},
	{1, 1, 2.1},
	{1, 1, 1.9},
	{1, 1, 1.1},
	{1, 1, 0.8},
	{1, 1, 1.5},
	{1, 1, 0.2},

	{1, 4, 3.1},

	{0.1, 0.4, 0.7},
	{6.7, 6.7, 5.1},
	{6.7, 7, 5.1},
	{6.7, 6.7, 5},
	{6.7, 7, 5}
};

std::vector<point3d> apoints = {
	{1, 1, 0},
	{1, 4, 3},
	{7, 7, 5}
};

std::vector<point6d> other6d = {
	{1, 1, 0.9, 3, 3, 3},
	{1, 1, 0.5, 3.1, 2.9, 3},
	{1, 1, 2.1, 3, 3, 3},
	{1, 1, 1.9, 3.1, 3, 2.8},
	{1, 1, 1.1, 3.1, 3, 2.8},
	{1, 1, 0.8, 3.1, 3, 2.8},
	{1, 1, 1.5, 3.1, 3, 2.8},
	{1, 4, 3.1, 3.1, 2.9, 3},
	{0.1, 0.4, 0.7, 3.1, 2.9, 3},
	{6.7, 6.7, 5.1, 2.7, 3.3, 3.5},
	{7, 7, 5, 2.9, 3, 3.1},
	{6.7, 6.7, 5, 2.7, 3.1, 3.5},
	{6.7, 7, 5, 2.9, 3.3, 3.6},
	{1, 1, 0.2, 2.7, 3.3, 3.5}
};

std::vector<point6d> base6d = {
	{1, 1, 0, 3.1, 2.9, 3},
	{1, 4, 3, 3.1, 2.8, 3},
	{7, 7, 5, 3.1, 2.8, 3.1}
};


#include "Test/Scripts/Test.h"

std::vector<ntpoint6d> a6d = {
	{3, 3, 1, 1, 0.9, 3, 3, 3},
	{3, 3, 1, 1, 0.5, 3.1, 2.9, 3},
	{3, 3, 1, 1, 2.1, 3, 3, 3},
	{3, 3, 1, 1, 1.9, 3.1, 3, 2.8},
	{3, 3, 1, 1, 1.1, 3.1, 3, 2.8},
	{3, 3, 1, 1, 0.8, 3.1, 3, 2.8},
	{3, 3, 1, 1, 1.5, 3.1, 3, 2.8},
	{3, 3, 1, 4, 3.1, 3.1, 2.9, 3},
	{3, 3, 0.1, 0.4, 0.7, 3.1, 2.9, 3},
	{3, 3, 6.7, 6.7, 5.1, 2.7, 3.3, 3.5},
	{3, 3, 7, 7, 5, 2.9, 3, 3.1},

	{3, 3, 6.7, 6.7, 5, 2.7, 3.1, 3.5},
	{3, 3, 6.7, 7, 5, 2.9, 3.3, 3.6},
	{3, 3, 1, 1, 0.2, 2.7, 3.3, 3.5}
};

std::vector<ntpoint6d> b6d = {
	{3, 3, 1, 1, 0, 3.1, 2.9, 3},
	{3, 3, 1, 4, 3, 3.1, 2.8, 3},
	{3, 3, 7, 7, 5, 3.1, 2.8, 3.1},

	{3, 3, 6.7, 6.8, 5, 2.7, 3.1, 3.5}
};

#include "Test/Scripts/CompileTest.h"
#include "Intersector/Intersector6d.h"

// TODO make hierarchy of collision


int main()
{
	//test__runall();

	/*intersect3d_test("test_plane_0.csv",
		"test_plane_1.csv",
		"test_plane_int.csv");

	intersect3d_test("test_manifold_3d_0.csv",
		"test_manifold_3d_1.csv",
		"test_manifold_3d_int.csv");*/

	intersect_and_write_tables("SEL1_190d_1419tr_3d.csv",
		"SEL2_190d_1437tr_3d.csv",
		"test_manifold_3d_cubes_int",
		"test_manifold_3d_points_int",
		{1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10});

	//f(a6d, b6d);

	return 0;
}
