#pragma once
#include <limits>
#include "Tree/cube3d.h"
#include <vector>
#include <algorithm>



template <typename ArrayT>
cube3d get_boundary_cube3d(const ArrayT& points)
{
	double dmax = std::numeric_limits<double>::max();
	
	point3d min(dmax, dmax, dmax);
	point3d max(-dmax, -dmax, -dmax);

	for (const auto& point : points) // Bad code
	{
		if (point.x < min.x)
			min.x = point.x;
		if (point.x > max.x)
			max.x = point.x;

		if (point.y < min.y)
			min.y = point.y;
		if (point.y > max.y)
			max.y = point.y;

		if (point.z < min.z)
			min.z = point.z;
		if (point.z > max.z)
			max.z = point.z;
	}

	return cube3d(min.x, max.x, min.y, max.y, min.z, max.z);
}


template <typename ArrayT>
cube3d get_boundary_cube3d(const ArrayT& apoints, const ArrayT& bpoints)
{
	cube3d a = get_boundary_cube3d(apoints);
	cube3d b = get_boundary_cube3d(bpoints);

	const double x_down = std::min(a.x_down, b.x_down);
	const double y_down = std::min(a.y_down, b.y_down);
	const double z_down = std::min(a.z_down, b.z_down);

	const double x_up = std::max(a.x_up, b.x_up);
	const double y_up = std::max(a.y_up, b.y_up);
	const double z_up = std::max(a.z_up, b.z_up);

	return cube3d(x_down, x_up, y_down, y_up, z_down, z_up);
}



//template <typename IArray>
/*cube6d getcube6d(const point6d* arr, size_t size)
{
	const point6d* data = arr;
	double cube[12];

	for (size_t i = 0; i < size; ++i)
	{
		const double* d = data[i].data;
		
		for (size_t j = 0; i < 6; ++i)
		{
			if (cube[j] < d[j])
				cube[j] = d[j];
			else if (cube[j + 1] > d[j])
				cube[j + 1] = d[j];
		}
	}

	return cube6d(cube);
}*/
