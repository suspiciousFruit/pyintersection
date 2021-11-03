#pragma once
#include "Intersector/algorithm_point3d_ptr.h"
#include <map>



struct ncollision3d
{
public:
	std::vector<npoint_iterator> apoints;
	std::vector<npoint_iterator> bpoints;
	cube3d cube;


	bool has_number(size_t number) const
	{
		for (size_t i = 0; i < apoints.size(); ++i)
			if (apoints[i].getnumber() == number)
				return true;

		return false;

		return std::find_if(std::begin(apoints), std::end(apoints),
			[&number](const npoint_iterator& it) { it.getnumber() == number; })
			!= std::end(apoints);
	}
};

auto linear(const std::vector<ncollision3d>& colls)
{
	std::map<double, std::vector<cube3d>> map;

	for (auto& [number, cubes] : map)
		for (const auto& coll : colls)
			if (coll.has_number(number))
				cubes.push_back(coll.cube);

	return map;
}

