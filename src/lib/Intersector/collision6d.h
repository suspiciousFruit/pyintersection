#pragma once
#include <vector>
#include "cube6d.h"


template <typename IIterator>
struct collision6d
{
	std::vector<IIterator> apoints;
	std::vector<IIterator> bpoints;
	cube6d cube;

	collision6d(std::vector<IIterator>&& a, std::vector<IIterator>&& b, const cube6d& c) :
		apoints(a), bpoints(b), cube(c)
	{ }

	collision6d(std::vector<IIterator>&& a, std::vector<IIterator>&& b,
		const cube3d& acube, const cube3d& bcube)
		: apoints(a), bpoints(b), cube(acube, bcube)
	{ }

	friend std::ostream& operator<< (std::ostream& st, const collision6d& p)
	{
		st << "Possible collision " << p.cube << std::endl;

		st << "apoints:" << std::endl;
		for (auto p : p.apoints)
			std::cout << "    " << *p << std::endl;
		st << "bpoints:" << std::endl;
		for (auto p : p.bpoints)
			std::cout << "    " << *p << std::endl;

		return st;
	}
};




