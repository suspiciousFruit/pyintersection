#pragma once
#include <vector>
#include <iostream>
#include "./cube3d.h"
#include "./point3d.h"


template <typename IIterator>
struct collision3d
{
	std::vector<IIterator> apoints;
	std::vector<IIterator> bpoints;
	cube3d cube;

	collision3d(std::vector<IIterator>&& a, std::vector<IIterator>&& b, const cube3d& c) :
		apoints(a), bpoints(b), cube(c)
	{ }

	friend std::ostream& operator<< (std::ostream& st, const collision3d& p)
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


