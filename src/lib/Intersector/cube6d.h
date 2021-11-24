#pragma once
#include "Tree/cube3d.h"

union cube6d
{
	struct
	{
		cube3d a;
		cube3d b;
	};
	double data[12];

	cube6d(const cube3d& q, const cube3d& w) :
		a(q), b(w)
	{ }

	cube6d(const double* d)
	{
		for (size_t i = 0; i < 6; ++i)
			data[i] = d[i];
	}

	friend std::ostream& operator<< (std::ostream& s, const cube6d& c)
	{
		s << '[';
		for (size_t i = 0; i < 11; ++i)
			s << c.data[i] << ", ";
		s << c.data[11] << ']';

		return s;
	}
};
