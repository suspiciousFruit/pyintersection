#pragma once
#include "Tree/point3d.h"



union point6d
{
	union
	{
		struct { point3d r, v; };
		double data[6];
	};

	point6d(const point3d& _r, const point3d& _v) : r(_r), v(_v)
	{ }

	point6d()
	{
		for (size_t i = 0; i < 6; ++i)
			data[i] = 0.0;
	}

	point6d(const std::initializer_list<double>& list)
	{
		auto n = list.begin();

		for (size_t i = 0; i < 6; ++i, ++n)
			data[i] = *n;
	}

	friend std::istream& operator>> (std::istream& stream, point6d& p)
	{
		double* data = (double*)(p.data);
		for (size_t i = 0; i < 6; ++i)
			stream >> data[i];

		return stream;
	}

	friend std::ostream& operator<< (std::ostream& stream, const point6d& p)
	{
		const char* del = ", ";

		for (size_t i = 0; i < 5; ++i)
			stream << p.data[i] << del;
		stream << p.data[5];

		return stream;
	}
};
