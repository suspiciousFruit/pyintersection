#pragma once
#include <fstream>



struct point3d
{
	union
	{
		struct { double x, y, z; };
		double data[3];
	};

	point3d(double a, double b, double c) : x(a), y(b), z(c)
	{ }

	point3d(const std::initializer_list<double>& list)
	{
		auto n = list.begin();

		for (size_t i = 0; i < 3; ++i, ++n)
			data[i] = *n;
	}

	point3d() : x(0), y(0), z(0)
	{ }

	inline bool operator== (const point3d& other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}

	inline bool operator!= (const point3d& other) const
	{
		return !(*this == other);
	}

	friend std::ostream& operator<< (std::ostream& stream, const point3d& p)
	{
		const char* del = ",";
		stream << p.x << del << p.y << del << p.z;

		return stream;
	}

	friend std::istream& operator>> (std::istream& stream, point3d& p)
	{
		char del;
		stream >> p.x >> del >> p.y >> del >> p.z;

		return stream;
	}
};
