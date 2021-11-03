#pragma once
#include "../Intersector/Tree/point3d.h"

struct npoint3d
{
	double number;
	point3d point;

	npoint3d() : point(0, 0, 0), number(0.0)
	{ }

	npoint3d(const std::initializer_list<double>& list)
	{
		auto iter = list.begin();
		number = *iter;

		for (size_t i = 0; i < 3; ++i)
		{
			++iter;
			point.data[i] = *iter;
		}
	}

	friend std::istream& operator>> (std::istream& stream, npoint3d& p)
	{
		stream >> p.number >> p.point;

		return stream;
	}

	friend std::ostream& operator<< (std::ostream& stream, const npoint3d& p)
	{
		stream << p.number << '#' << p.point;

		return stream;
	}
};

class npoint_iterator
{
private:
	const npoint3d* base_;
public:
	explicit npoint_iterator(const npoint3d* base) : base_(base)
	{ }

	inline const point3d& operator* () const
	{
		return base_->point;
	}

	inline npoint_iterator& operator++ ()
	{
		++base_;

		return *this;
	}

	inline bool operator!= (const npoint_iterator other) const
	{
		return base_ != other.base_;
	}

	inline bool operator!= (const npoint3d* other) const
	{
		return base_ != other;
	}

	inline double getnumber() const
	{
		return base_->number;
	}

	inline const point3d& getpoint() const
	{
		return base_->point;
	}
};

