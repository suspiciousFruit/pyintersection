#pragma once
#include "../Intersector/Tree/point3d.h"



struct ntpoint3d
{
	double number;
	double t;
	point3d point;

	ntpoint3d() : point(0, 0, 0), number(0.0), t(0.0)
	{ }

	ntpoint3d(const std::initializer_list<double>& list)
	{
		auto iter = list.begin();
		number = *iter;
		++iter;
		t = *iter;

		for (size_t i = 0; i < 3; ++i)
		{
			++iter;
			point.data[i] = *iter;
		}
	}

	friend std::istream& operator>> (std::istream& stream, ntpoint3d& p)
	{
		char del;
		stream >> p.number >> del
				>> p.t >> del
				>> p.point;

		return stream;
	}

	friend std::ostream& operator<< (std::ostream& stream, const ntpoint3d& p)
	{
		const char del = ',';
		stream << p.number << del
				<< p.t << del
				<< p.point;

		return stream;
	}
};



class ntpoint_iterator
{
private:
	const ntpoint3d* base_;
public:
	explicit ntpoint_iterator(const ntpoint3d* base) : base_(base)
	{ }

	inline const point3d& operator* () const
	{
		return base_->point;
	}

	inline ntpoint_iterator& operator++ ()
	{
		++base_;

		return *this;
	}

	inline bool operator!= (const ntpoint_iterator other) const
	{
		return base_ != other.base_;
	}

	inline bool operator!= (const ntpoint3d* other) const
	{
		return base_ != other;
	}

	inline double getnumber() const
	{
		return base_->number;
	}

	inline double gett() const
	{
		return base_->t;
	}

	inline const point3d& getpoint() const
	{
		return base_->point;
	}

	friend std::ostream& operator<< (std::ostream& stream, const ntpoint_iterator it)
	{
		const char del = ',';
		stream << it.base_->number << del << it.base_->t << del << *it;
		return stream;
	}
};