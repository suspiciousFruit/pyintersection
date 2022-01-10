#pragma once
// Move file to other dir
#include "../Intersector/point6d.h"



struct ntpoint6d
{
	double number;
	double t;
	point6d point;

	ntpoint6d(const std::initializer_list<double>& list)
	{
		auto iter = list.begin();
		number = *iter;
		++iter;
		t = *iter;
		++iter;
		for (size_t i = 0; i < 6; ++i)
		{
			point.data[i] = *iter;
			++iter;
		}
	}

	friend std::istream& operator>> (std::istream& s, ntpoint6d& p)
	{
		char del;
		s >> p.number >> del >> p.t >> del >> p.point;
		return s;
	}

	friend std::ostream& operator<< (std::ostream& s, const ntpoint6d& p)
	{
		const char del = ',';
		s << p.number << del << p.t << del << p.point;
		return s;
	}
};

class ntpoint6d_iterator_v;

// Iterator for space dimention
class ntpoint6d_iterator_r
{
private:
	friend class ntpoint6d_iterator_v;
	const ntpoint6d* base_;
public:
	ntpoint6d_iterator_r(const ntpoint6d* base) : base_(base)
	{ }

	const point3d& operator* () const
	{
		return base_->point.r;
	}

	ntpoint6d_iterator_r& operator++ ()
	{
		++base_;
		return *this;
	}

	bool operator!= (const ntpoint6d_iterator_r other)
	{
		return base_ != other.base_;
	}

	friend std::ostream& operator<< (std::ostream& s, const ntpoint6d_iterator_r it)
	{
		s << *(it.base_);
		return s;
	}
};

// Iterator for velocity dimention based on space iterator
class ntpoint6d_iterator_v
{
private:
	const ntpoint6d_iterator_r* iterator_r_;
public:
	ntpoint6d_iterator_v(const ntpoint6d_iterator_r* base) :
		iterator_r_(base)
	{ }

	const point3d& operator* () const
	{
		return iterator_r_->base_->point.v;
	}

	ntpoint6d_iterator_v& operator++ ()
	{
		++iterator_r_;
		return *this;
	}

	bool operator!= (const ntpoint6d_iterator_v other)
	{
		return iterator_r_ != other.iterator_r_;
	}

	friend std::ostream& operator<< (std::ostream& s, ntpoint6d_iterator_v p)
	{
		s << *(p.iterator_r_);
		return s;
	}

	inline const ntpoint6d& ntpoint() const {
		return *iterator_r_->base_;
	}
};
