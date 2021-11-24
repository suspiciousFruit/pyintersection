#pragma once
#include "Intersector3d.h"
#include "collision6d.h"


/*

template <typename Array6dT>
class Intersector6d
{
public:
	using Array6dT::SpaceAdapter;
	using Array6dT::VelocityAdapter;
	typedef decltype(__get_const_iterator<SpaceAdapter>()) SpaceInteratorT;
	typedef decltype(__get_const_iterator<VelocityAdapter>()) VelocityIteratorT;
private:
	Intersector3d<SpaceAdapter> inter_r_;
	Intersector3d<VelocityAdapter> inter_v_;
public:
	Intersector6d(size_t tree_depth) : inter_r_(tree_depth), inter_v_(tree_depth)
	{ }

	std::vector<collision6d<VelocityIterator>> intersect(
		const , double rprecision, double vprecision)
	{
		std::vector<collision6d<VelocityAdapter>> res;

		const auto colls_r = inter_r_.intesect(SpaceAdapter(a), SpaceAdapter(b), rprecision);

		for (const auto& coll : colls_r)
		{
			const auto colls_v = inter_v_.intesect(
				VelocityAdapter(coll.apoints), VelocityAdapter(coll.bpoints), vprecision);

			for (const auto& coll_v : colls_v)
				res.emplace_back(
					coll_v.apoints, coll_v.bpoints, coll_r.cube, coll_v.cube);
		}

		return res;
	}
};
*/



/*class ntpoint6d_iterator_v;

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
};


class RtoV
{
private:
	const ntpoint6d_iterator_r* data_;
	size_t size_;
public:
	typedef ntpoint6d_iterator_v const_iterator;

	RtoV(const std::vector<ntpoint6d_iterator_r>& vec) :
		data_(vec.data()), size_(vec.size())
	{ }

	RtoV() : data_(nullptr), size_(0)
	{ }

	ntpoint6d_iterator_v begin() const
	{
		return ntpoint6d_iterator_v(data_);
	}

	ntpoint6d_iterator_v end() const
	{
		return ntpoint6d_iterator_v(data_ + size_);
	}
};

class Adapter
{
private:
	const ntpoint6d* data_;
	size_t size_;
public:
	typedef ntpoint6d_iterator_r const_iterator;

	Adapter(const std::vector<ntpoint6d>& vec) :
		data_(vec.data()), size_(vec.size())
	{ }

	Adapter() : data_(nullptr), size_(0)
	{ }

	ntpoint6d_iterator_r begin() const
	{
		return ntpoint6d_iterator_r(data_);
	}

	ntpoint6d_iterator_r end() const
	{
		return ntpoint6d_iterator_r(data_ + size_);
	}
};



void f(const std::vector<ntpoint6d>& a, const std::vector<ntpoint6d>& b)
{
	const auto print = [](const auto& colls)
	{
		for (const auto& coll : colls)
		{
			std::cout << "APOINTS:\n";
			for (const auto& p : coll.apoints)
				std::cout << p << std::endl;
			std::cout << "BPOINTS:\n";
			for (const auto& p : coll.bpoints)
				std::cout << p << std::endl;
			std::cout << "\n\n";
		}
	};

	Intersector3d<Adapter> inter1(3);
	const auto colls_r = inter1.intersect(Adapter(a), Adapter(b), 1);

	std::cout << "SPACE PART\n\n";
	print(colls_r);

	std::cout << "VELOCITY PART\n\n";
	Intersector3d<RtoV> inter_v;

	for (const auto& coll : colls_r)
	{
		inter_v.set_arrays(RtoV(coll.apoints), RtoV(coll.bpoints));
		const auto colls = inter_v.intesect(0.2);
		print(colls);
	}
}*/