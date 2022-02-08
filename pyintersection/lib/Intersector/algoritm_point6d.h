#pragma once
#include <algorithm>
#include "./algorithm_point3d_ptr.h"
#include "./point6d.h"
#include "./cube6d.h"



class LowerWordAdapter
{
private:
	const point6d* array_;
	const size_t size_;
public:
	LowerWordAdapter(const point6d* ptr, const size_t size) :
		array_(ptr), size_(size)
	{ }

	inline const point3d* operator[] (size_t index) const
	{
		return &array_[index].r;
	}

	inline size_t size() const
	{
		return size_;
	}
};

class UpperWordAdapter
{
private:
	const point3d* const * array_;
	const size_t size_;
public:
	UpperWordAdapter(const point3d* const * ptr, const size_t size) :
		array_(ptr), size_(size)
	{ }

	inline const point3d* operator[] (size_t index) const
	{
		return &((const point6d*)array_[index])->v;
	}

	inline size_t size() const
	{
		return size_;
	}
};

std::vector<collision3d> __find6(
	const std::vector<point6d>& base,
	const std::vector<point6d>& other,
	const size_t iteration_number,
	const cube3d& cube
)
{
	LowerWordAdapter lowera(base.data(), base.size());
	LowerWordAdapter lowerb(other.data(), other.size());

	Lily lily(lowera, lowerb, cube, 1);

	return lily.make_iterations(iteration_number);
}


template <typename ... Args>
void log(const char* s, Args ... args)
{
	printf(s, args ...);
	putchar('\n');
}


/*
	class IArray
	{
	public:
		const point6d* data() const;
		size_t size() const;
	};
*/
template <typename IArray>
std::vector<cube6d> find6d(
	const IArray& a, const IArray& b, const cube3d& cube,
	size_t iteration_number = 1)
{
	std::vector<cube6d> res;
	const auto collisions = _find3ptr(
		LowerWordAdapter(a.data(), a.size()),
		LowerWordAdapter(b.data(), b.size()), cube, iteration_number);

	const cube3d mcube(0, 10, 0, 10, 0, 10);

	log("Start new comparing phase");

	for (const auto& c : collisions)
	{
		const auto& base = c.base_points;
		const auto& other = c.other_points;
		const std::vector<collision3d> rs = _find3ptr(
			UpperWordAdapter(base.data(), base.size()),
			UpperWordAdapter(other.data(), other.size()),
			mcube, iteration_number);
		for (const auto& r : rs)
			res.push_back(cube6d(c.cube, r.cube));
	}

	return res;
}

