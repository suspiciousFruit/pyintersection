#pragma once
#include "Leaf.h"
#include "./collision3d.h"
#include <iostream>


template <typename IIterator>
class LastLeaf : public Leaf<IIterator>
{
private:
	std::vector<IIterator> apoints_;
	std::vector<IIterator> bpoints_;
	using Leaf<IIterator>::cube_;
public:
	LastLeaf(double xDown, double xUp, double yDown, double yUp, double zDown, double zUp) :
		Leaf<IIterator>(xDown, xUp, yDown, yUp, zDown, zUp) // Прооптимизировать процесс передачи
	{ }

	LastLeaf(const cube3d& cube) : Leaf<IIterator>(cube)
	{ }

	LastLeaf() : Leaf<IIterator>()
	{ }

	virtual ~LastLeaf()
	{ }

	virtual void sieve_a(const IIterator p) override
	{
		if (cube_.isfit(p))
			apoints_.push_back(p);
	}

	virtual void sieve_b(const IIterator p) override
	{
		if (cube_.isfit(p))
			bpoints_.push_back(p);
	}

	virtual void make_childs(size_t) override
	{ }

	virtual void check_in_if_last(std::vector<LastLeaf*>& arr) override
	{
		arr.emplace_back(this);
	}

	virtual void print() const override
	{
		if (apoints_.size() == 0 && bpoints_.size() == 0)
			return;

		/*cube_.print();
		std::cout << "Base points:" << std::endl;
		for (const auto& p : apoints_)
			std::cout << *p << std::endl;

		std::cout << "Other points:" << std::endl;
		for (const auto& p : bpoints_)
			std::cout << *p << std::endl;*/
	}

	virtual void update_cube(double xDown, double xUp,
		double yDown, double yUp, double zDown, double zUp) override
	{
		cube_.update(xDown, xUp, yDown, yUp, zDown, zUp);
	}

	virtual void update_cube(const cube3d& cube)
	{
		cube_ = cube;
	}

	bool has_both_points() const
	{
		return apoints_.size() != 0 &&
			bpoints_.size() != 0;
	}

	std::vector<IIterator>&& get_apoints()
	{
		return std::move(apoints_);
	}

	std::vector<IIterator>&& get_bpoints()
	{
		return std::move(bpoints_);
	}

	const cube3d& get_cube() const
	{
		return cube_;
	}

	void clear_arrays()
	{
		apoints_.clear();
		bpoints_.clear();
	}
};
