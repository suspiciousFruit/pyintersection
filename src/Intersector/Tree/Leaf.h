#pragma once
#include <vector>
#include <iostream>
#include "./cube3d.h"



// Template - type of object which stored by LastLeaf
template <typename IIterator>
class LastLeaf;

// Base class for all octo leaves
template <typename IIterator>
class Leaf
{
protected:
	cube3d cube_;
public:
	Leaf(double xDown, double xUp, double yDown, double yUp, double zDown, double zUp) :
		cube_(xDown, xUp, yDown, yUp, zDown, zUp) // Стоит прооптимизировать процесс передачи
	{ }

	Leaf(const cube3d& cube) : cube_(cube) { }
	Leaf() { }

	virtual ~Leaf() { }

	virtual void sieve_a(const IIterator) = 0;
	virtual void sieve_b(const IIterator) = 0;

	virtual void make_childs(size_t depth) = 0; // recurrent
	virtual void check_in_if_last(std::vector<LastLeaf<IIterator>*>&) = 0; // recurrent

	virtual void print() const = 0; // recurrent

	virtual void update_cube(double xDown, double xUp, double yDown, double yUp, double zDown, double zUp) = 0; // recurrent
	virtual void update_cube(const cube3d&) = 0;

	inline bool isfit(const IIterator p) const { return cube_.isfit(p); };
	inline const cube3d& get_cube() const { return cube_; };
};
