#pragma once
#include "LastLeaf.h"



template <typename IIterator>
class MiddleLeaf : public Leaf<IIterator>
{
private:
	Leaf<IIterator>* childs_[8];
	using Leaf<IIterator>::cube_;
public:
	MiddleLeaf(double xDown, double xUp, double yDown, double yUp, double zDown, double zUp) :
		Leaf<IIterator>(xDown, xUp, yDown, yUp, zDown, zUp)
	{ }

	MiddleLeaf(const cube3d& cube) : Leaf<IIterator>(cube)
	{ }

	MiddleLeaf() : Leaf<IIterator>()
	{ }

	virtual ~MiddleLeaf()
	{
		for (size_t i = 0; i < 8; ++i)
			if (childs_[i] != nullptr)
				delete childs_[i];
	}

	virtual void sieve_a(const IIterator p) override
	{
		for (size_t i = 0; i < 8; ++i)
		{
			if (childs_[i]->isfit(p))
			{
				childs_[i]->sieve_a(p);
				break;
			}
		}
	}

	virtual void sieve_b(const IIterator p) override
	{
		for (size_t i = 0; i < 8; ++i)
		{
			if (childs_[i]->isfit(p))
			{
				childs_[i]->sieve_b(p);
				break;
			}
		}
	}

	virtual void print() const override
	{
		for (size_t i = 0; i < 8; ++i)
			childs_[i]->print();
	}

	virtual void check_in_if_last(std::vector<LastLeaf<IIterator>*>& arr) override
	{
		for (size_t i = 0; i < 8; ++i)
			childs_[i]->check_in_if_last(arr);
	}

	virtual void make_childs(size_t depth) override
	{
		if (depth == 0) return;

		if (depth == 1)
		{
			double x_up, x_down, y_up, y_down, z_up, z_down;

			const double x_width = std::abs(cube_.x_up - cube_.x_down);
			const double y_width = std::abs(cube_.y_up - cube_.y_down);
			const double z_width = std::abs(cube_.z_up - cube_.z_down);

			z_down = cube_.z_down;
			z_up = cube_.z_down + z_width / 2;

			x_down = cube_.x_down;
			x_up = cube_.x_down + x_width / 2;

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[0] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down;
			y_up = cube_.y_down + y_width / 2;

			childs_[1] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			x_down = cube_.x_down + x_width / 2;
			x_up = cube_.x_up;

			childs_[2] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[3] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);
			//

			z_down = cube_.z_down + z_width / 2;
			z_up = cube_.z_up;

			x_down = cube_.x_down;
			x_up = cube_.x_down + x_width / 2;

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[4] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down;
			y_up = cube_.y_down + y_width / 2;

			childs_[5] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			x_down = cube_.x_down + x_width / 2;
			x_up = cube_.x_up;

			childs_[6] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[7] = new LastLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);
		}
		else
		{
			double x_up, x_down, y_up, y_down, z_up, z_down;

			const double x_width = std::abs(cube_.x_up - cube_.x_down);
			const double y_width = std::abs(cube_.y_up - cube_.y_down);
			const double z_width = std::abs(cube_.z_up - cube_.z_down);

			z_down = cube_.z_down;
			z_up = cube_.z_down + z_width / 2;

			x_down = cube_.x_down;
			x_up = cube_.x_down + x_width / 2;

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[0] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down;
			y_up = cube_.y_down + y_width / 2;

			childs_[1] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			x_down = cube_.x_down + x_width / 2;
			x_up = cube_.x_up;

			childs_[2] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[3] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);
			//

			z_down = cube_.z_down + z_width / 2;
			z_up = cube_.z_up;

			x_down = cube_.x_down;
			x_up = cube_.x_down + x_width / 2;

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[4] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down;
			y_up = cube_.y_down + y_width / 2;

			childs_[5] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			x_down = cube_.x_down + x_width / 2;
			x_up = cube_.x_up;

			childs_[6] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			y_down = cube_.y_down + y_width / 2;
			y_up = cube_.y_up;

			childs_[7] = new MiddleLeaf<IIterator>(x_down, x_up, y_down, y_up, z_down, z_up);

			for (size_t i = 0; i < 8; ++i)
				childs_[i]->make_childs(depth - 1);
		}
	}

	virtual void update_cube(double xDown, double xUp,
		double yDown, double yUp, double zDown, double zUp) override
	{
		cube_.update(xDown, xUp, yDown, yUp, zDown, zUp);

		this->update_cube(cube_);
	}

	virtual void update_cube(const cube3d& cube)
	{
		cube_ = cube;

		double x_up, x_down, y_up, y_down, z_up, z_down;

		const double x_width = std::abs(cube.x_up - cube.x_down);
		const double y_width = std::abs(cube.y_up - cube.y_down);
		const double z_width = std::abs(cube.z_up - cube.z_down);

		z_down = cube.z_down;
		z_up = cube.z_down + z_width / 2;

		x_down = cube.x_down;
		x_up = cube.x_down + x_width / 2;

		y_down = cube.y_down + y_width / 2;
		y_up = cube.y_up;

		childs_[0]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

		y_down = cube.y_down;
		y_up = cube.y_down + y_width / 2;

		childs_[1]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

		x_down = cube.x_down + x_width / 2;
		x_up = cube.x_up;

		childs_[2]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

		y_down = cube.y_down + y_width / 2;
		y_up = cube.y_up;

		childs_[3]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);
		//
		z_down = cube.z_down + z_width / 2;
		z_up = cube.z_up;

		x_down = cube.x_down;
		x_up = cube.x_down + x_width / 2;

		y_down = cube.y_down + y_width / 2;
		y_up = cube.y_up;

		childs_[4]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

		y_down = cube.y_down;
		y_up = cube.y_down + y_width / 2;

		childs_[5]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

		x_down = cube.x_down + x_width / 2;
		x_up = cube.x_up;

		childs_[6]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

		y_down = cube.y_down + y_width / 2;
		y_up = cube.y_up;

		childs_[7]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);
	}
};


template <typename IIterator>
void split_cube(const cube3d& cube, Leaf<IIterator>* leaves)
{
	double x_up, x_down, y_up, y_down, z_up, z_down;

	const double x_width = std::abs(cube.x_up - cube.x_down);
	const double y_width = std::abs(cube.y_up - cube.y_down);
	const double z_width = std::abs(cube.z_up - cube.z_down);

	z_down = cube.z_down;
	z_up = cube.z_down + z_width / 2;

	x_down = cube.x_down;
	x_up = cube.x_down + x_width / 2;

	y_down = cube.y_down + y_width / 2;
	y_up = cube.y_up;

	leaves[0]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

	y_down = cube.y_down;
	y_up = cube.y_down + y_width / 2;

	leaves[1]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

	x_down = cube.x_down + x_width / 2;
	x_up = cube.x_up;

	leaves[2]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

	y_down = cube.y_down + y_width / 2;
	y_up = cube.y_up;

	leaves[3]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);
	//
	z_down = cube.z_down + z_width / 2;
	z_up = cube.z_up;

	x_down = cube.x_down;
	x_up = cube.x_down + x_width / 2;

	y_down = cube.y_down + y_width / 2;
	y_up = cube.y_up;

	leaves[4]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

	y_down = cube.y_down;
	y_up = cube.y_down + y_width / 2;

	leaves[5]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

	x_down = cube.x_down + x_width / 2;
	x_up = cube.x_up;

	leaves[6]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);

	y_down = cube.y_down + y_width / 2;
	y_up = cube.y_up;

	leaves[7]->update_cube(x_down, x_up, y_down, y_up, z_down, z_up);
}