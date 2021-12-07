#pragma once
#include <cmath>
#include "./Tree/Tree3d.h"
#include "boundary.h"



/*
	����� ������� ���� ����������� � ����������� ���������
	��� �������� - ���������, ������� �������� ����� � �������� ����������� IArray
	������������� ���� ��� �� ����������� ��� � ��������� � ������ Tree

	ArrayT::const_iterator �������� ��� ������������ ���������
	(�������� ������� ����� ���� �������, ������ �� �� ��� �� ��������� �� ����� ����������)
*/

// Calculate iteration number for octree
size_t calculate_iteration_number(double precision, const cube3d& base, size_t depth)
{
	const double sievesx = std::log2((base.x_up - base.x_down) / precision);
	const double sievesy = std::log2((base.y_up - base.y_down) / precision);
	const double sievesz = std::log2((base.z_up - base.z_down) / precision);

	return (size_t)std::ceil(std::max({ sievesx, sievesy, sievesz }) / depth);
}

template <typename T>
auto __get_const_iterator()
{
	const T t;
	return std::begin(t);
}

// TODO Think about friendly interface
// TODO Add the ability to explicitly specify the cube to search for

template <typename ArrayT>
class Intersector3d
{
private:
	const ArrayT* apoints_;
	const ArrayT* bpoints_;

	typedef decltype(__get_const_iterator<ArrayT>()) IteratorT;

	std::vector<collision3d<IteratorT>> cur_collisions_;
	std::vector<collision3d<IteratorT>> new_collisions_;

	Tree<IteratorT> tree_; // �������� ������������ ������ � ����, ��� runtime ���������
public:
	//Intersector3d(const ArrayT& apoints, const ArrayT& bpoints, size_t depth = 1)
	//	: tree_(depth), apoints_(&apoints), bpoints_(&bpoints)
	//{
	//	tree_.update_cube(get_boundary_cube3d(apoints, bpoints));
	//}
	// Default constructor
	Intersector3d(size_t tree_depth = 1) : apoints_(nullptr), bpoints_(nullptr), tree_(tree_depth)
	{ }
private:
	void make_first_iteration()
	{
		/*
		   ���������� ����� ������ ���������� ��������� ���������
		   ������� ������� ��������� �� �����! ����� ����� ����� �������� ����
		   ���� ������� sieve
		*/
		// TODO check for nullptr
		tree_.sieve_a(*apoints_); // vector<point3d>
		tree_.sieve_b(*bpoints_);

		//tree_.print_buffers();

		tree_.get_full_buffers(cur_collisions_);
	}

	void make_iteration()
	{
		for (const auto& collision : cur_collisions_)
		{
			// ��������� ��������� ��� ������
			tree_.update_cube(collision.cube);

			// ���������� �����
			tree_.sieve_a_iterable(collision.apoints);
			tree_.sieve_b_iterable(collision.bpoints);

			// ��������� ������ ����������
			tree_.get_full_buffers(new_collisions_);
			// ������� ������
			tree_.clean_buffers();
		}

		cur_collisions_.clear();
		std::swap(new_collisions_, cur_collisions_);
	}

	void print_iteration(size_t i) const
	{
		printf("##################### Iteration %d ####################################\n", i);
		for (const auto& ps : cur_collisions_)
			std::cout << ps << std::endl;
	}

	std::vector<collision3d<IteratorT>>&& make_iterations(size_t number_of_iterations)
	{
		// std::cout << "Need iterations: " << number_of_iterations << std::endl;
		// ������ ������ ������
		make_first_iteration();

		// ������� ���������� ������� ������
		tree_.clean_buffers();

		for (size_t i = 1; i < number_of_iterations; ++i)
		{
			// ������ ��������
			make_iteration();
			// ������� ���������� �������
			tree_.clean_buffers();
		}

		return std::move(cur_collisions_);
	}

	size_t get_iteration_number(double precision) const
	{
		const cube3d& cube = tree_.get_cube();
		const size_t depth = tree_.get_depth();

		return calculate_iteration_number(precision, cube, depth);
	}
public:
	std::vector<collision3d<IteratorT>>&& intersect(const ArrayT& a, const ArrayT& b,
		double precision)
	{
		tree_.update_cube(get_boundary_cube3d(a, b));
		const size_t it_num = calculate_iteration_number(
			precision, tree_.get_cube(), tree_.get_depth());

		// ���� �� ����� ��������� � ��������� ������?
		apoints_ = &a;
		bpoints_ = &b;

		return this->make_iterations(it_num);
	}

	// Work only after .intersect()
	/*std::tuple<double, double, double> get_real_precision(double needed_precision) const
	{
		const size_t it_num = get_iteration_number(needed_precision);
		const cube3d& cube = tree_.get_cube();
		const double depth = tree_.get_depth();

		const double xprec = (cube.x_up - cube.x_down) / std::pow(2.0, it_num * depth);
		const double yprec = (cube.y_up - cube.y_down) / std::pow(2.0, it_num * depth);
		const double zprec = (cube.z_up - cube.z_down) / std::pow(2.0, it_num * depth);

		return { xprec, yprec, zprec };
	}*/
};

