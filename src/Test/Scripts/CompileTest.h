#pragma once
#include <iostream>
#include "Intersector/Intersector3d.h"
#include "../../TreeAdapter/TreeAdapter.h"

std::vector<point3d> apoints3d = {
	{1, 1, 0.9},
	{1, 1, 0.5},
	{1, 1, 2.1},
	{1, 1, 1.9},
	{1, 1, 1.1},
	{1, 1, 0.8},
	{1, 1, 1.5},
	{1, 1, 0.2},

	{1, 4, 3.1},

	{0.1, 0.4, 0.7},
	{6.7, 6.7, 5.1},
	{6.7, 7, 5.1},
	{6.7, 6.7, 5},
	{6.7, 7, 5}
};

std::vector<point3d> bpoints3d = {
	{1, 1, 0},
	{1, 4, 3},
	{7, 7, 5}
};


std::vector<ntpoint3d> antpoins3d = {
	{1, 0, 1, 1, 1},
	{1, 0.5, 1, 1, 1}
};

std::vector<ntpoint3d> bntpoints3d = {
	{2, 0, 1, 1, 1},
	{3, 0.5, 1, 1, 1}
};


void test__Tree()
{
	cube3d cube(0, 1, 0, 1, 0, 1);
	// Test for npoint3d vector
	std::vector<npoint3d> vec_t1(4);
	Tree<typename VectorAdapter::const_iterator> tree_t1(cube, 1);
	TreeAdapter a(vec_t1);
	tree_t1.sieve_a(TreeAdapter(vec_t1));


	// Test for point3d vector
	std::vector<point3d> vec_t2(4);
	Tree<std::vector<point3d>::const_iterator> tree_t2(cube, 1);
	tree_t2.sieve_b(vec_t2);
}



void test__Intersector3d()
{
	std::vector<npoint3d> apoints = {
		{1, 0, 2, 2},
		{2, 0, 0, 0},
		{1, 0, 0, 0}
	};
	std::vector<npoint3d> bpoints = {
		{34, 0, 2, 1.9},
		{2, 0, 0, 0},
		{12, 0, 0, 0.1}
	};
	
	TreeAdapter a(apoints);
	TreeAdapter b(bpoints);
	Intersector3d<TreeAdapter<std::vector<npoint3d>>> inter(1);

	const auto colls = inter.intersect(a, b, 0.5);
	for (const auto& col : colls)
	{
		std::cout << "apoints" << std::endl;
		for (const auto p : col.apoints)
			std::cout << p.getnumber() << ' ' << *p << std::endl;
		std::cout << "bpoints" << std::endl;
		for (const auto p : col.bpoints)
			std::cout << p.getnumber() << ' ' << *p << std::endl;
	}

	TreeAdapter a1(antpoins3d);
	TreeAdapter b1(bntpoints3d);
	Intersector3d<TreeAdapter<std::vector<ntpoint3d>>> inter_ntpoints(1);

	inter_ntpoints.intersect(a1, b1, 0.4);
}

void test__boundary()
{
	

}

/*std::vector<npoint3d> a(4);
	std::vector<npoint3d> b(4);
	cube3d cube(0, 10, 0, 10, 0, 10);

	Intersector3d inter(
		base_points,
		other_points,
		cube);

	const auto colls = inter.make_iterations(4);
	for (const auto& col : colls)
		std::cout << col << std::endl;
	
	#include <algorithm>


bool __in__(const npoint_iterator item, std::vector<npoint_iterator>& items)
{
	const auto& p = item.getpoint();
	for (size_t i = 0; i < items.size(); ++i)
		if (*items[i] == p)
			return true;

	return false;
}

bool __equal__(std::vector<npoint_iterator>& a, std::vector<npoint_iterator>& b)
{
	if (a.size() != b.size())
		return false;

	for (const auto iter : a)
		if (!__in__(iter, b))
			return false;

	return true;
}

	*/

void test__null()
{
	
}

void test__runall()
{
	test__Tree();
	test__Intersector3d();
}