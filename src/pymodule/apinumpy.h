#pragma once
#include <iostream>
#include <Python.h>

#include "../lib/Intersector/Intersector3d.h"
#include "../lib/TreeAdapter/TreeAdapter.h"
#include <arrayobject.h>


class NumpyTreeAdapter
{
private:
	const ntpoint3d* data_;
	size_t size_;
public:
	typedef ntpoint_iterator const_iterator;

	NumpyTreeAdapter() : data_(nullptr), size_(0)
	{ }

	NumpyTreeAdapter(const ntpoint3d* data, size_t size) : data_(data), size_(size)
	{ }

	ntpoint_iterator begin() const
	{
		return ntpoint_iterator(data_);
	}

	ntpoint_iterator end() const
	{
		return ntpoint_iterator(data_ + size_);
	}
};

bool check_np_array3d(PyArrayObject* ndarray)
{
	const auto ndim = ndarray->nd;

	// TODO Check the strides! 24 and 8!
	return ndim == 2 && ndarray->dimensions[1] == 5;
}

auto __intersect(const ntpoint3d* a, size_t asize, const ntpoint3d* b, size_t bsize)
{
	NumpyTreeAdapter apoints(a, asize), bpoints(b, bsize);

	Intersector3d<NumpyTreeAdapter> intersector(2);
	const auto res = intersector.intersect(apoints, bpoints, 0.5);

	for (const auto& r : res)
		std::cout << r << std::endl;

	return res;
}

// struct NumpyCollisions3d
// {
// 	PyArrayObject* cubes;
// 	PyArrayObject* points;
// };


// template <typename T>
// size_t f(const T& cols)
// {
// 	size_t counter;
// 	for (const auto& col : cols)
// 	{
// 		for (const auto& p : col.apoints) ++counter;
// 		for (const auto& p : col.bpoints) ++counter;
// 	}

// 	return counter;
// }

// template <typename T>
// NumpyCollisions3d convert_collisions3d(const T& collisions)
// {
// 	int cubes_dims[] = { collisions.size(), 7 };
// 	int points_dims[] = { f(collisions), 7 };
// 	PyArrayObject* cubes = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
// 	PyArrayObject* points = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);


// 	for (size_t i = 0; i < collisions.size(); ++i)
// 	{
// 		const auto& col = collisions[i];

// 	}
// }

PyArrayObject* intersect(PyArrayObject* a, PyArrayObject* b)
{
	const ntpoint3d* apoints = (const ntpoint3d*)a->data;
	const size_t a_size = a->dimensions[0];
	const ntpoint3d* bpoints = (const ntpoint3d*)b->data;
	const size_t b_size = b->dimensions[0];

	const auto res = __intersect(apoints, a_size, bpoints, b_size);

	//int dims[] = { res.size(), 6 };
	npy_intp dims[] = { res.size(), 6 };
	PyArrayObject* nparray = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	
	for (size_t i = 0; i < res.size(); ++i)
	{
		cube3d& cube = ((cube3d*)nparray->data)[i];
		cube = res[i].cube;
	}

	return nparray;
}

