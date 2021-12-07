#pragma once
#include <iostream>
#include <Python.h>
#include <tuple>

#include "../lib/Intersector/Intersector3d.h"
#include "../lib/TreeAdapter/TreeAdapter.h"
#include <arrayobject.h>


// auto __intersect(const ntpoint3d* a, size_t asize, const ntpoint3d* b, size_t bsize)
// {
// 	NumpyTreeAdapter apoints(a, asize), bpoints(b, bsize);

// 	Intersector3d<NumpyTreeAdapter> intersector(2);
// 	const auto res = intersector.intersect(apoints, bpoints, 0.5);

// 	for (const auto& r : res) {
// 		// std::cout << r << std::endl;
// 		const auto pts = r.apoints;
// 		for (const auto& p : pts)
// 			std::cout << p.getpoint() << std::endl;
// 	}
// 	// res === std::vector<collision3d<IteratorT>>
// 	return res;
// }

template <typename T>
void dprint(const T& a) {
	std::cout << "----------------------------  debug ----------------------------\n";
	for (const auto& b : a)
		std::cout << b << '\n';
	std::cout << "---------------------------- /debug ----------------------------\n";
}

namespace npApi {
	class TreeAdapter
	{
	private:
		const ntpoint3d* data_;
		size_t size_;
	public:
		typedef ntpoint_iterator const_iterator;

		TreeAdapter() : data_(nullptr), size_(0)
		{ }

		TreeAdapter(PyArrayObject* ndarray)
		: data_((const ntpoint3d*)ndarray->data), size_(ndarray->dimensions[0]) { }

		TreeAdapter(const ntpoint3d* data, size_t size) : data_(data), size_(size)
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

	size_t getPointsCount(const std::vector<collision3d<ntpoint_iterator>>& collisions) {
		size_t res = 0;
		for (const auto& col : collisions)
			res += col.apoints.size() + col.bpoints.size();
		return res;
	}

	struct IPoint {
		double cid;
		double m;
		double n;
		double t;
		point3d p;

		IPoint(double cid_, double m_, double n_, double t_, const point3d& p_)
			: cid(cid_), m(m_), n(n_), t(t_), p(p_) {}
	};

	PyArrayObject* toPoints(const std::vector<collision3d<ntpoint_iterator>>& collisions) {
		const size_t pointsCount = getPointsCount(collisions);
		const npy_intp fieldsNumber = 7;
		npy_intp dims[] = { pointsCount, fieldsNumber };
		// PyObject* PyArray_SimpleNew(int nd, npy_intp* dims, int typenum);
		PyArrayObject* nparray = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
		size_t n = 0;
		for (size_t i = 0; i < collisions.size(); ++i) {
			const auto& col = collisions[i];

			const auto& apoints = col.apoints;
			for (size_t j = 0; j < apoints.size(); ++j, ++n) {
				const auto it = apoints[j];
				((IPoint*)nparray->data)[n] = { i, 0, it.getnumber(), it.gett(), it.getpoint() };
			}
			const auto& bpoints = col.bpoints;
			for (size_t j = 0; j < bpoints.size(); ++j, ++n) {
				const auto it = bpoints[j];
				((IPoint*)nparray->data)[n] = { i, 1, it.getnumber(), it.gett(), it.getpoint() };
			}
		}
		return nparray;
	}

	struct ICube {
		double cid;
		cube3d cube;

		ICube(double cid_, const cube3d& cube_)
			: cid(cid_), cube(cube_) {}
	};

	PyArrayObject* toCubes(const std::vector<collision3d<ntpoint_iterator>>& collisions) {
		const size_t cubesCount = collisions.size();
		const npy_intp fieldsNumber = 7;
		npy_intp dims[] = { cubesCount, fieldsNumber };

		PyArrayObject* nparray = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
		for (size_t i = 0; i < collisions.size(); ++i)
			((ICube*)nparray->data)[i] = { i, collisions[i].cube };
		return nparray;
	}

	auto intersect3d(const npApi::TreeAdapter& a, const npApi::TreeAdapter& b,
					double precision = 0.5, size_t treeDepth = 2) {
		Intersector3d<TreeAdapter> intersector(treeDepth);
		const auto collisions = intersector.intersect(a, b, precision); // std::vector<collision3d<IteratorT>>
		PyArrayObject* npPoints = npApi::toPoints(collisions);
		PyArrayObject* npCubes = npApi::toCubes(collisions);
		return std::make_tuple(npCubes, npPoints);
	}

	bool checkArray3d(PyArrayObject* ndarray) {
		const auto ndim = ndarray->nd;

		// TODO Check the strides! 24 and 8!
		return ndim == 2 && ndarray->dimensions[1] == 5;
	}
}
