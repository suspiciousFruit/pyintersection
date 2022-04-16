#pragma once
#include <tuple>
#include <Python.h>
#include <arrayobject.h>

#include "../lib/Intersector/Intersector.h"

#include "TreeAdapterNumpy3d.h"
#include "TreeAdapterNumpy6d.h"

#include <iostream>

namespace npApi {
	bool checkArray3d(PyArrayObject* ndarray) {
		const auto ndim = ndarray->nd;

		// TODO Check the strides! 24 and 8!
		return ndim == 2 && ndarray->dimensions[1] == 5;
	}

	bool checkArray6d(PyArrayObject* ndarray) {
		const auto ndim = ndarray->nd;

		// TODO Check the strides! 24 and 8!
		return ndim == 2 && ndarray->dimensions[1] == 8;
	}

    template <typename CubeT>
    struct OutCube {
        double cid;
        CubeT cube;

        OutCube(double cid_, const CubeT& cube_)
			: cid(cid_), cube(cube_) {}
	};

    template <typename CubeT, typename Collision>
    PyArrayObject* toCubes(const std::vector<Collision>& collisions) {
        typedef OutCube<CubeT> Cube;
		const size_t cubesCount = collisions.size();
		const npy_intp fieldsNumber = sizeof(Cube) / sizeof(double);
		npy_intp dims[] = { cubesCount, fieldsNumber };

		PyArrayObject* nparray = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
		for (size_t i = 0; i < collisions.size(); ++i)
			((Cube*)nparray->data)[i] = { (double)i, collisions[i].cube };
		return nparray;
	}

	template <typename Collision>
	size_t getPointsCount(const std::vector<Collision>& collisions) {
		size_t res = 0;
		for (const auto& coll : collisions)
			res += coll.apoints.size() + coll.bpoints.size();
		return res;
	}

	template <typename PointT>
    struct OutPoint {
		double cid;
		double m;
		PointT ntpoint;

		OutPoint(double cid_, double m_, const PointT& p_)
			: cid(cid_), m(m_), ntpoint(p_) { }
	};

    template <typename PointT, typename Collision>
    PyArrayObject* toPoints(const std::vector<Collision>& collisions) {
        typedef OutPoint<PointT> Point;
		const size_t pointsCount = getPointsCount(collisions);
		const npy_intp fieldsNumber = sizeof(Point) / sizeof(double);
		npy_intp dims[] = { pointsCount, fieldsNumber };
		// PyObject* PyArray_SimpleNew(int nd, npy_intp* dims, int typenum);
		PyArrayObject* nparray = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
		size_t n = 0;
		for (size_t i = 0; i < collisions.size(); ++i) {
			const auto& coll = collisions[i];

			const auto& apoints = coll.apoints;
			for (size_t j = 0; j < apoints.size(); ++j, ++n)
				((Point*)nparray->data)[n] = { (double)i, 0, apoints[j].ntpoint() };

			const auto& bpoints = coll.bpoints;
			for (size_t j = 0; j < bpoints.size(); ++j, ++n)
				((Point*)nparray->data)[n] = { (double)i, 1, bpoints[j].ntpoint() };
		}
		return nparray;
	}

	PyArrayObject* toNdarray(const std::vector<double>& tols) {
		npy_intp dims[] = { tols.size() };
		PyArrayObject* ndarray = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

		for (size_t i = 0; i < tols.size(); ++i)
			((double*)ndarray->data)[i] = tols[i];

		return ndarray;
	}

	PyArrayObject* toNdarray(const double* data, size_t size) {
		npy_intp dims[] = { size };
		PyArrayObject* ndarray = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

		for (size_t i = 0; i < size; ++i)
			((double*)ndarray->data)[i] = data[i];

		return ndarray;
	}

	TreeAdapterNumpy6d makeAdapter6d(PyArrayObject* ndarray) {
		return TreeAdapterNumpy6d(ndarray);
	}

    TreeAdapterNumpy3d makeAdapter3d(PyArrayObject* ndarray) {
        return TreeAdapterNumpy3d(ndarray);
    }

    auto intersect3d(const TreeAdapterNumpy3d& a, const TreeAdapterNumpy3d& b,
					double tolerance, size_t treeDepth) {

		Intersector3d<TreeAdapterNumpy3d> intersector(treeDepth);
		const auto collisions = intersector.intersect(a, b, tolerance); // std::vector<collision3d<IteratorT>>
		const auto tols = intersector.get_real_precision(tolerance);
		PyArrayObject* npTols = npApi::toNdarray(tols);
		PyArrayObject* npPoints = npApi::toPoints<ntpoint3d>(collisions);
		PyArrayObject* npCubes = npApi::toCubes<cube3d>(collisions);
		return std::make_tuple(npCubes, npPoints, npTols);
	}

    auto intersect6d(const TreeAdapterNumpy6d& a, const TreeAdapterNumpy6d& b,
                    double tolerance, size_t treeDepth) {

        Intersector6d<TreeAdapterNumpy6d> intersector(treeDepth);
		const auto collisions = intersector.intersect(a, b, tolerance, tolerance); // std::vector<collision6d<IteratorT>>
		const auto tols = intersector.get_real_precision(tolerance);
		PyArrayObject* npTols = npApi::toNdarray(tols);
		PyArrayObject* npPoints = npApi::toPoints<ntpoint6d>(collisions);
		PyArrayObject* npCubes = npApi::toCubes<cube6d>(collisions);
		return std::make_tuple(npCubes, npPoints, npTols);
    }

    auto intersect6d_ronly(const TreeAdapterNumpy6d& a, const TreeAdapterNumpy6d& b,
                    double tolerance, size_t treeDepth) {

        Intersector6dRonly<TreeAdapterNumpy6d> intersector(treeDepth);
		const auto collisions = intersector.intersect(a, b, tolerance); // std::vector<collision6d<IteratorT>>
		const auto tols = intersector.get_real_precision(tolerance);
		PyArrayObject* npTols = npApi::toNdarray(tols);
		PyArrayObject* npPoints = npApi::toPoints<ntpoint6d>(collisions);
		PyArrayObject* npCubes = npApi::toCubes<cube6d>(collisions);
		return std::make_tuple(npCubes, npPoints, npTols);
    }

	PyArrayObject* get_boundary_cube3d(const TreeAdapterNumpy3d& a, const TreeAdapterNumpy3d& b) {
		const auto cube = ::get_boundary_cube3d(a, b);

		return npApi::toNdarray(cube.data, 6);
	}
}
