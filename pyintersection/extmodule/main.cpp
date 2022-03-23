#ifdef _DEBUG
	#undef _DEBUG
		#include <Python.h>
	#define _DEBUG
#else
	#include <Python.h>
#endif

#include <iostream>
#include <arrayobject.h>
#include "api.h"

// PyArrayObject*, PyArrayObject*, atolerance, tree_depth
// tree_iteration == 0 then tree calculate iterations automatic

PyObject* intersect3d(PyObject* self, PyObject* args)
{
	PyArrayObject* a_ndarray, *b_ndarray;
	double atolerance;
	size_t tree_depth;

	PyArg_ParseTuple(args, "O!O!dI",
		&PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray, &atolerance, &tree_depth);

	if (npApi::checkArray3d(a_ndarray) && npApi::checkArray3d(b_ndarray) && tree_depth > 0) {
		auto [cubes, points, tols] = npApi::intersect3d(
			npApi::makeAdapter3d(a_ndarray),
			npApi::makeAdapter3d(b_ndarray),
			atolerance, tree_depth);
		return Py_BuildValue("OOO", cubes, points, tols);
	}

	return Py_None;
}

PyObject* intersect6d(PyObject* self, PyObject* args) {
	PyArrayObject* a_ndarray, *b_ndarray;
	double atolerance;
	size_t tree_depth;

	PyArg_ParseTuple(args, "O!O!dI",
		&PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray, &atolerance, &tree_depth);

	if (npApi::checkArray6d(a_ndarray) && npApi::checkArray6d(b_ndarray)  && tree_depth > 0) {
		auto [cubes, points, tols] = npApi::intersect6d(
			npApi::makeAdapter6d(a_ndarray),
			npApi::makeAdapter6d(b_ndarray),
			atolerance, tree_depth);
		return Py_BuildValue("OOO", cubes, points, tols);
	}
	return Py_None;
}

PyObject* get_boundary_cube3d(PyObject* self, PyObject* args) {
	PyArrayObject* a_ndarray, *b_ndarray;

	PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray);

	if (npApi::checkArray3d(a_ndarray) && npApi::checkArray3d(b_ndarray)) {
		auto cube = npApi::get_boundary_cube3d(npApi::makeAdapter3d(a_ndarray),
			npApi::makeAdapter3d(b_ndarray));

		return Py_BuildValue("O", cube); //
	}
	return Py_None;
}

static PyMethodDef extmodule_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{ "__intersect3d", (PyCFunction)intersect3d, METH_VARARGS, nullptr },
	{ "__intersect6d", (PyCFunction)intersect6d, METH_VARARGS, nullptr },
	{ "__get_boundary_cube3d", (PyCFunction)get_boundary_cube3d, METH_VARARGS, nullptr },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef extmodule_module = {
	PyModuleDef_HEAD_INIT,
	"extmodule", // Module name to use with Python import statements
	"intersections module", // Module description
	0,
	extmodule_methods // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_extmodule() {
	import_array();
	return PyModule_Create(&extmodule_module);
}

