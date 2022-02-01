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

PyObject* intersect3d(PyObject* self, PyObject* args)
{
	PyArrayObject* a_ndarray, *b_ndarray;
	double precision;

	PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray, &precision);

	if (npApi::checkArray3d(a_ndarray) && npApi::checkArray3d(b_ndarray)) {
		auto [cubes, points] = npApi::intersect3d(
			npApi::makeAdapter3d(a_ndarray),
			npApi::makeAdapter3d(b_ndarray),
			precision);
		return Py_BuildValue("OO", cubes, points);
	}

	return Py_None;
}

PyObject* intersect6d(PyObject* self, PyObject* args) {
	PyArrayObject* a_ndarray, *b_ndarray;
	double tolerance;

	PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray, &tolerance);

	if (npApi::checkArray6d(a_ndarray) && npApi::checkArray6d(b_ndarray)) {
		auto [cubes, points] = npApi::intersect6d(
			npApi::makeAdapter6d(a_ndarray),
			npApi::makeAdapter6d(b_ndarray),
			tolerance);
		return Py_BuildValue("OO", cubes, points);
	}
	return Py_None;
}

PyObject* test(PyObject* self, PyObject* arg) { return Py_None; }

static PyMethodDef pyintersection_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{"test", (PyCFunction)test, METH_O, nullptr},
	{ "intersect3d", (PyCFunction)intersect3d, METH_VARARGS, nullptr },
	{ "intersect6d", (PyCFunction)intersect6d, METH_VARARGS, nullptr },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef pyintersection_module = {
	PyModuleDef_HEAD_INIT,
	"pyintersection", // Module name to use with Python import statements
	"intersections module", // Module description
	0,
	pyintersection_methods // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_pyintersection() {
	import_array();
	return PyModule_Create(&pyintersection_module);
}


