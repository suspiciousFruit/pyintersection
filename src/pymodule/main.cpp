#ifdef _DEBUG
	#undef _DEBUG
		#include <Python.h>
	#define _DEBUG
#else
	#include <Python.h>
#endif

#include <iostream>
#include <arrayobject.h>
#include "apinumpy.h"

PyObject* intersect3d(PyObject* self, PyObject* args)
{
	PyArrayObject* a_ndarray, *b_ndarray;
	double precision;

	PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray, &precision);

	if (npApi::checkArray3d(a_ndarray) && npApi::checkArray3d(b_ndarray)) {
		auto [cubes, points] = npApi::intersect3d(
			npApi::TreeAdapter(a_ndarray),
			npApi::TreeAdapter(b_ndarray),
			precision);
		return Py_BuildValue("OO", cubes, points);
	}

	return Py_None;
}

PyObject* test(PyObject* self, PyObject* arg) { return Py_None; }

static PyMethodDef pymodule_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{"test", (PyCFunction)test, METH_O, nullptr},
	{ "intersect3d", (PyCFunction)intersect3d, METH_VARARGS, nullptr },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef pymodule_module = {
	PyModuleDef_HEAD_INIT,
	"superfastcode", // Module name to use with Python import statements
	"intersections module", // Module description
	0,
	pymodule_methods // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_pymodule() {
	import_array();
	return PyModule_Create(&pymodule_module);
}


