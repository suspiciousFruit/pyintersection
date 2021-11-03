#ifdef _DEBUG
	#undef _DEBUG
		#include <Python.h>
	#define _DEBUG
#else
	#include <Python.h>
#endif

#include <iostream>
#include "C:\Users\Xiaomi\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\numpy\core\include\numpy\arrayobject.h"
#include "numpy_api.h"


PyObject* test_numpy(PyObject* self, PyObject* args)
{
	PyArrayObject* a_ndarray, *b_ndarray;

	PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_ndarray, &PyArray_Type, &b_ndarray);

	if (check_np_array3d(a_ndarray) && check_np_array3d(b_ndarray))
		return (PyObject*)intersect(a_ndarray, b_ndarray);


	return Py_None;
}



PyObject* test(PyObject* self, PyObject* tuple_with_lists)
{
	const auto a_list = PyTuple_GetItem(tuple_with_lists, 0);
	const auto b_list = PyTuple_GetItem(tuple_with_lists, 1);
	//const auto a_points = points_from_list(a_list);
	//const auto b_points = points_from_list(b_list);
	/*for (const auto& a : a_points)
		std::cout << a << ' ';
	std::cout << std::endl;
	for (const auto& a : b_points)
		std::cout << a << ' ';*/
	//Intersector3d<std::vector<point3d>> intersector(2);
	//const auto& res = intersector.intersect(a_points, b_points, 0.2);
	//for (const auto& r : res)
	//	std::cout << r << std::endl;

	return Py_None;
}

static PyMethodDef pymodule_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{"test", (PyCFunction)test, METH_O, nullptr},
	{ "test_numpy", (PyCFunction)test_numpy, METH_VARARGS, nullptr },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef pymodule_module = {
	PyModuleDef_HEAD_INIT,
	"superfastcode",                        // Module name to use with Python import statements
	"intersections module",  // Module description
	0,
	pymodule_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_pymodule() {
	import_array();
	return PyModule_Create(&pymodule_module);
}


