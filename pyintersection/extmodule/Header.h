#pragma once

/*PyObject* test(PyObject* self, PyObject* arg)
{
	//PyObject* arr;
	//arr = PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	//return arr;

	PyArrayIterObject* iter = (PyArrayIterObject *)PyArray_IterNew(arg);
	if (iter == NULL) return Py_None;
	while (iter->index < iter->size) {
		// do something with the data at it->dataptr
		char* a = iter->dataptr;
		std::cout << *((double*)(a)) << ' ';
		PyArray_ITER_NEXT(iter);
	}
}
*/

void printList(PyObject* list)
{
	const Py_ssize_t listSize = PyList_Size(list);

	if (listSize > 0)
	{
		for (Py_ssize_t i = 0; i < listSize; ++i)
		{
			PyObject* const item = PyList_GetItem(list, i);
			const double value = PyFloat_AsDouble(item);
			std::cout << value << ':';
		}
	}
}

/*point3d point3d_from_list(PyObject* list)
{
	const Py_ssize_t list_size = PyList_Size(list);

	if (list_size == 3)
	{
		const double x = PyFloat_AsDouble(PyList_GetItem(list, 0));
		const double y = PyFloat_AsDouble(PyList_GetItem(list, 1));
		const double z = PyFloat_AsDouble(PyList_GetItem(list, 2));

		return point3d(x, y, z);
	}

	return point3d();
}

template <typename T>
T __no_conts(const T)
{
	return T;
}

#include <vector>
std::vector<point3d> points_from_list(PyObject* list)
{
	std::vector<point3d> array;
	const auto size = PyList_Size(list);

	for (decltype(__no_conts(size)) i = 0; i < size; ++i)
	{
		auto inner_list = PyList_GetItem(list, i);
		array.push_back(point3d_from_list(inner_list));
	}

	return array;
}

void print_numpy_array(PyArrayObject* ndarray)
{
	int size = ndarray->dimensions[0];
	const point3d* points = (const point3d*)ndarray->data;
	for (size_t i = 0; i < size / 3; ++i) std::cout << points[i] << '+';
}

PyObject* make_squares(PyObject* self, PyObject* num)
{
	double x = PyFloat_AsDouble(num);
	PyObject *my_list = PyList_New(0);

	for (double i = 0; i < x; i += 1.0)
		PyList_Append(my_list, Py_BuildValue("dd", i * i, i * i * i));

	return my_list;
}


*/


/*static PyObject * do_nothing(PyObject * self, PyObject *) { Py_RETURN_NONE; }

static PyMethodDef methods[] = {

 {"do_nothing", do_nothing, METH_NOARGS, "do nothing function"},
 {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
 PyModuleDef_HEAD_INIT, "do_nothing",
 "do nothing module with do nothing function", -1, methods
};

PyMODINIT_FUNC PyInit_do_nothing(void) {
	return PyModule_Create(&module);
}

PyMODINIT_FUNC
PyInit__test(void) {
	PyObject *mod = PyModule_Create(&module);

	// Добавляем глобальные переменные
	PyModule_AddObject(mod, "a", PyLong_FromLong(a)); // int
	PyModule_AddObject(mod, "b", PyFloat_FromDouble(b)); // double
	PyModule_AddObject(mod, "c", Py_BuildValue("b", c)); // char

	// Добавляем структуру

	// Завершение инициализации структуры
	if (PyType_Ready(&test_st_t_Type) < 0)
		return NULL;

	Py_INCREF(&test_st_t_Type);
	PyModule_AddObject(mod, "test_st_t", (PyObject *)&test_st_t_Type);

	return mod;
}*/