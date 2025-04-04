// Copyright 2025 Intel Corporation
// SPDX: Apache-2.0

// fast_df.c
// compile with gcc -O3 -fopenmp -shared -std=c99 -fPIC -I/usr/include/python3.8 fast_df.c -o fast_df.so
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pandas/headers.h>
#include <omp.h>

typedef struct
{
    char *name;
    int dtype;
    void *data;
} ColumnMeta;

static PyObject *list_of_dicts_to_dataframe(PyObject *self, PyObject *args)
{
    PyObject *input_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input_list))
        return NULL;

    const Py_ssize_t num_rows = PyList_Size(input_list);
    if (num_rows == 0)
        return Py_BuildValue("{}");

    // Get first row to determine schema
    PyObject *first_row = PyList_GetItem(input_list, 0);
    if (!PyDict_Check(first_row))
    {
        PyErr_SetString(PyExc_TypeError, "Input must be list of dictionaries");
        return NULL;
    }

    // Extract column metadata
    PyObject *keys = PyDict_Keys(first_row);
    const Py_ssize_t num_cols = PyList_Size(keys);
    ColumnMeta *columns = malloc(num_cols * sizeof(ColumnMeta));
    npy_intp dims[1] = {num_rows};

#pragma omp parallel for schedule(static)
    for (Py_ssize_t col = 0; col < num_cols; col++)
    {
        PyObject *key = PyList_GetItem(keys, col);
        const char *name = PyUnicode_AsUTF8(key);
        PyObject *sample_val = PyDict_GetItem(first_row, key);

        columns[col].name = strdup(name);

        if (PyLong_Check(sample_val))
        {
            columns[col].dtype = NPY_INT64;
            columns[col].data = malloc(num_rows * sizeof(int64_t));
        }
        else if (PyFloat_Check(sample_val))
        {
            columns[col].dtype = NPY_DOUBLE;
            columns[col].data = malloc(num_rows * sizeof(double));
        }
        else
        {
            columns[col].dtype = NPY_OBJECT;
            columns[col].data = malloc(num_rows * sizeof(PyObject *));
        }
    }

// Parallel data extraction
#pragma omp parallel for schedule(dynamic)
    for (Py_ssize_t row = 0; row < num_rows; row++)
    {
        PyObject *row_dict = PyList_GetItem(input_list, row);

        for (Py_ssize_t col = 0; col < num_cols; col++)
        {
            PyObject *key = PyList_GetItem(keys, col);
            PyObject *val = PyDict_GetItem(row_dict, key);
            void *data_ptr = (char *)columns[col].data + row * dtype_size(columns[col].dtype);

            switch (columns[col].dtype)
            {
            case NPY_INT64:
                *(int64_t *)data_ptr = PyLong_AsLongLong(val);
                break;
            case NPY_DOUBLE:
                *(double *)data_ptr = PyFloat_AsDouble(val);
                break;
            case NPY_OBJECT:
                *(PyObject **)data_ptr = val;
                Py_INCREF(val);
                break;
            }
        }
    }

    // Build DataFrame
    PyObject *df_dict = PyDict_New();
    for (Py_ssize_t col = 0; col < num_cols; col++)
    {
        PyArrayObject *arr = PyArray_SimpleNewFromData(1, dims,
                                                       columns[col].dtype, columns[col].data);
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);
        PyDict_SetItemString(df_dict, columns[col].name, (PyObject *)arr);
        Py_DECREF(arr);
        free(columns[col].name);
    }

    PyObject *pd = PyImport_ImportModule("pandas");
    PyObject *df = PyObject_CallMethod(pd, "DataFrame", "O", df_dict);

    // Cleanup
    free(columns);
    Py_DECREF(df_dict);
    Py_DECREF(pd);

    return df;
}

static PyMethodDef methods[] = {
    {"list_of_dicts_to_dataframe", list_of_dicts_to_dataframe, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "fast_df",
    NULL,
    -1,
    methods};

PyMODINIT_FUNC PyInit_fast_df(void)
{
    import_array();
    import_pandas();
    return PyModule_Create(&module);
}