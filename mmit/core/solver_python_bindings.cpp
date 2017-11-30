#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <iostream>
#include "solver.h"


static PyObject *
optimal_costs(PyObject *self, PyObject *args){
    PyArrayObject *lower_vec, *upper_vec; //borrowed
    double margin;
    int loss;

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!O!di",
                         &PyArray_Type, &lower_vec,
                         &PyArray_Type, &upper_vec,
                         &margin,
                         &loss
    )){
        return NULL;
    }

    // Check the data types of the numpy arrays
    if(PyArray_TYPE(lower_vec)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "lower_vec must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(upper_vec)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "upper_vec must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy arrays
    npy_intp n_lower = PyArray_DIM(lower_vec,0);
    npy_intp n_upper = PyArray_DIM(upper_vec,0);
    if(n_lower != n_upper){
        PyErr_SetString(PyExc_ValueError,
                        "lower_vec and upper_vec must be same length");
        return NULL;
    }

    // Check the loss function type
    if(!(loss == 0 ||loss == 1))
    {
        PyErr_SetString(PyExc_ValueError,
                        "loss must be 0 (hinge) or 1 (squared hinge)");
        return NULL;
    }

    // Access the array data
    double *lower_vecA = (double*)PyArray_DATA(lower_vec);
    double *upper_vecA = (double*)PyArray_DATA(upper_vec);

    // Initialize arrays for return
    PyObject *moves_vec = PyArray_SimpleNew(1, &n_lower, PyArray_INT);
    int *moves_vecA = (int*)PyArray_DATA(moves_vec);
    PyObject *pred_vec = PyArray_SimpleNew(1, &n_lower, PyArray_DOUBLE);
    double *pred_vecA = (double*)PyArray_DATA(pred_vec);
    PyObject *cost_vec = PyArray_SimpleNew(1, &n_lower, PyArray_DOUBLE);
    double *cost_vecA = (double*)PyArray_DATA(cost_vec);

    int status = compute_optimal_costs(n_lower, lower_vecA, upper_vecA, margin, loss, moves_vecA, pred_vecA, cost_vecA);
    if(status != 0){
        return NULL;
    }

    return Py_BuildValue("N,N,N",
                         moves_vec,
                         pred_vec,
                         cost_vec);
}

static PyMethodDef Methods[] = {
        {"compute_optimal_costs", optimal_costs, METH_VARARGS,
         "Compute the optimal cost solution for each split point."},
        {NULL, NULL, 0, NULL}
};



#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef solver_module = {
    PyModuleDef_HEAD_INIT,
    "solver",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_solver() {
    import_array();
    return PyModule_Create(&solver_module);
};

#else


PyMODINIT_FUNC
initsolver
        (void){
    (void)Py_InitModule("solver",Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}

// PyMODINIT_FUNC
// initsolver
//         (void){
//     (void)Py_InitModule("solver", Methods);
//     import_array();//necessary from numpy otherwise we crash with segfault
// }

#endif
