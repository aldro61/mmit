/* -*- compile-command: "cd ../.. && python setup.py build" -*- */

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <iostream>
#include "modelSelection.h"

static PyObject *
modelSelection_interface(PyObject *self, PyObject *args){
    PyArrayObject *loss_vec, *complexity_vec; //borrowed
    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &loss_vec,
                         &PyArray_Type, &complexity_vec
    )){
        return NULL;
    }
    // Check the data types of the numpy arrays
    if(PyArray_TYPE(loss_vec)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "loss_vec must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(complexity_vec)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "complexity_vec must be numpy.ndarray type double");
        return NULL;
    }
    // Check the dimensions of the numpy arrays
    npy_intp n_loss = PyArray_DIM(loss_vec,0);
    npy_intp n_complexity = PyArray_DIM(complexity_vec,0);
    if(n_loss != n_complexity){
        PyErr_SetString(PyExc_ValueError,
                        "loss_vec and complexity_vec must be same length");
        return NULL;
    }
    if(n_loss < 2){
        PyErr_SetString(PyExc_ValueError,
                        "len(loss_vec) must be at least 2");
        return NULL;
    }      
    // Access the array data
    double *loss_vecA = (double*)PyArray_DATA(loss_vec);
    double *complexity_vecA = (double*)PyArray_DATA(complexity_vec);
    // Initialize arrays for return
    PyObject *optimal_after_vec = PyArray_SimpleNew(1, &n_loss, PyArray_INT);
    int *optimal_after_vecA = (int*)PyArray_DATA(optimal_after_vec);
    PyObject *lambda_vec = PyArray_SimpleNew(1, &n_loss, PyArray_DOUBLE);
    double *lambda_vecA = (double*)PyArray_DATA(lambda_vec);
    int status = modelSelection
      (loss_vecA, complexity_vecA, n_loss,
       optimal_after_vecA, lambda_vecA);
    if(status == ERROR_LOSS_NOT_DECREASING){
      PyErr_SetString(PyExc_ValueError, "loss not decreasing");
    }
    if(status == ERROR_COMPLEXITY_NOT_INCREASING){
      PyErr_SetString(PyExc_ValueError, "complexity not increasing");
    }
    if(status != 0){
      return NULL;
    }
    return Py_BuildValue("N,N", optimal_after_vec, lambda_vec);
}

static PyMethodDef Methods[] = {
        {"modelSelection_interface", modelSelection_interface, METH_VARARGS,
         "modelSelection_interface(loss, complexity) solves min_i loss[i] + lambda*complexity[i] for all lambda"},
        {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initmodelSelection
        (void){
    (void)Py_InitModule("modelSelection",Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}
