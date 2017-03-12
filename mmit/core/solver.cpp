//
// Created by Alexandre Drouin on 2017-03-09.
//
#include <iostream>
#include "solver.h"

// return int is an error status code
int compute_optimal_costs(
//inputs
        int n_data,
        double *feature_vec, // array[n_data] of sorted input features
        double *lower_vec, // array[n_data] of output lower limits (can be -INFINITY)
        double *upper_vec, // array[n_data] of output upper limits (can be INFINITY)
        double margin,
        int loss, //0=hinge, 1=squared hinge
// outputs
        int *moves_vec, //array[n_data] of number of pointer moves
        double *pred_vec, //array[n_data] of optimal predicted values
        double *cost_vec // array[n_data] of optimal cost
) {
    std::cout << "It works!" << std::endl;
    return 0;
}