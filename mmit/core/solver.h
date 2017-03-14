//
// Created by Alexandre Drouin on 2017-03-09.
//

#ifndef MMIT_SOLVER_H
#define MMIT_SOLVER_H

#include "piecewise_function.h"

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
        double *cost_vec); // array[n_data] of optimal cost

#endif //MMIT_SOLVER_H
