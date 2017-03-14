//
// Created by Alexandre Drouin on 2017-03-09.
//
#include <cmath>
#include <iostream>
#include "solver.h"
#include "piecewise_function.h"

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
    PiecewiseFunction function(margin, loss == 0 ? hinge : squared_hinge);

    // Compute the optimal solution for each interval
    for(int i = 0; i < n_data; i++){
        pred_vec[i] = function.get_minimum_position();
        cost_vec[i] = function.get_minimum_value();

        moves_vec[i] = 0;

        // Add the lower bound
        if(lower_vec[i] > -INFINITY)
            moves_vec[i] += function.insert_point(lower_vec[i], false);

        // Add the upper bound
        if(upper_vec[i] < INFINITY)
            moves_vec[i] += function.insert_point(upper_vec[i], true);
    }

    return 0;
}
