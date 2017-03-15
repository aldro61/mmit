/*
MMIT: Max Margin Interval Trees
Copyright (C) 2017 Toby Dylan Hocking, Alexandre Drouin
This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/
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
        moves_vec[i] = 0;

        // Add the lower bound
        if(lower_vec[i] > -INFINITY)
            moves_vec[i] += function.insert_point(lower_vec[i], false);

        // Add the upper bound
        if(upper_vec[i] < INFINITY)
            moves_vec[i] += function.insert_point(upper_vec[i], true);

        pred_vec[i] = function.get_minimum_position();
        cost_vec[i] = function.get_minimum_value();
    }

    return 0;
}
