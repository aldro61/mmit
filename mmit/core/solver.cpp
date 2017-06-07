/* Dear emacs, please set these variables: -*- c-basic-offset: 4 -*-
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
#include "coefficients.h"
#include "double_utils.h"
#include "function_types.h"
#include "solver.h"
#include "piecewise_function.h"


inline Coefficients get_coefficients(const double y, const double margin, const bool is_upper_limit,
                                     const FunctionType loss_type){
    double s = (is_upper_limit ? 1. : -1.);

    Coefficients F, Fs;
    if(loss_type == linear_hinge){
        F = Coefficients(0, s, margin - s * y);
    }
    else if(loss_type == squared_hinge){
        F = Coefficients(1, 2 * margin * s - 2 * y, -2 * margin * s * y + y * y + margin * margin);
    }

    return F;
}

inline double get_breakpoint(const double y, const double margin, const bool is_upper_limit){
    return y - (is_upper_limit ? 1. : -1.) * margin;
}

// return int is an error status code
int compute_optimal_costs(
//inputs
        int n_data,
        double *lower_vec, // array[n_data] of output lower limits (can be -INFINITY)
        double *upper_vec, // array[n_data] of output upper limits (can be INFINITY)
        double margin,
        int loss, //0=linear_hinge, 1=squared_hinge
// outputs
        int *moves_vec, //array[n_data] of number of pointer moves
        double *pred_vec, //array[n_data] of optimal predicted values
        double *cost_vec // array[n_data] of optimal cost
) {
    PiecewiseFunction function;
    FunctionType loss_type = (loss == 0 ? linear_hinge : squared_hinge);

    // Compute the optimal solution for each interval
    for(int i = 0; i < n_data; i++){

        moves_vec[i] = 0;

        // Uncensored output
        if(equal(lower_vec[i], upper_vec[i])) {
            if (!isinf(lower_vec[i])){
                Coefficients F1 = get_coefficients(lower_vec[i], 0., false, loss_type);
                Coefficients F2 = get_coefficients(upper_vec[i], 0., true, loss_type);
                moves_vec[i] += function.insert_point(lower_vec[i], F1, false);
                moves_vec[i] += function.insert_point(upper_vec[i], F2, true);
            }
        }

        // Censored output
        else {
            // Add the lower bound
            if (!isinf(lower_vec[i])){
                Coefficients F = get_coefficients(lower_vec[i], margin, false, loss_type);
                double b = get_breakpoint(lower_vec[i], margin, false);
                moves_vec[i] += function.insert_point(b, F, false);
            }

            // Add the upper bound
            if (!isinf(upper_vec[i])){
                Coefficients F = get_coefficients(upper_vec[i], margin, true, loss_type);
                double b = get_breakpoint(upper_vec[i], margin, true);
                moves_vec[i] += function.insert_point(b, F, true);
            }
        }

        pred_vec[i] = function.get_minimum_position();
        cost_vec[i] = function.get_minimum_value();
    }
    return 0;
}
