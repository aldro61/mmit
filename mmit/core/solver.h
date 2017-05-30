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
#ifndef MMIT_SOLVER_H
#define MMIT_SOLVER_H

#include "piecewise_function.h"

int compute_optimal_costs(
//inputs
        int n_data,
        double *lower_vec, // array[n_data] of output lower limits (can be -INFINITY)
        double *upper_vec, // array[n_data] of output upper limits (can be INFINITY)
        double margin,
        int loss, //0=linear hinge, 1=squared hinge
// outputs
        int *moves_vec, //array[n_data] of number of pointer moves
        double *pred_vec, //array[n_data] of optimal predicted values
        double *cost_vec); // array[n_data] of optimal cost

#endif //MMIT_SOLVER_H
