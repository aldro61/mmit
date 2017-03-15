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

#include "main.h"
#include "solver.h"

int main() {

    int n_data = 11;
    double feature_vec[] = {0, 1, 1, 3, 4, 4, 6, 7, 8, 9, 10};
    double upper_vec[] = {0, 1, 2, 3, 4, 5, 6, INFINITY, 8, 9, INFINITY};
    double lower_vec[] = {-1, -1, -2, -INFINITY, -4, -5, -6, -7, -INFINITY, -2, -3};

    double margin = 0.3;
    int loss = 1;

    int *moves_vec = new int[n_data];
    double *pred_vec = new double[n_data];
    double *cost_vec = new double[n_data];


    compute_optimal_costs(n_data, feature_vec, lower_vec, upper_vec, margin, loss, moves_vec, pred_vec, cost_vec);

    for(int i = 0; i < n_data; i++){
        std::cout << "Cost[" << i << "] = " << cost_vec[i] << std::endl;
    }

    for(int i = 0; i < n_data; i++){
        std::cout << "Pred[" << i << "] = " << pred_vec[i] << std::endl;
    }

    return 0;
}

