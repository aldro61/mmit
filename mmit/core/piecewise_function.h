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
#ifndef MMIT_PIECEWISE_FUNCTION_H
#define MMIT_PIECEWISE_FUNCTION_H

#include <map>
#include <set>

#include "coefficients.h"
#include "double_utils.h"

typedef std::map<double, Coefficients, DoubleComparatorLess> breakpoint_list_t;
typedef std::pair<double, Coefficients> breakpoint_t;

enum FunctionType{
    linear_hinge,
    squared_hinge,
    no_ptr_move
};

class PiecewiseFunction {

private:
    // Function parameters
    FunctionType function_type;
    double margin;

    // Breakpoint and their coefficients
    breakpoint_list_t breakpoint_coefficients;

    // Minimum solution
    Coefficients min_coefficients;
    breakpoint_list_t::iterator min_ptr;  // Always on the right of the minimum

    // Utility vars + functions
    void construct(double margin, FunctionType loss, bool verbose){this->margin = margin; this->function_type = loss; this->verbose = verbose; this->min_ptr = breakpoint_coefficients.end();}
    inline double get_breakpoint_position(breakpoint_list_t::iterator b_ptr);
    inline bool is_end(breakpoint_list_t::iterator b_ptr);
    inline bool is_begin(breakpoint_list_t::iterator b_ptr);
    bool verbose;

public:
    PiecewiseFunction(double margin, FunctionType loss, bool verbose){
        construct(margin, loss, verbose);
    }

    PiecewiseFunction(double margin, FunctionType loss){
        construct(margin, loss, false);
    }

    // Point insertion
    int insert_point(double y, bool is_upper_bound);

    // Minimum pointer functions
    double get_minimum_position();
    double get_minimum_value();
};


#endif //MMIT_PIECEWISE_FUNCTION_H
