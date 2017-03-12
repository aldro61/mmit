#ifndef MMIT_PIECEWISE_FUNCTION_H
#define MMIT_PIECEWISE_FUNCTION_H

#include <map>
#include <set>

#include "coefficients.h"

using namespace std;

enum FunctionType{
    hinge,
    squared_hinge
};

class PiecewiseFunction {

private:
    // Function info
    FunctionType function_type;

    // Breakpoint and their coefficients
    map<double, Coefficients> breakpoint_coefficients;

    // Minimum solution
    Coefficients min_coefficients;
    map<double, Coefficients>::iterator min_ptr = breakpoint_coefficients.end();  // Always on the right of the minimum

    // Variables that might get moved into a solver class
    double margin;

    // Minimum pointer functions
    void move_minimum_pointer_left();
    void move_minimum_pointer_right();

    // Utility functions
    double get_breakpoint_position(double y, bool is_upper_bound);

public:
    PiecewiseFunction(double margin){
        this->margin = margin;
    }

    // Point insertion
    void insert_point(double y, bool is_upper_bound);
    void insert_points();

    // Minimum pointer functions
    double get_minimum_position();
    double get_minimum_value();
};


#endif //MMIT_PIECEWISE_FUNCTION_H
