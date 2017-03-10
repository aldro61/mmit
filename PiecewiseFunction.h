//
// Created by Alexandre Drouin on 2017-03-09.
//
#ifndef MMIT_PIECEWISE_FUNCTION_H
#define MMIT_PIECEWISE_FUNCTION_H

#include <map>
#include <set>

using namespace std;

struct Coefficients{
    double a = 0;
    double b = 0;
    double c = 0;
};

enum FunctionType{
    hinge,
    squared_hinge
};

class PiecewiseFunction {

private:
    // Function info
    FunctionType function_type;

    // Breakpoint and their coefficients
    set<double> breakpoints;
    map<double, Coefficients> breakpoint_coefficients;

    // Minimum solution
    Coefficients min_coefficients;
    map<double, Coefficients>::iterator min_ptr = breakpoint_coefficients.end();  // Always on the right of the minimum
    double min_val;
    double min_pos;

    // Variables that might get moved into a solver class
    double margin;

    // Minimum pointer movement functions
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
};


#endif //MMIT_PIECEWISE_FUNCTION_H
