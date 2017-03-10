//
// Created by Alexandre Drouin on 2017-03-09.
//
#include <cmath>
#include <iostream>
#include <utility>

#include "PiecewiseFunction.h"

double PiecewiseFunction::get_breakpoint_position(double y, bool is_upper_bound) {
    return y + (is_upper_bound ? -1 : 1) * this->margin;
}

void PiecewiseFunction::insert_point(double y, bool is_upper_bound) {

    // New breakpoint's position
    double breakpoint_position = this->get_breakpoint_position(y, is_upper_bound);

    if(this->min_ptr == this->breakpoint_coefficients.end()) {
        cout << "The function is empty" << endl;

        // Compute the new breakpoint's coefficients
        Coefficients new_breakpoint_coefficients;
        new_breakpoint_coefficients.a = 0;
        new_breakpoint_coefficients.b = 1;
        new_breakpoint_coefficients.c = -breakpoint_position;

        // Insert the new breakpoint
        pair<map<double, Coefficients>::iterator, bool> insert = this->breakpoint_coefficients.insert(pair<double, Coefficients>(breakpoint_position, new_breakpoint_coefficients));
        auto breakpoint_ptr = insert.first;
        auto breakpoint_already_exists = insert.second;

        if (is_upper_bound) {
            this->min_ptr = breakpoint_ptr;
            this->min_pos = -INFINITY;
        } else {
            this->min_pos = INFINITY;
        }
        this->min_val = INFINITY;
    }
    else{
        // Compute the new breakpoint's coefficients
        Coefficients new_breakpoint_coefficients;
        new_breakpoint_coefficients.a = 0;
        new_breakpoint_coefficients.b = 1;
        new_breakpoint_coefficients.c = -breakpoint_position;
        // XXX: I think that this is always the same

        // Insert the new breakpoint
        pair<map<double, Coefficients>::iterator, bool> insert = this->breakpoint_coefficients.insert(pair<double, Coefficients>(breakpoint_position, new_breakpoint_coefficients));
        auto breakpoint_ptr = insert.first;
        auto breakpoint_already_exists = insert.second;

        // If the breakpoint already existed, double all its coefficients
        breakpoint_ptr->second.a *= 2;
        breakpoint_ptr->second.b *= 2;
        breakpoint_ptr->second.c *= 2;

        if(is_upper_bound){
            if(this->min_ptr->first <= breakpoint_position){
                // No change in minimum
            }
            else{
                // Change in minimum
                

            }
        } else{
            if(this->min_ptr->first <= breakpoint_position){
                // Change in minimum
            }
            else{
                // No change in minimum
            }
        }
    }

    // Log progress
    cout << "\n\nINSERTION COMPLETED\n----------------------------" << endl;
    cout << "Minimum value: " << this->min_val << endl;
    cout << "Minimum position: " << this->min_pos << endl;
    cout << "Minimum coefficients: a=" << this->min_coefficients.a << ", b=" << this->min_coefficients.b << ", c=" << this->min_coefficients.c << endl;
    cout << "The minimum pointer points to the breakpoint at " << this->min_ptr->first << endl;
    cout << "\n\n" << endl;
}