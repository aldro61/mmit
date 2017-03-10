#include <cmath>
#include <iostream>
#include <utility>

#include "PiecewiseFunction.h"

ostream& operator<<(ostream &os, Coefficients const &c){
    return os << "a=" << c.a << ", b=" << c.b << ", c=" << c.c;
}

double find_min_pos(Coefficients coefficients){
    if(coefficients.a != 0)
        return -coefficients.b / (2 * coefficients.a);
    else if(coefficients.a == 0 && coefficients.b == 0 && coefficients.c == 0)
        return -INFINITY;
    else
        return -coefficients.b * INFINITY;
}

double PiecewiseFunction::get_breakpoint_position(double y, bool is_upper_bound) {
    return y + (is_upper_bound ? -1 : 1) * this->margin;
}

void PiecewiseFunction::insert_point(double y, bool is_upper_bound) {

    // New breakpoint's position
    double breakpoint_position = this->get_breakpoint_position(y, is_upper_bound);

    // Compute the new breakpoint's coefficients
    Coefficients new_breakpoint_coefficients;
    new_breakpoint_coefficients.a = 0;
    new_breakpoint_coefficients.b = 1;
    new_breakpoint_coefficients.c = -breakpoint_position;

    // Insert the new breakpoint
    pair<map<double, Coefficients>::iterator, bool> insert = this->breakpoint_coefficients.insert(pair<double, Coefficients>(breakpoint_position, new_breakpoint_coefficients));
    auto breakpoint_ptr = insert.first;
    auto breakpoint_was_added = insert.second;

    if(this->breakpoint_coefficients.size() == 1) {
        if (is_upper_bound) {
            this->min_ptr = breakpoint_ptr;
            this->min_pos = -INFINITY;
        } else {
            this->min_pos = INFINITY;
        }
        this->min_val = INFINITY;
    }
    else{
        // If the breakpoint already existed, double all its coefficients
        if (!breakpoint_was_added)
            breakpoint_ptr->second *= 2;

        if(is_upper_bound){
            if(this->min_ptr->first <= breakpoint_position){
                // No change in minimum
            }
            else{
                // First update the minimum coefficients
                this->min_coefficients += new_breakpoint_coefficients;

                while(this->min_ptr != this->breakpoint_coefficients.begin() && find_min_pos(this->min_coefficients) < prev(this->min_ptr)->first){
                    // Move pointer left
                    this->min_ptr = prev(this->min_ptr);
                    this->min_coefficients -= this->min_ptr->second;
                }
                // We went too far left, go back one
                if(find_min_pos(this->min_coefficients) > this->min_ptr->first)
                {
                    // Move pointer right
                    this->min_ptr = next(this->min_ptr);
                    this->min_coefficients += this->min_ptr->second;
                }
                this->min_pos = find_min_pos(this->min_coefficients);
                this->min_val = this->min_coefficients.a * pow(this->min_pos, 2) + this->min_coefficients.b * this->min_pos + this->min_coefficients.c;
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
    cout << "Minimum coefficients: " << this->min_coefficients << endl;
    cout << "The minimum pointer points to the breakpoint at " << this->min_ptr->first << endl;
    cout << "Current breakpoints are: [";
    for(auto b: this->breakpoint_coefficients){
        cout << b.first << ", ";
    }
    cout << "]" << endl;
    cout << "\n\n" << endl;
}