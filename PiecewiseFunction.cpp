#include <cmath>
#include <iostream>
#include <utility>

#include "PiecewiseFunction.h"

/*
 * Utility functions
 */
ostream& operator<<(ostream &os, Coefficients const &c){
    return os << "a=" << c.a << ", b=" << c.b << ", c=" << c.c;
}

double find_function_min(Coefficients coefficients){
    if(coefficients.a != 0)
        return -coefficients.b / (2 * coefficients.a);
    else if(coefficients.a == 0 && coefficients.b == 0 && coefficients.c == 0)
        return -INFINITY;
    else
        return -coefficients.b * INFINITY;
}


/*
 * Breakpoint insertion functions
 */
double PiecewiseFunction::get_breakpoint_position(double y, bool is_upper_bound) {
    return y + (is_upper_bound ? -1 : 1) * this->margin;
}

void PiecewiseFunction::insert_point(double y, bool is_upper_bound) {

    // New breakpoint's position
    double breakpoint_position = this->get_breakpoint_position(y, is_upper_bound);

    // Compute the new breakpoint's coefficients
    // TODO: I use 0 for a for now, but we'll need to patch that for the squared hinge loss
    Coefficients new_breakpoint_coefficients(0, 1, -breakpoint_position);

    // Insert the new breakpoint
    pair<map<double, Coefficients>::iterator, bool> insert = this->breakpoint_coefficients.insert(pair<double, Coefficients>(breakpoint_position, new_breakpoint_coefficients));
    auto breakpoint_ptr = insert.first;
    auto breakpoint_was_added = insert.second;

    if(this->breakpoint_coefficients.size() == 1) {
        if (is_upper_bound) {
            this->min_ptr = breakpoint_ptr;
        }
    }
    else{
        // If the breakpoint already existed, double all its coefficients
        if (!breakpoint_was_added)
            breakpoint_ptr->second *= 2;

        if(is_upper_bound && this->min_ptr->first > breakpoint_position){
            // First update the minimum coefficients
            this->min_coefficients += new_breakpoint_coefficients;

            // Move the pointer
            while(this->min_ptr != this->breakpoint_coefficients.begin() &&
                    find_function_min(this->min_coefficients) < prev(this->min_ptr)->first)
                this->move_minimum_pointer_left();
            if(find_function_min(this->min_coefficients) > this->min_ptr->first)
                // We went too far left, go back one
                this->move_minimum_pointer_right();

        } else if(!is_upper_bound && this->min_ptr->first <= breakpoint_position){
            // First update the minimum coefficients
            this->min_coefficients -= new_breakpoint_coefficients;

            // Move the pointer
            while(next(this->min_ptr) != this->breakpoint_coefficients.end() && find_function_min(this->min_coefficients) > this->min_ptr->first)
                this->move_minimum_pointer_right();
        }
    }

    // Log progress
//    cout << "\n\nINSERTION COMPLETED\n----------------------------" << endl;
//    cout << "Minimum value: " << this->get_minimum_value() << endl;
//    cout << "Minimum position: " << this->get_minimum_position() << endl;
//    cout << "Minimum coefficients: " << this->min_coefficients << endl;
//    cout << "The minimum pointer points to the breakpoint at " << this->min_ptr->first << endl;
//    cout << "Current breakpoints are: [";
//    for(auto b: this->breakpoint_coefficients){
//        cout << b.first << ", ";
//    }
//    cout << "]" << endl;
//    cout << "\n\n" << endl;
}


/*
 * Function global minimum functions
 */
void PiecewiseFunction::move_minimum_pointer_left() {
    this->min_ptr = prev(this->min_ptr);
    this->min_coefficients -= this->min_ptr->second;
}

void PiecewiseFunction::move_minimum_pointer_right() {
    this->min_ptr = next(this->min_ptr);
    this->min_coefficients += this->min_ptr->second;
}

double PiecewiseFunction::get_minimum_position() {
    // Find the position of the minimum segment's minimum
    double theoretical_min = find_function_min(this->min_coefficients);

    // If there is another breakpoint to the right of the minimum pointer, the minimum is the middle point
    if(this->min_ptr != this->breakpoint_coefficients.begin()){
        return (prev(this->min_ptr)->first + this->min_ptr->first) / 2;
    }
    else
        return theoretical_min;
}

double PiecewiseFunction::get_minimum_value() {
    double min_pos = this->get_minimum_position();
    if(pow(min_pos, 2) == INFINITY)  // Unbounded
        return INFINITY;
    return this->min_coefficients.a * pow(min_pos, 2) + this->min_coefficients.b * min_pos + this->min_coefficients.c;
}