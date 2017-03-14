#include <cmath>
#include <iostream>
#include <iterator>

#include "piecewise_function.h"

/*
 * Utility functions
 */
std::ostream& operator<<(std::ostream &os, Coefficients const &c){
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

int PiecewiseFunction::insert_point(double y, bool is_upper_bound) {
    int n_pointer_moves = 0;

    // New breakpoint's position
    double breakpoint_position = this->get_breakpoint_position(y, is_upper_bound);

    // Compute the new breakpoint's coefficients
    // TODO: I use 0 for a for now, but we'll need to patch that for the squared hinge loss
    Coefficients new_breakpoint_coefficients(0, 1, -breakpoint_position);

    // Insert the new breakpoint
    std::pair<std::map<double, Coefficients>::iterator, bool> insert = this->breakpoint_coefficients.insert(std::pair<double, Coefficients>(breakpoint_position, new_breakpoint_coefficients));
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

        if(is_upper_bound && (this->min_ptr == this->breakpoint_coefficients.end() || this->min_ptr->first > breakpoint_position)){
            // First update the minimum coefficients
            this->min_coefficients += new_breakpoint_coefficients;

            // Move the pointer
            while(this->min_ptr != this->breakpoint_coefficients.begin() &&
                    find_function_min(this->min_coefficients) < std::prev(this->min_ptr)->first) {
                this->move_minimum_pointer_left();
                n_pointer_moves ++;
            }
            if(find_function_min(this->min_coefficients) > this->min_ptr->first) {
                // We went too far left, go back one
                this->move_minimum_pointer_right();
                n_pointer_moves--;
            }

        } else if(!is_upper_bound && this->min_ptr->first <= breakpoint_position){
            // First update the minimum coefficients
            this->min_coefficients -= new_breakpoint_coefficients;

            // Move the pointer
            while(std::next(this->min_ptr) != this->breakpoint_coefficients.end() && find_function_min(this->min_coefficients) > this->min_ptr->first)
                this->move_minimum_pointer_right();
                n_pointer_moves++;
        }
    }

//    // Log progress
//    std::cout << "\n\nINSERTION COMPLETED\n----------------------------" << std::endl;
//    std::cout << "N pointer moves: " << n_pointer_moves << std::endl;
//    std::cout << "Minimum value: " << this->get_minimum_value() << std::endl;
//    std::cout << "Minimum position: " << this->get_minimum_position() << std::endl;
//    std::cout << "Minimum coefficients: " << this->min_coefficients << std::endl;
//    std::cout << "The minimum pointer points to the breakpoint at " << ((this->min_ptr != this->breakpoint_coefficients.end()) ? this->min_ptr->first : -9999999) << std::endl;
//    std::cout << "Current breakpoints are: [";
//    for(auto b: this->breakpoint_coefficients){
//        std::cout << b.first << ", ";
//    }
//    std::cout << "]" << std::endl;
//    std::cout << "\n\n" << std::endl;

    return n_pointer_moves;
}


/*
 * Function global minimum functions
 */
void PiecewiseFunction::move_minimum_pointer_left() {
    this->min_ptr = std::prev(this->min_ptr);
    this->min_coefficients -= this->min_ptr->second;
}

void PiecewiseFunction::move_minimum_pointer_right() {
    this->min_coefficients += this->min_ptr->second;
    this->min_ptr = std::next(this->min_ptr);
}

double PiecewiseFunction::get_minimum_position() {
    // Find the position of the minimum segment's minimum
    double theoretical_min = find_function_min(this->min_coefficients);

    // If the function is not open on any side
    if(this->min_ptr != this->breakpoint_coefficients.begin() && this->min_ptr != this->breakpoint_coefficients.end()){
        return (std::prev(this->min_ptr)->first + this->min_ptr->first) / 2;
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