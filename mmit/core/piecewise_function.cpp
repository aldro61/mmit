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
#include <iterator>

#include "piecewise_function.h"

/*
 * Utility functions
 */
std::ostream& operator<<(std::ostream &os, Coefficients const &c){
    return os << "a=" << c.quadratic << ", b=" << c.linear << ", c=" << c.constant;
}

double find_function_min(Coefficients coefficients){
    if(coefficients.quadratic != 0)
        return -coefficients.linear / (2 * coefficients.quadratic);
    else if(coefficients.linear == 0)
        return -INFINITY;
    else
        return -coefficients.linear * INFINITY;
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

    //std::cout << "Request to insert point at " << breakpoint_position << std::endl;

    // Compute the new breakpoint's coefficients
    // TODO: I use 0 for quadratic for now, but we'll need to patch that for the squared hinge loss
    if(this->function_type == squared_hinge){
        std::cout << "Squared hinge is not supported yet." << std::endl;
        return 1;
    }
    Coefficients new_breakpoint_coefficients(0, 1, -breakpoint_position);

    // Insert the new breakpoint
    std::pair<breakpoint_list_t::iterator, bool> insert = this->breakpoint_coefficients.insert(breakpoint_t(breakpoint_position, new_breakpoint_coefficients));
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

        } else if(!is_upper_bound && this->min_ptr != this->breakpoint_coefficients.end() && this->min_ptr->first <= breakpoint_position){
            // First update the minimum coefficients
            this->min_coefficients -= new_breakpoint_coefficients;

            // Move the pointer
            while(std::next(this->min_ptr) != this->breakpoint_coefficients.end() && find_function_min(this->min_coefficients) > this->min_ptr->first)
                this->move_minimum_pointer_right();
                n_pointer_moves++;
        }
    }

    // Log progress
    if(this->verbose){
        std::cout << "\n\nINSERTION COMPLETED\n----------------------------" << std::endl;
        std::cout << "N pointer moves: " << n_pointer_moves << std::endl;
        std::cout << "Minimum value: " << this->get_minimum_value() << std::endl;
        std::cout << "Minimum position: " << this->get_minimum_position() << std::endl;
        std::cout << "Minimum coefficients: " << this->min_coefficients << std::endl;
        std::cout << "The minimum pointer points to the breakpoint at " << ((this->min_ptr != this->breakpoint_coefficients.end()) ? this->min_ptr->first : -9999999) << std::endl;
        std::cout << "Current breakpoints are: [";
        for(auto b: this->breakpoint_coefficients){
            std::cout << b.first << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "\n\n" << std::endl;
    }

    return n_pointer_moves;
}


/*
 * Function global minimum functions
 */
void PiecewiseFunction::move_minimum_pointer_left() {
    //std::cout << "LEFT MOVE" << std::endl;
    this->min_ptr = std::prev(this->min_ptr);
    this->min_coefficients -= this->min_ptr->second;
    //std::cout << "Moved to new coefficients " << this->min_coefficients << std::endl;
}

void PiecewiseFunction::move_minimum_pointer_right() {
    //std::cout << "RIGHT MOVE" << std::endl;
    this->min_coefficients += this->min_ptr->second;
    this->min_ptr = std::next(this->min_ptr);
    //std::cout << "Moved to new coefficients " << this->min_coefficients << std::endl;
}

double PiecewiseFunction::get_minimum_position() {
    // Find the position of the minimum segment's minimum
    // double theoretical_min = find_function_min(this->min_coefficients);
    // std::cout << "Theoretical minimum is " << theoretical_min << std::endl;

    if(this->min_coefficients.linear == 0){
        // If only lower bounds, return +inf
        if(this->min_ptr == this->breakpoint_coefficients.end() && Coefficients(0, 0, 0) == this->min_coefficients){
            //std::cout << "Case: only lower bounds" << std::endl;
            return INFINITY;
        }
        // If only upper bounds, return -inf
        else if(this->min_ptr == this->breakpoint_coefficients.begin() && Coefficients(0, 0, 0) == this->min_coefficients){
            //std::cout << "Case: only upper bounds" << std::endl;
            return -INFINITY;
        }
        // Otherwise, we take the center point of the flat region
        else{
            return (std::prev(this->min_ptr)->first + this->min_ptr->first) / 2;
        }
    }
    else{
        // \x/ case
        return std::prev(this->min_ptr)->first;
    }
}

double PiecewiseFunction::get_minimum_value() {
    double min_pos = this->get_minimum_position();
    double x_square = min_pos * min_pos;
    if(x_square == INFINITY)  // Unbounded
        return 0;
    return this->min_coefficients.quadratic * x_square + this->min_coefficients.linear * min_pos + this->min_coefficients.constant;
}