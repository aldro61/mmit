#include <algorithm>
#include <iostream>
#include <set>
#include <utility>

#include "main.h"

#define INFTY 9999999

/*
 * Finds the split that minimizes the sum of slacks.
 */
std::pair<float, float> FindOptimalSplit(float x[], float y_lower[], float y_upper[], int size, float epsilon){
    // Sort all the arrays according to the x values

    // Merge any duplicate x values and increment their weights (+ and -)
    
    // XXX: From now on, we assume that everything is sorted and that duplicates are properly weighted
    // A duplicate that required reweighting is a value of y that is identical, has the same sign and occurs at the same x

    // Assume that all y_upper and y_lower have been flattened into a single y array
    // A weight of +1 indicates that it is the only y that is an upper bound and that occurs at this particular x position (similar for -1)
    // Weights that are not 1 indicate that more than 1 value of the same type occurred at the same x

    // Dummy variables just to test the solver
    float x_dummy[] = {1, 2, 3, 4, 5, 6};
    float y_dummy[] = {1, -1, 2, -2, 3, -3};
    int w_dummy[] = {1, -1, 1, -1, 1, -1};
    x = x_dummy;

    // Prepare variables for the solver
    std::set<bkpnt_t> breakpoints;
    float optimal_point;
    float optimal_cost;
    std::set<bkpnt_t>::iterator minimum_breakpoint;

    // For each value of x, call the solver
    int range_start = 0;
    int range_stop = -1;
    int x_threshold;
    while(range_stop < size - 1) {

        // Find all new intervals for which the x values is <= x_threshold
        range_start = range_stop + 1;
        range_stop = range_start;
        x_threshold = x[range_start];
        while(x[range_stop] <= x_threshold && range_stop  < size){
            range_stop++;
        }
        range_stop--;
        std::cout << "Intervals in range [" << range_start << ", " << range_stop << "] must be added to the solution." << std::endl;

        MinSlackSolver(y_dummy, range_start, range_stop, w_dummy, epsilon, breakpoints, optimal_point, optimal_cost, minimum_breakpoint);
    }

    std::pair<float, float> dummy;
    dummy.first = 0;
    dummy.second = 0;
    return dummy;
}

/*
 * Parameters:
 * -----------
 * y: the interval bounds to add
 * weights: the weight associated to each y (positive for upper bounds, negative for lower bounds)
 * size: the number of bounds to add
 * breakpoints: the set of breakpoints of the current solution (will be updated)
 * optimal_point: the c-value that minimizes the objective function (will be updated)
 * optimal_cost: the cost of the minimum solution
 * minimum_breakpoint: an iterator that points to the minimum breakpoint
 */
std::pair<float, float> MinSlackSolver(float y[], int y_start, int y_end, int weights[], float epsilon,
                                       std::set<bkpnt_t>& breakpoints, float& optimal_point, float& optimal_cost,
                                       std::set<bkpnt_t>::iterator& minimum_breakpoint){

    std::cout << "Solver on " << y_end - y_start + 1 << " intervals." << std::endl;
    // Adding all the ys is equivalent to calling the algo multiple times
    for(int i = y_start; i <= y_end; i++){
        float bkpnt_pos = weights[i] > 0 ? y[i] - epsilon: y[i] + epsilon;

        if(breakpoints.empty()){
            // Base case
            auto insert = breakpoints.insert(bkpnt_t (bkpnt_pos, weights[i]));
            minimum_breakpoint = insert.first;
            optimal_cost = INFTY;
            optimal_point = INFTY;
        }
        else{
            // General case
            std::cout << "minimum weight: " << minimum_breakpoint->second << std::endl;
            std::cout << "minimum at : " << minimum_breakpoint->first << std::endl;
            std::cout << "new point at : " << bkpnt_pos << std::endl;

            // Need to check if point is already in breakpoints and increment weight is it is
            auto insert = breakpoints.insert(bkpnt_t (bkpnt_pos, weights[i])).first;

            // Check on which side of the minimum the new point falls
            if(minimum_breakpoint->first < bkpnt_pos){
                std::cout << "Right of the minimum" << std::endl;
                if(minimum_breakpoint->second > 0){
                    std::cout << "No change to minimum region" << std::endl;
                }
                else{
                    std::cout << "The minimum region changes" << std::endl;
                    minimum_breakpoint = insert;
                    // Selon le signe, le changement n'est pas le même
                }
            }
            else{
                std::cout << "Left of the minimum" << std::endl;
                if(minimum_breakpoint->second > 0){
                    std::cout << "The minimum region changes" << std::endl;
                    minimum_breakpoint = insert;
                }
                else{
                    std::cout << "No change to minimum region" << std::endl;
                }
            }

            if(minimum_breakpoint->second < 0)
                optimal_point = (minimum_breakpoint->first + std::next(minimum_breakpoint)->first) / 2;
            else
                optimal_point = (minimum_breakpoint->first + std::prev(minimum_breakpoint)->first) / 2;
        }

        std::cout << "Minimum breakpoint is at " << minimum_breakpoint->first << std::endl;
        std::cout << "The optimal point is at " << optimal_point << std::endl;
        std::cout << std::endl;
    }

    std::pair<float, float> dummy;
    dummy.first = 0;
    dummy.second = 0;
    return dummy;
};


int main() {
    float x[] = {1, 1, 1, 4, 4, 7};
    float y_upper[] = {1, 2, 3, 4, 6, 7};
    float y_lower[] = {-1, -2, -3, -4, -6, -7};
    FindOptimalSplit(x, y_lower, y_upper, 6, 0.1);
    return 0;
}

