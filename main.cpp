#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <utility>

#include "main.h"
#include "PiecewiseFunction.h"

/*
 * Finds the split that minimizes the sum of slacks.
 */
std::pair<double, double> FindOptimalSplit(double x[], double y_lower[], double y_upper[], int size, double epsilon){
    // Sort all the arrays according to the x values

    // Merge any duplicate x values and increment their weights (+ and -)

    // XXX: From now on, we assume that everything is sorted and that duplicates are properly weighted
    // A duplicate that required reweighting is a value of y that is identical, has the same sign and occurs at the same x

    // Assume that all y_upper and y_lower have been flattened into a single y array
    // A weight of +1 indicates that it is the only y that is an upper bound and that occurs at this particular x position (similar for -1)
    // Weights that are not 1 indicate that more than 1 value of the same type occurred at the same x

    // Dummy variables just to test the solver
    size = 7;
    double x_dummy[] = {1, 2, 3, 4, 5, 6, 7};
    double y_dummy[] = {1, -1, 2, -2, 3, -3, -8, -10};
    int w_dummy[] = {1, -1, 1, -1, 1, -1, 1, -1};
    x = x_dummy;

    // Prepare variables for the solver
    std::set<bkpnt_t> breakpoints;
    double optimal_point;
    double optimal_cost;
    std::pair<std::set<bkpnt_t>::iterator, std::set<bkpnt_t>::iterator> minimum_region;

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

        MinSlackSolver(y_dummy, range_start, range_stop, w_dummy, epsilon, breakpoints, optimal_point, optimal_cost, minimum_region);
    }

    std::pair<double, double> dummy;
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
std::pair<double, double> MinSlackSolver(double y[], int y_start, int y_end, int weights[], double epsilon,
                                       std::set<bkpnt_t>& breakpoints, double& optimal_point, double& optimal_cost,
                                       std::pair<std::set<bkpnt_t>::iterator, std::set<bkpnt_t>::iterator>& minimum_region){

    std::cout << "Solver on " << y_end - y_start + 1 << " intervals." << std::endl;
    // Adding all the ys is equivalent to calling the algo multiple times
    for(int i = y_start; i <= y_end; i++){
        double bkpnt_pos = weights[i] > 0 ? y[i] - epsilon: y[i] + epsilon;

        if(breakpoints.empty()){
            // Base case: we have no points yet
            auto insert = breakpoints.insert(bkpnt_t (bkpnt_pos, weights[i]));
            if(weights[i] > 0) {
                minimum_region.first = breakpoints.end();
                minimum_region.second = insert.first;  // Point is on the right of the minimum
            }
            else{
                minimum_region.first = insert.first;  // Point is on the left of the minimum
                minimum_region.second = breakpoints.end();
            }

            optimal_cost = INFINITY;
            optimal_point = INFINITY;
            std::cout << "The new breakpoint is the first one." << std::endl;
        }
        else{
            // General case
            // TODO: Need to check if point is already in breakpoints and increment weight if it is
            auto insert = breakpoints.insert(bkpnt_t (bkpnt_pos, weights[i])).first;

            // Check if the new breakpoint falls withing the minimum region
            if((optimal_cost == INFINITY && minimum_region.first != breakpoints.end() && minimum_region.first->first <= bkpnt_pos) ||
               (optimal_cost == INFINITY && minimum_region.second != breakpoints.end() && minimum_region.second->first >= bkpnt_pos) ||
               (optimal_cost != INFINITY && minimum_region.first->first <= bkpnt_pos && bkpnt_pos <= minimum_region.second->first)){
                // Effects: * The minimum region is shortened
                //          * The objective value does not change (unless it is infinity)
                if(weights[i] > 0)
                    minimum_region.second = insert;
                else
                    minimum_region.first = insert;

                if(optimal_cost == INFINITY)
                    // The minimum region is now closed and the objective value is 0
                    optimal_cost = 0;

                std::cout << "The new breakpoint falls within the current minimum region." << std::endl<< std::endl<< std::endl<< std::endl<< std::endl;
            }
            else{
                std::cout << "The new breakpoint falls outside the current minimum region." << std::endl;

                // The minimum only changes if its neighbouring slope becomes 0
                if(weights[i] > 0 && bkpnt_pos < minimum_region.first->first){
                    // Modify the minimum
                    std::cout << "MIN CHANGE +1" << bkpnt_pos << std::endl;
                    minimum_region.first = insert;
                    // compute change in opti value
                }
                else if(weights[i] < 0 && bkpnt_pos > minimum_region.second->first){
                    // Modify the minimum
                    std::cout << "MIN CHANGE -1" << bkpnt_pos << std::endl;
                    minimum_region.second = insert;
                }
                else{
                    std::cout << minimum_region.first->first << " " << minimum_region.second->first << " " << bkpnt_pos << std::endl << std::endl;
                }
            }
        }

        //std::cout << "Minimum breakpoint is at " << minimum_breakpoint->first << std::endl;
        //std::cout << "The optimal point is at " << optimal_point << std::endl;
        //std::cout << std::endl;
    }

    std::pair<double, double> dummy;
    dummy.first = 0;
    dummy.second = 0;
    return dummy;
};


int main() {
    PiecewiseFunction test(0.1);
    test.insert_point(1.0, true);

    PiecewiseFunction test1(0.1);
    test1.insert_point(1.0, false);


    //double x[] = {1, 1, 1, 4, 4, 7};
    //double y_upper[] = {1, 2, 3, 4, 6, 7};
    //double y_lower[] = {-1, -2, -3, -4, -6, -7};

    //FindOptimalSplit(x, y_lower, y_upper, 6, 0.1);
    //return 0;
}

