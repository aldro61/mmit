#include <set>
#include <utility>

#ifndef MMIT_MAIN_H
#define MMIT_MAIN_H

typedef std::pair<double, int> bkpnt_t;

std::pair<double, double> FindOptimalSplit(double x[], double y_lower[], double y_upper[], int size, double epsilon);

std::pair<double, double> MinSlackSolver(double y[], int y_start, int y_end, int weights[], double epsilon,
                                       std::set<bkpnt_t>& breakpoints, double& optimal_point, double& optimal_cost,
                                       std::pair<std::set<bkpnt_t>::iterator, std::set<bkpnt_t>::iterator>& minimum_region);

#endif //MMIT_MAIN_H
