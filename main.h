#include <set>
#include <utility>

#ifndef MMIT_MAIN_H
#define MMIT_MAIN_H

typedef std::pair<float, int> bkpnt_t;

std::pair<float, float> FindOptimalSplit(float x[], float y_lower[], float y_upper[], int size, float epsilon);

std::pair<float, float> MinSlackSolver(float y[], int y_start, int y_end, int weights[], float epsilon,
                                       std::set<bkpnt_t>& breakpoints, float& optimal_point, float& optimal_cost,
                                       std::set<bkpnt_t>::iterator& minimum_breakpoint);

#endif //MMIT_MAIN_H
