#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <utility>

#include "main.h"
#include "piecewise_function.h"

int main() {

    PiecewiseFunction test(0.1);
    test.insert_point(1.0, true);
    test.insert_point(0.0, true);
    test.insert_point(0.5, false);

    for(int i = 0; i < 50000; i++){
        bool is_upper_bound = ((double) rand() / (RAND_MAX)) < 0.5 ? true : false;
        int multiplier =  ((double) rand() / (RAND_MAX)) < 0.5 ? -1 : 1;
        double y = rand() * multiplier;
        test.insert_point(y, is_upper_bound);
        cout << "Minimum: " << test.get_minimum_value() << endl;
    }
    cout << "Finished" << endl;
    return 0;
}

