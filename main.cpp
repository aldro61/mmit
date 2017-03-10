#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <utility>

#include "main.h"
#include "PiecewiseFunction.h"

int main() {

    PiecewiseFunction test(0.1);
    test.insert_point(1.0, true);
    test.insert_point(0.0, true);
    test.insert_point(0.5, false);

    return 0;
}

