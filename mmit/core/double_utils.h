#ifndef CORE_DOUBLE_UTILS_H
#define CORE_DOUBLE_UTILS_H

#define TOL 1e-6

inline bool equal(double x, double y){
    return std::abs(x - y) <= TOL;
}

inline bool not_equal(double x, double y){
    return std::abs(x - y) > TOL;
}

inline bool greater(double x, double y){
    if(std::abs(x) == INFINITY || std::abs(y) == INFINITY){
        return x > y;
    }
    else{
        return std::abs(x - y) > TOL && x > y;
    }
}

inline bool less(double x, double y){
    if(std::abs(x) == INFINITY || std::abs(y) == INFINITY){
        return x < y;
    }
    else{
        return std::abs(x - y) > TOL && x < y;
    }
}

struct DoubleComparatorLess
{
    bool operator()(double a, double b)
    {
        return less(a, b);
    }
};

#endif //CORE_DOUBLE_UTILS_H
