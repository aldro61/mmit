#ifndef MMIT_COEFFICIENTS_H
#define MMIT_COEFFICIENTS_H


class Coefficients{
public:
    double quadratic = 0;
    double linear = 0;
    double constant = 0;

    Coefficients(){this->quadratic = 0; this->linear = 0; this->constant = 0;}
    Coefficients(double a, double b, double c){this->quadratic = a; this->linear = b; this->constant = c;}
    Coefficients operator+(Coefficients &other);
    void operator+=(Coefficients &other);
    Coefficients operator-(Coefficients &other);
    void operator-=(Coefficients &other);
    Coefficients operator*(double scalar);
    void operator*=(double scalar);
    Coefficients operator/(double scalar);
    void operator/=(double scalar);
};


#endif //MMIT_COEFFICIENTS_H
