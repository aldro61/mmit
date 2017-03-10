#ifndef MMIT_COEFFICIENTS_H
#define MMIT_COEFFICIENTS_H


class Coefficients{
public:
    double a = 0;
    double b = 0;
    double c = 0;

    Coefficients(){this->a = 0; this->b = 0; this->c = 0;}
    Coefficients(double a, double b, double c){this->a = a; this->b = b; this->c = c;}
    Coefficients operator+(Coefficients other);
    void operator+=(Coefficients other);
    Coefficients operator-(Coefficients other);
    void operator-=(Coefficients other);
    Coefficients operator*(double scalar);
    void operator*=(double scalar);
    Coefficients operator/(double scalar);
    void operator/=(double scalar);
};


#endif //MMIT_COEFFICIENTS_H
