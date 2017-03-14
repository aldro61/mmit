#include <cmath>
#include <iostream>
#include "coefficients.h"


Coefficients Coefficients::operator+(Coefficients &other) {
    return Coefficients(this->a + other.a, this->b + other.b, this->c + other.c);
}

void Coefficients::operator+=(Coefficients &other) {
    this->a += other.a;
    this->b += other.b;
    this->c += other.c;
}

Coefficients Coefficients::operator-(Coefficients &other) {
    return Coefficients(this->a - other.a, this->b - other.b, this->c - other.c);
}

void Coefficients::operator-=(Coefficients &other) {
    this->a -= other.a;
    this->b -= other.b;
    this->c -= other.c;
}

Coefficients Coefficients::operator*(double scalar) {
    return Coefficients(this->a * scalar, this->b * scalar, this->c * scalar);
}

void Coefficients::operator*=(double scalar) {
    this->a *= scalar;
    this->b *= scalar;
    this->c *= scalar;
}

Coefficients Coefficients::operator/(double scalar) {
    return Coefficients(this->a / scalar, this->b / scalar, this->c / scalar);
}

void Coefficients::operator/=(double scalar) {
    this->a /= scalar;
    this->b /= scalar;
    this->c /= scalar;
}