#include <cmath>
#include <iostream>
#include "coefficients.h"


Coefficients Coefficients::operator+(Coefficients &other) {
    return Coefficients(this->quadratic + other.quadratic, this->linear + other.linear, this->constant + other.constant);
}

void Coefficients::operator+=(Coefficients &other) {
    this->quadratic += other.quadratic;
    this->linear += other.linear;
    this->constant += other.constant;
}

Coefficients Coefficients::operator-(Coefficients &other) {
    return Coefficients(this->quadratic - other.quadratic, this->linear - other.linear, this->constant - other.constant);
}

void Coefficients::operator-=(Coefficients &other) {
    this->quadratic -= other.quadratic;
    this->linear -= other.linear;
    this->constant -= other.constant;
}

Coefficients Coefficients::operator*(double scalar) {
    return Coefficients(this->quadratic * scalar, this->linear * scalar, this->constant * scalar);
}

void Coefficients::operator*=(double scalar) {
    this->quadratic *= scalar;
    this->linear *= scalar;
    this->constant *= scalar;
}

Coefficients Coefficients::operator/(double scalar) {
    return Coefficients(this->quadratic / scalar, this->linear / scalar, this->constant / scalar);
}

void Coefficients::operator/=(double scalar) {
    this->quadratic /= scalar;
    this->linear /= scalar;
    this->constant /= scalar;
}

bool Coefficients::operator==(Coefficients &other){
    return this->quadratic == other.quadratic && this->linear == other.linear && this->constant == other.constant;
}