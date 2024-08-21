#ifndef DFPBRT_UTIL_MATH_H
#define DFPBRT_UTIL_MATH_H

#include <dfpbrt/dfpbrt.h>
#include <cmath>

namespace dfpbrt{

template <typename Ta, typename Tb, typename Tc, typename Td>
inline auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d) {
    //calculate ab-cd with high precision
    auto cd = c * d;
    auto differenceOfProducts = std::fma(a, b, -cd);
    auto error = std::fma(-c, d, cd);
    return differenceOfProducts + error;
}

inline bool Quadratic(float a, float b, float c, float *t0, float *t1) {
    // solve the quadratic equation ax^2 + bx + c = 0 and two roots are t0 and t1
    // Handle case of $a=0$ for quadratic solution
    if (a == 0) {
        if (b == 0)
            return false;
        *t0 = *t1 = -c / b;
        return true;
    }

    // Find quadratic discriminant
    float discrim = DifferenceOfProducts(b, b, 4 * a, c);
    if (discrim < 0)
        return false;
    float rootDiscrim = std::sqrt(discrim);

    // Compute quadratic _t_ values
    float q = -0.5f * (b + std::copysign(rootDiscrim, b));
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1)
        std::swap(*t0, *t1);

    return true;
}


inline bool Quadratic(double a, double b, double c, double *t0, double *t1) {
    // Find quadratic discriminant
    double discrim = DifferenceOfProducts(b, b, 4 * a, c);
    if (discrim < 0)
        return false;
    double rootDiscrim = std::sqrt(discrim);

    if (a == 0) {
        *t0 = *t1 = -c / b;
        return true;
    }

    // Compute quadratic _t_ values
    double q = -0.5 * (b + std::copysign(rootDiscrim, b));
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1)
        std::swap(*t0, *t1);
    return true;
}

}

#endif