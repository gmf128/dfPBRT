#ifndef DFPBRT_UTIL_MATH_H
#define DFPBRT_UTIL_MATH_H

#include <dfpbrt/dfpbrt.h>
#include <cmath>

namespace dfpbrt{

// Mathematical Constants
constexpr Float ShadowEpsilon = 0.0001f;

constexpr Float Pi = 3.14159265358979323846;
constexpr Float InvPi = 0.31830988618379067154;
constexpr Float Inv2Pi = 0.15915494309189533577;
constexpr Float Inv4Pi = 0.07957747154594766788;
constexpr Float PiOver2 = 1.57079632679489661923;
constexpr Float PiOver4 = 0.78539816339744830961;
constexpr Float Sqrt2 = 1.41421356237309504880;

template <typename T>
inline constexpr T Sqr(T v) {
    return v * v;
}

template <typename T, typename U, typename V>
inline constexpr T Clamp(T val, U low, V high) {
    if (val < low)
        return T(low);
    else if (val > high)
        return T(high);
    else
        return val;
}

inline Float SinXOverX(Float x) {
    if (1 - x * x == 1)
        return 1;
    return std::sin(x) / x;
}

inline float SafeASin(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}
inline float SafeACos(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

inline double SafeASin(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}

inline double SafeACos(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

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