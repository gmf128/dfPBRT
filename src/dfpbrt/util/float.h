#ifndef DFPBRT_UTIL_FLOAT_H
#define DFPBRT_UTIL_FLOAT_H

#include <dfpbrt/dfpbrt.h>
//#include <dfpbrt/util/math.h> Never cross-including headerfile!

#include <cmath>
#include <limits>
#include <bit>
#include <algorithm>

namespace dfpbrt{
    
    //constants
    #define MachineEpsilon std::numeric_limits<Float>::epsilon() * 0.5f
    //using _ as alias in order not to cause same definitions errors
    #define Infinity_  std::numeric_limits<Float>::infinity() 


    // Floating-point Inline Functions
    inline Float FMA(Float a, Float b, Float c){
        return std::fma(a, b, c);
    }
    inline double FMA(double a, double b, double c){
        return std::fma(a, b, c);
    }
   
    //old version using std::enable_if_t<bool_expression, type>
    // template <typename T>
    // inline typename std::enable_if_t<std::is_floating_point_v<T>, bool> IsNaN(
    //         T v) {
    // return std::isnan(v);
    //}

    //new version: using C++ STANDARD 20--Concepts
    template <std::floating_point T>
    inline bool IsNaN (T v){
        return std::isnan(v);
    }

    template <std::integral T>
    inline bool IsNaN (T v){
        return false;
    }


    template<std::floating_point T>
    inline bool IsInf(T v){
        return std::isinf(v);
    }

    template<std::integral T>
    inline bool IsInf(T v){
        return false;
    }

    template <std::floating_point T>
    inline bool IsFinite(T v){
        return std::isfinite(v);
    }
    template <std::integral T>
    inline bool IsFinite(T v){
        return true;
    }

    
inline uint32_t FloatToBits(float f) {
    return std::bit_cast<uint32_t>(f);
}

inline uint64_t FloatToBits(double f) {
    return std::bit_cast<uint64_t>(f);
}


inline float BitsToFloat(uint32_t ui) {
     return std::bit_cast<float>(ui);
}

inline double BitsToFloat(uint64_t ui) {
    return std::bit_cast<double>(ui);
}

    
inline float NextFloatUp(float v) {
    // Handle infinity and negative zero for _NextFloatUp()_
    if (IsInf(v) && v > 0.f)
        return v;
    if (v == -0.f)
        v = 0.f;

    // Advance _v_ to next higher float
    uint32_t ui = FloatToBits(v);
    if (v >= 0)
        ++ui;
    else
        --ui;
    return BitsToFloat(ui);
}


inline float NextFloatDown(float v) {
    // Handle infinity and positive zero for _NextFloatDown()_
    if (IsInf(v) && v < 0.)
        return v;
    if (v == 0.f)
        v = -0.f;
    uint32_t ui = FloatToBits(v);
    if (v > 0)
        --ui;
    else
        ++ui;
    return BitsToFloat(ui);
}

inline constexpr Float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

inline  Float AddRoundUp(Float a, Float b) {
    //CPU code
    return NextFloatUp(a + b);

}
inline  Float AddRoundDown(Float a, Float b) {
    return NextFloatDown(a + b);
}

inline  Float SubRoundUp(Float a, Float b) {
    return AddRoundUp(a, -b);
}
inline  Float SubRoundDown(Float a, Float b) {
    return AddRoundDown(a, -b);
}

inline  Float MulRoundUp(Float a, Float b) {
    return NextFloatUp(a * b);
}

inline  Float MulRoundDown(Float a, Float b) {
    return NextFloatDown(a * b);
}

inline  Float DivRoundUp(Float a, Float b) {
    return NextFloatUp(a / b);

}

inline  Float DivRoundDown(Float a, Float b) {
    return NextFloatDown(a / b);
}

inline  Float SqrtRoundUp(Float a) {
    return NextFloatUp(std::sqrt(a));
}

inline  Float SqrtRoundDown(Float a) {
    return std::max<Float>(0, NextFloatDown(std::sqrt(a)));
}

inline  Float FMARoundUp(Float a, Float b, Float c) {
    return NextFloatUp(FMA(a, b, c));
}

inline  Float FMARoundDown(Float a, Float b, Float c) {
    return NextFloatDown(FMA(a, b, c));
}


inline double NextFloatUp(double v) {
    if (IsInf(v) && v > 0.)
        return v;
    if (v == -0.f)
        v = 0.f;
    uint64_t ui = FloatToBits(v);
    if (v >= 0.f)
        ++ui;
    else
        --ui;
    return BitsToFloat(ui);
}


inline double NextFloatDown(double v) {
    if (IsInf(v) && v < 0.)
        return v;
    if (v == 0.f)
        v = -0.f;
    uint64_t ui = FloatToBits(v);
    if (v > 0.)
        --ui;
    else
        ++ui;
    return BitsToFloat(ui);
}

}


#endif