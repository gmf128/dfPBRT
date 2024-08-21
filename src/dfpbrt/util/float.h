#ifndef DFPBRT_UTIL_FLOAT_H
#define DFPBRT_UTIL_FLOAT_H

#include <dfpbrt/dfpbrt.h>

#include <cmath>
#include <limits>

namespace dfpbrt{
    // Floating-point Inline Functions
   
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
}


#endif