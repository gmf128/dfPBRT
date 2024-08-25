#ifndef DFPBRT_H
#define DFPBRT_H

#include <stdio.h>
#include <cstddef>
#include <cstdint>

//Cpp 20
#include <concepts>
#include <type_traits>
#include <format>

//Platform specific
//todo: using macro to check the platform
#include <Windows.h>


namespace dfpbrt{

#ifdef DFPBRT_FLOAT_AS_DOUBLE
using Float = double;
using FloatBits = uint64_t;
#else
using Float = float;
using FloatBits = uint32_t;
#endif
}

#endif