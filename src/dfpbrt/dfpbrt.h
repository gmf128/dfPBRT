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

#if defined(DFPBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
#ifndef DFPBRT_NOINLINE
#define DFPBRT_NOINLINE __attribute__((noinline))
#endif
#define DFPBRT_CPU_GPU __host__ __device__
#define DFPBRT_GPU __device__
#if defined(DFPBRT_IS_GPU_CODE)
#define DFPBRT_CONST __device__ const
#else
#define DFPBRT_CONST const
#endif
#else
#define DFPBRT_CONST const
#define DFPBRT_CPU_GPU
#define DFPBRT_GPU
#endif


#ifdef DFPBRT_FLOAT_AS_DOUBLE
using Float = double;
using FloatBits = uint64_t;
#else
using Float = float;
using FloatBits = uint32_t;
#endif
}

#endif