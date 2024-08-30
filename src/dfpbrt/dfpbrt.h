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
//Shit
#define NOMINMAX
#include <Windows.h>

// From ABSL_ARRAYSIZE
#define DFPBRT_ARRAYSIZE(array) (sizeof(::dfpbrt::detail::ArraySizeHelper(array)))

namespace dfpbrt{

namespace detail {

template <typename T, uint64_t N>
auto ArraySizeHelper(const T (&array)[N]) -> char (&)[N];

}  // namespace detail

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

template <typename T>
class Vector2;
template <typename T>
class Vector3;
template <typename T>
class Point3;
template <typename T>
class Point2;
template <typename T>
class Normal3;
using Point2f = Point2<Float>;
using Point2i = Point2<int>;
using Point3f = Point3<Float>;
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;

template <typename T>
class Bounds2;
using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
template <typename T>
class Bounds3;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;

class AnimatedTransform;
class BilinearPatchMesh;
class Interaction;
class Interaction;
class MediumInteraction;
class Ray;
class RayDifferential;
class SurfaceInteraction;
class SurfaceInteraction;
class Transform;
class TriangleMesh;

class RGB;
class RGBColorSpace;
class RGBSigmoidPolynomial;
class RGBIlluminantSpectrum;
class SampledSpectrum;
class SampledWavelengths;
class SpectrumWavelengths;
class XYZ;
enum class SpectrumType;

class BSDF;
class CameraTransform;
class Image;
class ParameterDictionary;
struct NamedTextures;
class TextureParameterDictionary;
struct ImageMetadata;
struct MediumInterface;
struct PBRTOptions;

class PiecewiseConstant1D;
class PiecewiseConstant2D;
class ProgressReporter;
class RNG;
struct FileLoc;
class Interval;
template <typename T>
class Array2D;

template <typename T>
struct SOA;
class ScratchBuffer;
class DFPBRTOptions;
// Define _Allocator_
// TODO: substitude std pmr and Float -> std::bite
using Allocator = std::pmr::polymorphic_allocator<Float>;

// Initialization and Cleanup Function Declarations
void InitDFPBRT(const DFPBRTOptions &);
void CleanupPBRT();


}

#endif