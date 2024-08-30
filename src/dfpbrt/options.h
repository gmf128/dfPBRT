#ifndef DFPBRT_OPTIONS_H
#define DFPBRT_OPTIONS_H

#include <dfpbrt/dfpbrt.h>

#include <optional>
#include <dfpbrt/util/vecmath.h>

namespace dfpbrt{
// RenderingCoordinateSystem Definition
enum class RenderingCoordinateSystem { Camera, CameraWorld, World };
std::string ToString(const RenderingCoordinateSystem &);

// BasicDFPBRTOptions Definition
struct BasicDFPBRTOptions {
    int seed = 0;
    bool quiet = false;
    bool disablePixelJitter = false, disableWavelengthJitter = false;
    bool disableTextureFiltering = false;
    bool disableImageTextures = false;
    bool forceDiffuse = false;
    bool useGPU = false;
    bool wavefront = false;
    bool interactive = false;
    bool fullscreen = false;
    RenderingCoordinateSystem renderingSpace = RenderingCoordinateSystem::CameraWorld;
};

// PBRTOptions Definition
struct DFPBRTOptions : BasicDFPBRTOptions {
    int nThreads = 0;
    LogLevel logLevel = LogLevel::Error;
    std::string logFile;
    bool logUtilization = false;
    bool writePartialImages = false;
    bool recordPixelStatistics = false;
    bool printStatistics = false;
    std::optional<int> pixelSamples;
    std::optional<int> gpuDevice;
    bool quickRender = false;
    bool upgrade = false;
    std::string imageFile;
    std::string mseReferenceImage, mseReferenceOutput;
    std::string debugStart;
    std::string displayServer;
    std::optional<Bounds2f> cropWindow;
    std::optional<Bounds2i> pixelBounds;
    std::optional<Point2i> pixelMaterial;
    Float displacementEdgeScale = 1;

    std::string ToString() const;
};

// Options Global Variable Declaration
extern DFPBRTOptions *Options;

#if defined(DFPBRT_BUILD_GPU_RENDERER)
#if defined(__CUDACC__)
extern __constant__ BasicDFPBRTOptions OptionsGPU;
#endif  // __CUDACC__

void CopyOptionsToGPU();
#endif  // PBRT_BUILD_GPU_RENDERER

// Options Inline Functions
DFPBRT_CPU_GPU inline const BasicDFPBRTOptions &GetOptions();

DFPBRT_CPU_GPU inline const BasicDFPBRTOptions &GetOptions() {
#if defined(PBRT_IS_GPU_CODE)
    return OptionsGPU;
#else
    return *Options;
#endif
}
}
#endif