#ifndef DFPBRT_BASE_FILM_H
#define DFPBRT_BASE_FILM_H

#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/taggedptr.h>

namespace dfpbrt {

class VisibleSurface;
class RGBFilm;
class GBufferFilm;
class SpectralFilm;
class PixelSensor;

// Film Definition
class Film : public TaggedPointer<RGBFilm, GBufferFilm, SpectralFilm> {
  public:
    // Film Interface
    DFPBRT_CPU_GPU inline void AddSample(Point2i pFilm, SampledSpectrum L,
                                       const SampledWavelengths &lambda,
                                       const VisibleSurface *visibleSurface,
                                       Float weight);

    DFPBRT_CPU_GPU inline Bounds2f SampleBounds() const;

    DFPBRT_CPU_GPU
    bool UsesVisibleSurface() const;

    DFPBRT_CPU_GPU
    void AddSplat(Point2f p, SampledSpectrum v, const SampledWavelengths &lambda);

    DFPBRT_CPU_GPU inline SampledWavelengths SampleWavelengths(Float u) const;

    DFPBRT_CPU_GPU inline Point2i FullResolution() const;
    DFPBRT_CPU_GPU inline Bounds2i PixelBounds() const;
    DFPBRT_CPU_GPU inline Float Diagonal() const;

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);

    DFPBRT_CPU_GPU inline RGB ToOutputRGB(SampledSpectrum L,
                                        const SampledWavelengths &lambda) const;

    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    DFPBRT_CPU_GPU
    RGB GetPixelRGB(Point2i p, Float splatScale = 1) const;

    DFPBRT_CPU_GPU inline Filter GetFilter() const;
    DFPBRT_CPU_GPU inline const PixelSensor *GetPixelSensor() const;
    std::string GetFilename() const;

    using TaggedPointer::TaggedPointer;

    static Film Create(const std::string &name, const ParameterDictionary &parameters,
                       Float exposureTime, const CameraTransform &cameraTransform,
                       Filter filter, const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    DFPBRT_CPU_GPU inline void ResetPixel(Point2i p);
};



}


#endif