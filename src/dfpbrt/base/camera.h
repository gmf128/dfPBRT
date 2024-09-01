#ifndef DFPBRT_BASE_CAMERA_H
#define DFPBRT_BASE_CAMERA_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/vecmath.h>
#include <dfpbrt/util/transform.h>
#include <dfpbrt/base/film.h>
#include <dfpbrt/base/filter.h>
#include <dfpbrt/base/sampler.h>
#include <dfpbrt/util/taggedptr.h>

namespace dfpbrt{

//camera-related struct
struct CameraRay;
struct CameraRayDifferential;
struct CameraWiSample;

struct CameraSample;
class CameraTransform;

//all possible camera classes
class PerspectiveCamera;
class OrthographicCamera;
class SphericalCamera;
class RealisticCamera;

class Camera: public TaggedPointer<PerspectiveCamera, OrthographicCamera, SphericalCamera, RealisticCamera>{
    public:
        // Camera Interface
        // If not, we will not able to use Camera(T * ptr)
        using TaggedPointer::TaggedPointer;

        static Camera Create(const std::string &name, const ParameterDictionary &parameters,
                         Medium medium, const CameraTransform &cameraTransform, Film film,
                         const FileLoc *loc, Allocator alloc);

        std::string ToString() const;

        DFPBRT_CPU_GPU inline std::optional<CameraRay> GenerateRay(
            CameraSample sample, SampledWavelengths &lambda) const;

        DFPBRT_CPU_GPU
        std::optional<CameraRayDifferential> GenerateRayDifferential(
            CameraSample sample, SampledWavelengths &lambda) const;

        DFPBRT_CPU_GPU inline Film GetFilm() const;

        DFPBRT_CPU_GPU inline Float SampleTime(Float u) const;

        void InitMetadata(ImageMetadata *metadata) const;

        DFPBRT_CPU_GPU inline const CameraTransform &GetCameraTransform() const;

        DFPBRT_CPU_GPU
        void Approximate_dp_dxy(Point3f p, Normal3f n, Float time, int samplesPerPixel,
                            Vector3f *dpdx, Vector3f *dpdy) const;

        DFPBRT_CPU_GPU
        SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                        Point2f *pRasterOut = nullptr) const;

        DFPBRT_CPU_GPU
        void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;

        DFPBRT_CPU_GPU
        std::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const;

}


}


#endif