#ifndef DFPBRT_CAMERA_H
#define DFPBRT_CAMERA_H

#include <dfpbrt/base/camera.h>
#include <dfpbrt/base/film.h>
#include <dfpbrt/interaction.h>
#include <dfpbrt/ray.h>
#include <dfpbrt/util/spectrum.h>
#include <dfpbrt/util/transform.h>

namespace dfpbrt{
// CameraRay Definition
// The CameraRay structure that is returned by GenerateRay() includes both a ray and a spectral weight associated with it. 
// Simple camera models leave the weight at the default value of one, while more sophisticated ones like RealisticCamera return a weight that is used in modeling the radiometry of image formation. 
struct CameraRay {
    Ray ray;
    SampledSpectrum weight = SampledSpectrum(1);// == {1, 2, 1, 1}
};
struct CameraRayDifferential{
    RayDifferential ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

// CameraTransform Definition
class CameraTransform {
  public:
    // CameraTransform Public Methods
    CameraTransform() = default;
    explicit CameraTransform(const AnimatedTransform &worldFromCamera);

    DFPBRT_CPU_GPU
    Point3f RenderFromCamera(Point3f p, Float time) const {
        return renderFromCamera(p, time);
    }
    DFPBRT_CPU_GPU
    Point3f CameraFromRender(Point3f p, Float time) const {
        return renderFromCamera.ApplyInverse(p, time);
    }
    DFPBRT_CPU_GPU
    Point3f RenderFromWorld(Point3f p) const { return worldFromRender.ApplyInverse(p); }

    DFPBRT_CPU_GPU
    Transform RenderFromWorld() const { return Inverse(worldFromRender); }
    DFPBRT_CPU_GPU
    Transform CameraFromRender(Float time) const {
        return Inverse(renderFromCamera.Interpolate(time));
    }
    DFPBRT_CPU_GPU
    Transform CameraFromWorld(Float time) const {
        return Inverse(worldFromRender * renderFromCamera.Interpolate(time));
    }

    DFPBRT_CPU_GPU
    bool CameraFromRenderHasScale() const { return renderFromCamera.HasScale(); }

    DFPBRT_CPU_GPU
    Vector3f RenderFromCamera(Vector3f v, Float time) const {
        return renderFromCamera(v, time);
    }

    DFPBRT_CPU_GPU
    Normal3f RenderFromCamera(Normal3f n, Float time) const {
        return renderFromCamera(n, time);
    }

    DFPBRT_CPU_GPU
    Ray RenderFromCamera(const Ray &r) const { return renderFromCamera(r); }

    DFPBRT_CPU_GPU
    RayDifferential RenderFromCamera(const RayDifferential &r) const {
        return renderFromCamera(r);
    }

    DFPBRT_CPU_GPU
    Vector3f CameraFromRender(Vector3f v, Float time) const {
        return renderFromCamera.ApplyInverse(v, time);
    }

    DFPBRT_CPU_GPU
    Normal3f CameraFromRender(Normal3f v, Float time) const {
        return renderFromCamera.ApplyInverse(v, time);
    }

    DFPBRT_CPU_GPU
    const AnimatedTransform &RenderFromCamera() const { return renderFromCamera; }

    DFPBRT_CPU_GPU
    const Transform &WorldFromRender() const { return worldFromRender; }

    std::string ToString() const;

  private:
    // CameraTransform Private Members
    // CameraTransform maintains two transformations: one from camera space to the rendering space, and one from the rendering space to world space. 
    // In pbrt, the latter transformation cannot be animated; any animation in the camera transformation is kept in the first transformation. 
    // This ensures that a moving camera does not cause static geometry in the scene to become animated, which in turn would harm performance.
    AnimatedTransform renderFromCamera;
    Transform worldFromRender;
};

}

#endif