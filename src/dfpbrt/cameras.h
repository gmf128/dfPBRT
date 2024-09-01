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

// CameraBaseParameters Definition
struct CameraBaseParameters {
    // One of the most important is the transformation that places the camera in the scene, which is represented by a CameraTransform and is stored in the cameraTransform member variable.
    CameraTransform cameraTransform;
    // Next is a pair of floating-point values that give the times at which the camera’s shutter opens and closes.
    Float shutterOpen = 0, shutterClose = 1;
    // A Film instance stores the final image and models the film sensor.
    Film film;
    // Last is a Medium instance that represents the scattering medium that the camera lies in, if any
    Medium medium;
    CameraBaseParameters() = default;
    CameraBaseParameters(const CameraTransform &cameraTransform, Film film, Medium medium,
                         const ParameterDictionary &parameters, const FileLoc *loc);
};

// CameraBase Definition
class CameraBase {
  public:
    // CameraBase Public Methods
    DFPBRT_CPU_GPU
    Film GetFilm() const { return film; }
    DFPBRT_CPU_GPU
    const CameraTransform &GetCameraTransform() const { return cameraTransform; }

    DFPBRT_CPU_GPU
    Float SampleTime(Float u) const { return Lerp(u, shutterOpen, shutterClose); }

    void InitMetadata(ImageMetadata *metadata) const;
    std::string ToString() const;

    DFPBRT_CPU_GPU
    void Approximate_dp_dxy(Point3f p, Normal3f n, Float time, int samplesPerPixel,
                            Vector3f *dpdx, Vector3f *dpdy) const {
        // Compute tangent plane equation for ray differential intersections
        Point3f pCamera = CameraFromRender(p, time);
        Transform DownZFromCamera =
            RotateFromTo(Normalize(Vector3f(pCamera)), Vector3f(0, 0, 1));
        Point3f pDownZ = DownZFromCamera(pCamera);
        Normal3f nDownZ = DownZFromCamera(CameraFromRender(n, time));
        Float d = nDownZ.z * pDownZ.z;

        // Find intersection points for approximated camera differential rays
        Ray xRay(Point3f(0, 0, 0) + minPosDifferentialX,
                 Vector3f(0, 0, 1) + minDirDifferentialX);
        Float tx = -(Dot(nDownZ, Vector3f(xRay.o)) - d) / Dot(nDownZ, xRay.d);
        Ray yRay(Point3f(0, 0, 0) + minPosDifferentialY,
                 Vector3f(0, 0, 1) + minDirDifferentialY);
        Float ty = -(Dot(nDownZ, Vector3f(yRay.o)) - d) / Dot(nDownZ, yRay.d);
        Point3f px = xRay(tx), py = yRay(ty);

        // Estimate $\dpdx$ and $\dpdy$ in tangent plane at intersection point
        Float sppScale =
            GetOptions().disablePixelJitter
                ? 1
                : std::max<Float>(.125, 1 / std::sqrt((Float)samplesPerPixel));
        *dpdx =
            sppScale * RenderFromCamera(DownZFromCamera.ApplyInverse(px - pDownZ), time);
        *dpdy =
            sppScale * RenderFromCamera(DownZFromCamera.ApplyInverse(py - pDownZ), time);
    }

  protected:
    // CameraBase Protected Members
    CameraTransform cameraTransform;
    Float shutterOpen, shutterClose;
    Film film;
    Medium medium;
    Vector3f minPosDifferentialX, minPosDifferentialY;
    Vector3f minDirDifferentialX, minDirDifferentialY;

    // CameraBase Protected Methods
    CameraBase() = default;
    CameraBase(CameraBaseParameters p);

    // The below function is a virtual function without using virtual funtion grammar
    // Its first parameter is Camera, Camera is a taggedptr whose template parameters are RealisticCamera, which are derived from CameraBase
    // So, CameraBase cannot call GenerateRayDifferential itself, but its derived class can call this funtion by offer own *this* pointer
    // Therefore, it acts like a virtual function
    DFPBRT_CPU_GPU
    static std::optional<CameraRayDifferential> GenerateRayDifferential(
        Camera camera, CameraSample sample, SampledWavelengths &lambda);

    DFPBRT_CPU_GPU
    Ray RenderFromCamera(const Ray &r) const {
        return cameraTransform.RenderFromCamera(r);
    }

    DFPBRT_CPU_GPU
    RayDifferential RenderFromCamera(const RayDifferential &r) const {
        return cameraTransform.RenderFromCamera(r);
    }

    DFPBRT_CPU_GPU
    Vector3f RenderFromCamera(Vector3f v, Float time) const {
        return cameraTransform.RenderFromCamera(v, time);
    }

    DFPBRT_CPU_GPU
    Normal3f RenderFromCamera(Normal3f v, Float time) const {
        return cameraTransform.RenderFromCamera(v, time);
    }

    DFPBRT_CPU_GPU
    Point3f RenderFromCamera(Point3f p, Float time) const {
        return cameraTransform.RenderFromCamera(p, time);
    }

    DFPBRT_CPU_GPU
    Vector3f CameraFromRender(Vector3f v, Float time) const {
        return cameraTransform.CameraFromRender(v, time);
    }

    DFPBRT_CPU_GPU
    Normal3f CameraFromRender(Normal3f v, Float time) const {
        return cameraTransform.CameraFromRender(v, time);
    }

    DFPBRT_CPU_GPU
    Point3f CameraFromRender(Point3f p, Float time) const {
        return cameraTransform.CameraFromRender(p, time);
    }

    void FindMinimumDifferentials(Camera camera);
};

// ProjectiveCamera Definition
class ProjectiveCamera : public CameraBase {
  public:
    // ProjectiveCamera Public Methods
    ProjectiveCamera() = default;
    void InitMetadata(ImageMetadata *metadata) const;

    std::string BaseToString() const;

    // General parameters are also needed to provide in baseParameters
    // Specific parameters is provided like screenFromCamera, screenWindow, lensRadius and focalLength
    ProjectiveCamera(CameraBaseParameters baseParameters,
                     const Transform &screenFromCamera, Bounds2f screenWindow,
                     Float lensRadius, Float focalDistance)
        : CameraBase(baseParameters),
          screenFromCamera(screenFromCamera),
          lensRadius(lensRadius),
          focalDistance(focalDistance) {
        // Compute projective camera transformations
        // Compute projective camera screen transformations
        // Screen Space x, y \in ScreenWindow; z\in [0, 1]
        // NDC Space x, y \in [0, 1], z remain unchanged
        // Transform NDCFromScreen =
        //     Scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
        //           1 / (screenWindow.pMax.y - screenWindow.pMin.y), 1) *
        //     Translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
        // Transform rasterFromNDC =
        //     Scale(film.FullResolution().x, -film.FullResolution().y, 1);

        // I changed here to satisfy the definition of NDC Space
        Transform NDCFromScreen = 
            Scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                1 / (screenWindow.pMin.y - screenWindow.pMax.y), 1) *
            Translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
        Transform rasterFromNDC = Scale(film.FullResolution().x, film.FullResolution().y, 1);
        rasterFromScreen = rasterFromNDC * NDCFromScreen;
        screenFromRaster = Inverse(rasterFromScreen);

        cameraFromRaster = Inverse(screenFromCamera) * screenFromRaster;
    }

  protected:
    // ProjectiveCamera Protected Members
    Transform screenFromCamera, cameraFromRaster;
    Transform rasterFromScreen, screenFromRaster;
    Float lensRadius, focalDistance;
};

// OrthographicCamera Definition
class OrthographicCamera : public ProjectiveCamera {
  public:
    // OrthographicCamera Public Methods
    OrthographicCamera(CameraBaseParameters baseParameters, Bounds2f screenWindow,
                       Float lensRadius, Float focalDist)
        : ProjectiveCamera(baseParameters, Orthographic(0, 1), screenWindow, lensRadius,
                           focalDist) {
        // The constructor here precomputes how much the ray origins shift in camera space coordinates due to a single pixel shift in the x and y directions on the film plane.
        // Compute differential changes in origin for orthographic camera rays
        dxCamera = cameraFromRaster(Vector3f(1, 0, 0));
        dyCamera = cameraFromRaster(Vector3f(0, 1, 0));

        // Compute minimum differentials for orthographic camera
        minDirDifferentialX = minDirDifferentialY = Vector3f(0, 0, 0);
        minPosDifferentialX = dxCamera;
        minPosDifferentialY = dyCamera;
    }

    DFPBRT_CPU_GPU
    std::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    DFPBRT_CPU_GPU
    std::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const;

    static OrthographicCamera *Create(const ParameterDictionary &parameters,
                                      const CameraTransform &cameraTransform, Film film,
                                      Medium medium, const FileLoc *loc,
                                      Allocator alloc = {});

    DFPBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for OrthographicCamera");
        return {};
    }

    DFPBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for OrthographicCamera");
    }

    DFPBRT_CPU_GPU
    std::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for OrthographicCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // OrthographicCamera Private Members
    Vector3f dxCamera, dyCamera;
};



}

#endif