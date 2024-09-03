#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/cameras.h>
#include <dfpbrt/options.h>

#include <dfpbrt/util/log.h>
#include <dfpbrt/util/error.h>

namespace dfpbrt{
// CameraTransform Method Definitions
CameraTransform::CameraTransform(const AnimatedTransform &worldFromCamera) {
    switch (Options->renderingSpace) {
    case RenderingCoordinateSystem::Camera: {
        // Compute _worldFromRender_ for camera-space rendering
        // However, because worldFromRender cannot be animated, the implementation takes the world-from-camera transformation at the midpoint of the frame and then folds the effect of any animation in the camera transformation into renderFromCamera.
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        worldFromRender = worldFromCamera.Interpolate(tMid);
        break;
    }
    case RenderingCoordinateSystem::CameraWorld: {
        // Compute _worldFromRender_ for camera-world space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        Point3f pCamera = worldFromCamera(Point3f(0, 0, 0), tMid);
        worldFromRender = Translate(Vector3f(pCamera));
        break;
    }
    case RenderingCoordinateSystem::World: {
        // Compute _worldFromRender_ for world-space rendering
        worldFromRender = Transform();
        break;
    }
    default:
        LOG_FATAL("Unhandled rendering coordinate space");
    }
    LOG_VERBOSE(std::format("World-space position: {}", worldFromRender(Point3f(0, 0, 0)).ToString()));
    // Compute _renderFromCamera_ transformation
    Transform renderFromWorld = Inverse(worldFromRender);
    Transform rfc[2] = {renderFromWorld * worldFromCamera.startTransform,
                        renderFromWorld * worldFromCamera.endTransform};
    renderFromCamera = AnimatedTransform(rfc[0], worldFromCamera.startTime, rfc[1],
                                         worldFromCamera.endTime);
}

std::string CameraTransform::ToString() const {
    return std::format("[ CameraTransform renderFromCamera: {} worldFromRender: {} ]",
                        renderFromCamera.ToString(), worldFromRender.ToString());
}

// Camera Method Definitions(Dispatch funtions)
std::optional<CameraRayDifferential> Camera::GenerateRayDifferential(
    CameraSample sample, SampledWavelengths &lambda) const {
    auto gen = [&](auto ptr) { return ptr->GenerateRayDifferential(sample, lambda); };
    return Dispatch(gen);
}

SampledSpectrum Camera::We(const Ray &ray, SampledWavelengths &lambda,
                           Point2f *pRaster2) const {
    auto we = [&](auto ptr) { return ptr->We(ray, lambda, pRaster2); };
    return Dispatch(we);
}

void Camera::PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_We(ray, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

std::optional<CameraWiSample> Camera::SampleWi(const Interaction &ref, Point2f u,
                                                SampledWavelengths &lambda) const {
    auto sample = [&](auto ptr) { return ptr->SampleWi(ref, u, lambda); };
    return Dispatch(sample);
}

void Camera::InitMetadata(ImageMetadata *metadata) const {
    auto init = [&](auto ptr) { return ptr->InitMetadata(metadata); };
    return DispatchCPU(init);
}

std::string Camera::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

// CameraBase Method Definitions
CameraBase::CameraBase(CameraBaseParameters p)
    : cameraTransform(p.cameraTransform),
      shutterOpen(p.shutterOpen),
      shutterClose(p.shutterClose),
      film(p.film),
      medium(p.medium) {
    if (cameraTransform.CameraFromRenderHasScale())
        Warning("Scaling detected in rendering space to camera space transformation!\n"
                "The system has numerous assumptions, implicit and explicit,\n"
                "that this transform will have no scale factors in it.\n"
                "Proceed at your own risk; your image may have errors or\n"
                "the system may crash as a result of this.");
}

//GenerateRay() variates in different derived class, but GenerateRayDifferential which uses GenerateRay in the same manner can be impl in an abstract way.
std::optional<CameraRayDifferential> CameraBase::GenerateRayDifferential(
    Camera camera, CameraSample sample, SampledWavelengths &lambda) {
    // Generate regular camera ray _cr_ for ray differential
    std::optional<CameraRay> cr = camera.GenerateRay(sample, lambda);
    if (!cr)
        // fail to generate ray, neither able to generate differentiable ray
        return {};
    RayDifferential rd(cr->ray);

    // Find camera ray after shifting one pixel in the $x$ direction
    std::optional<CameraRay> rx;
    for (Float eps : {.05f, -.05f}) {
        CameraSample sshift = sample;
        sshift.pFilm.x += eps;
        // Try to generate ray with _sshift_ and compute $x$ differential
        if (rx = camera.GenerateRay(sshift, lambda); rx) {
            rd.rxOrigin = rd.o + (rx->ray.o - rd.o) / eps;  // when eps is 0.05, it is forward differencing; when eps is -0.05, it is backward differencing
            rd.rxDirection = rd.d + (rx->ray.d - rd.d) / eps;   // since it is either forward or backward *difference*, remember to divide eps
            break;// !!! default: forward difference; only when forward difference is not available, we will calculate backward difference, otherwise the loop will break after forward one is done.
        }
    }

    // Find camera ray after shifting one pixel in the $y$ direction
    std::optional<CameraRay> ry;
    for (Float eps : {.05f, -.05f}) {
        CameraSample sshift = sample;
        sshift.pFilm.y += eps;
        if (ry = camera.GenerateRay(sshift, lambda); ry) {
            rd.ryOrigin = rd.o + (ry->ray.o - rd.o) / eps;
            rd.ryDirection = rd.d + (ry->ray.d - rd.d) / eps;
            break;// !!! default: forward difference; only when forward difference is not available, we will calculate backward difference, otherwise the loop will break after forward one is done.
        }
    }

    // Return approximate ray differential and weight
    rd.hasDifferentials = rx && ry;
    return CameraRayDifferential{rd, cr->weight};
}

void CameraBase::FindMinimumDifferentials(Camera camera) {
    minPosDifferentialX = minPosDifferentialY = minDirDifferentialX =
        minDirDifferentialY = Vector3f(Infinity_, Infinity_, Infinity_);

    CameraSample sample;
    sample.pLens = Point2f(0.5, 0.5);
    sample.time = 0.5;
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5);

    int n = 512;
    for (int i = 0; i < n; ++i) {
        sample.pFilm.x = Float(i) / (n - 1) * film.FullResolution().x;
        sample.pFilm.y = Float(i) / (n - 1) * film.FullResolution().y;

        std::optional<CameraRayDifferential> crd =
            camera.GenerateRayDifferential(sample, lambda);
        if (!crd)
            continue;

        RayDifferential &ray = crd->ray;
        Vector3f dox = CameraFromRender(ray.rxOrigin - ray.o, ray.time);
        if (Length(dox) < Length(minPosDifferentialX))
            minPosDifferentialX = dox;
        Vector3f doy = CameraFromRender(ray.ryOrigin - ray.o, ray.time);
        if (Length(doy) < Length(minPosDifferentialY))
            minPosDifferentialY = doy;

        ray.d = Normalize(ray.d);
        ray.rxDirection = Normalize(ray.rxDirection);
        ray.ryDirection = Normalize(ray.ryDirection);

        Frame f = Frame::FromZ(ray.d);
        Vector3f df = f.ToLocal(ray.d);  // should be (0, 0, 1);
        Vector3f dxf = Normalize(f.ToLocal(ray.rxDirection));
        Vector3f dyf = Normalize(f.ToLocal(ray.ryDirection));

        if (Length(dxf - df) < Length(minDirDifferentialX))
            minDirDifferentialX = dxf - df;
        if (Length(dyf - df) < Length(minDirDifferentialY))
            minDirDifferentialY = dyf - df;
    }

    LOG_VERBOSE(std::format("Camera min pos differentials: {}, {}", minPosDifferential.ToString()X,
                minPosDifferentialY.ToString()));
    LOG_VERBOSE(std::format("Camera min dir differentials: {}, {}", minDirDifferentialX.ToString(),
                minDirDifferentialY.ToString()));
}

void CameraBase::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld = cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();
}

std::string CameraBase::ToString() const {
    return std::format( "cameraTransform: {} shutterOpen: {} shutterClose: {} film: {} "
                        "medium: {} minPosDifferentialX: {} minPosDifferentialY: {} "
                        "minDirDifferentialX: {} minDirDifferentialY: {} ",
                        cameraTransform.ToString(), shutterOpen.ToString(), shutterClose.ToString(), film.ToString(),
                        medium ? medium.ToString() : "(nullptr)",
                        minPosDifferentialX.ToString(), minPosDifferentialY.ToString(), minDirDifferentialX.ToString(),
                        minDirDifferentialY.ToString());
}

std::string CameraSample::ToString() const {
    return std::format("[ CameraSample pFilm: {} pLens: {} time: %f filterWeight: %f ]",
                        pFilm.ToString(), pLens.ToString(), time, filterWeight);
}

std::string ProjectiveCamera::BaseToString() const {
    return CameraBase::ToString() +
           std::format("screenFromCamera: {} cameraFromRaster: {} "
                        "rasterFromScreen: {} screenFromRaster: {} "
                        "lensRadius: {} focalDistance: {}",
                        screenFromCamera.ToString(), cameraFromRaster.ToString(), rasterFromScreen.ToString(),
                        screenFromRaster.ToString(), lensRadius.ToString(), focalDistance);
}

// OrthographicCamera Method Definitions
std::optional<CameraRay> OrthographicCamera::GenerateRay(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }
    // From camera space to render space
    return CameraRay{RenderFromCamera(ray)};//CameraRay = Ray + SampleWeight
}

std::optional<CameraRayDifferential> OrthographicCamera::GenerateRayDifferential(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute main orthographic viewing ray
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    RayDifferential ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    // Compute ray differentials for _OrthographicCamera_
    if (lensRadius > 0) {
        // Compute _OrthographicCamera_ ray differentials accounting for lens
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = pCamera + dxCamera + (ft * Vector3f(0, 0, 1));
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        pFocus = pCamera + dyCamera + (ft * Vector3f(0, 0, 1));
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);

    } else {
        ray.rxOrigin = ray.o + dxCamera;
        ray.ryOrigin = ray.o + dyCamera;
        ray.rxDirection = ray.ryDirection = ray.d;
    }

    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string OrthographicCamera::ToString() const {
    return std::format("[ OrthographicCamera{} dxCamera: {} dyCamera: {} ]",
                        BaseToString(), dxCamera.ToString(), dyCamera.ToString());
}

OrthographicCamera *OrthographicCamera::Create(const ParameterDictionary &parameters,
                                               const CameraTransform &cameraTransform,
                                               Film film, Medium medium,
                                               const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e6f);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (Options->fullscreen) {
                Warning("\"screenwindow\" is ignored in fullscreen mode");
        } else {
            if (sw.size() == 4) {
                screen.pMin.x = sw[0];
                screen.pMax.x = sw[1];
                screen.pMin.y = sw[2];
                screen.pMax.y = sw[3];
            } else {
                Error("\"screenwindow\" should have four values");
            }
        }
    }
    return alloc.new_object<OrthographicCamera>(cameraBaseParameters, screen, lensradius,
                                                focaldistance);
}

// PerspectiveCamera Method Definitions
std::optional<CameraRay> PerspectiveCamera::GenerateRay(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(Point3f(0, 0, 0), Normalize(Vector3f(pCamera)), SampleTime(sample.time),
            medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    return CameraRay{RenderFromCamera(ray)};
}

std::optional<CameraRayDifferential> PerspectiveCamera::GenerateRayDifferential(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);
    Vector3f dir = Normalize(Vector3f(pCamera.x, pCamera.y, pCamera.z));
    RayDifferential ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    // Compute offset rays for _PerspectiveCamera_ ray differentials
    if (lensRadius > 0) {
        // Compute _PerspectiveCamera_ ray differentials accounting for lens
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute $x$ ray differential for _PerspectiveCamera_ with lens
        Vector3f dx = Normalize(Vector3f(pCamera + dxCamera));
        Float ft = focalDistance / dx.z;
        Point3f pFocus = Point3f(0, 0, 0) + (ft * dx);
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        // Compute $y$ ray differential for _PerspectiveCamera_ with lens
        Vector3f dy = Normalize(Vector3f(pCamera + dyCamera));
        ft = focalDistance / dy.z;
        pFocus = Point3f(0, 0, 0) + (ft * dy);
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);

    } else {
        ray.rxOrigin = ray.ryOrigin = ray.o;
        ray.rxDirection = Normalize(Vector3f(pCamera) + dxCamera);
        ray.ryDirection = Normalize(Vector3f(pCamera) + dyCamera);
    }

    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string PerspectiveCamera::ToString() const {
    return std::foramt("[ PerspectiveCamera {} dxCamera: {} dyCamera: {} A: "
                        "{} cosTotalWidth: {} ]",
                        BaseToString(), dxCamera.ToString(), dyCamera.ToString(), A, cosTotalWidth);
}

PerspectiveCamera *PerspectiveCamera::Create(const ParameterDictionary &parameters,
                                             const CameraTransform &cameraTransform,
                                             Film film, Medium medium, const FileLoc *loc,
                                             Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e6);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (Options->fullscreen) {
                Warning("\"screenwindow\" is ignored in fullscreen mode");
        } else {
            if (sw.size() == 4) {
                screen.pMin.x = sw[0];
                screen.pMax.x = sw[1];
                screen.pMin.y = sw[2];
                screen.pMax.y = sw[3];
            } else {
                Error(loc, "\"screenwindow\" should have four values");
            }
        }
    }
    Float fov = parameters.GetOneFloat("fov", 90.);
    return alloc.new_object<PerspectiveCamera>(cameraBaseParameters, fov, screen,
                                               lensradius, focaldistance);
}

// SphericalCamera Method Definitions
std::optional<CameraRay> SphericalCamera::GenerateRay(CameraSample sample,
                                                       SampledWavelengths &lambda) const {
    // Compute spherical camera ray direction
    // Film space position as uv and transform it into the 3D sphere directions
    Point2f uv(sample.pFilm.x / film.FullResolution().x,
               sample.pFilm.y / film.FullResolution().y);
    Vector3f dir;
    if (mapping == EquiRectangular) {
        // Compute ray direction using equirectangular mapping
        Float theta = Pi * uv[1], phi = 2 * Pi * uv[0];
        dir = SphericalDirection(std::sin(theta), std::cos(theta), phi);

    } else {
        // Compute ray direction using equal area mapping
        uv = WrapEqualAreaSquare(uv);
        dir = EqualAreaSquareToSphere(uv);
    }
    std::swap(dir.y, dir.z);

    Ray ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    return CameraRay{RenderFromCamera(ray)};
}

SphericalCamera *SphericalCamera::Create(const ParameterDictionary &parameters,
                                         const CameraTransform &cameraTransform,
                                         Film film, Medium medium, const FileLoc *loc,
                                         Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e30f);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (Options->fullscreen) {
                Warning("\"screenwindow\" is ignored in fullscreen mode");
        } else {
            if (sw.size() == 4) {
                screen.pMin.x = sw[0];
                screen.pMax.x = sw[1];
                screen.pMin.y = sw[2];
                screen.pMax.y = sw[3];
            } else {
                Error(loc, "\"screenwindow\" should have four values");
            }
        }
    }
    (void)lensradius;     // don't need this
    (void)focaldistance;  // don't need this

    std::string m = parameters.GetOneString("mapping", "equalarea");
    Mapping mapping;
    if (m == "equalarea")
        mapping = EqualArea;
    else if (m == "equirectangular")
        mapping = EquiRectangular;
    else
        ErrorExit(loc,
                  "%s: unknown mapping for spherical camera. (Must be "
                  "\"equalarea\" or \"equirectangular\".)",
                  m);

    return alloc.new_object<SphericalCamera>(cameraBaseParameters, mapping);
}

std::string SphericalCamera::ToString() const {
    return std::format("[ SphericalCamera {} mapping: {} ]", CameraBase::ToString(),
                        mapping == EquiRectangular ? "EquiRectangular" : "EqualArea");
}


}