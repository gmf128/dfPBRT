#ifndef DFPBRT_INTERACTION_H
#define DFPBRT_INTERACTION_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/check.h>
#include <dfpbrt/util/float.h>
#include <dfpbrt/util/vecmath.h>
#include <dfpbrt/util/math.h>
#include <dfpbrt/util/transform.h>


namespace dfpbrt{

// Interaction Definition
class Interaction {
  public:
    // Interaction Public Methods
    Interaction() = default;

    DFPBRT_CPU_GPU
    Interaction(Point3fi pi, Normal3f n, Point2f uv, Vector3f wo, Float time)
        : pi(pi), n(n), uv(uv), wo(Normalize(wo)), time(time) {}

    DFPBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

    DFPBRT_CPU_GPU
    bool IsSurfaceInteraction() const { return n != Normal3f(0, 0, 0); }
    DFPBRT_CPU_GPU
    bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }

    DFPBRT_CPU_GPU
    const SurfaceInteraction &AsSurface() const {
        CHECK(IsSurfaceInteraction());
        return (const SurfaceInteraction &)*this;
    }

    DFPBRT_CPU_GPU
    SurfaceInteraction &AsSurface() {
        CHECK(IsSurfaceInteraction());
        return (SurfaceInteraction &)*this;
    }

    // used by medium ctor
    // DFPBRT_CPU_GPU
    // Interaction(Point3f p, Vector3f wo, Float time, Medium medium)
    //     : pi(p), time(time), wo(wo), medium(medium) {}
    // DFPBRT_CPU_GPU
    // Interaction(Point3f p, Normal3f n, Float time, Medium medium)
    //     : pi(p), n(n), time(time), medium(medium) {}
    // DFPBRT_CPU_GPU
    // Interaction(Point3f p, Point2f uv) : pi(p), uv(uv) {}
    // DFPBRT_CPU_GPU
    // Interaction(const Point3fi &pi, Normal3f n, Float time = 0, Point2f uv = {})
    //     : pi(pi), n(n), uv(uv), time(time) {}
    // DFPBRT_CPU_GPU
    // Interaction(const Point3fi &pi, Normal3f n, Point2f uv) : pi(pi), n(n), uv(uv) {}
    // DFPBRT_CPU_GPU
    // Interaction(Point3f p, Float time, Medium medium)
    //     : pi(p), time(time), medium(medium) {}
    // DFPBRT_CPU_GPU
    // Interaction(Point3f p, const MediumInterface *mediumInterface)
    //     : pi(p), mediumInterface(mediumInterface) {}
    // DFPBRT_CPU_GPU
    // Interaction(Point3f p, Float time, const MediumInterface *mediumInterface)
    //     : pi(p), time(time), mediumInterface(mediumInterface) {}
    // DFPBRT_CPU_GPU
    // const MediumInteraction &AsMedium() const {
    //     CHECK(IsMediumInteraction());
    //     return (const MediumInteraction &)*this;
    // }
    // DFPBRT_CPU_GPU
    // MediumInteraction &AsMedium() {
    //     CHECK(IsMediumInteraction());
    //     return (MediumInteraction &)*this;
    // }

    std::string ToString() const;

    DFPBRT_CPU_GPU
    Point3f OffsetRayOrigin(Vector3f w) const { return pbrt::OffsetRayOrigin(pi, n, w); }

    DFPBRT_CPU_GPU
    Point3f OffsetRayOrigin(Point3f pt) const { return OffsetRayOrigin(pt - p()); }

    DFPBRT_CPU_GPU
    RayDifferential SpawnRay(Vector3f d) const {
        return RayDifferential(OffsetRayOrigin(d), d, time, GetMedium(d));
    }

    DFPBRT_CPU_GPU
    Ray SpawnRayTo(Point3f p2) const {
        Ray r = pbrt::SpawnRayTo(pi, n, time, p2);
        r.medium = GetMedium(r.d);
        return r;
    }

    DFPBRT_CPU_GPU
    Ray SpawnRayTo(const Interaction &it) const {
        Ray r = pbrt::SpawnRayTo(pi, n, time, it.pi, it.n);
        r.medium = GetMedium(r.d);
        return r;
    }

    DFPBRT_CPU_GPU
    Medium GetMedium(Vector3f w) const {
        if (mediumInterface)
            return Dot(w, n) > 0 ? mediumInterface->outside : mediumInterface->inside;
        return medium;
    }

    DFPBRT_CPU_GPU
    Medium GetMedium() const {
        if (mediumInterface)
            DCHECK_EQ(mediumInterface->inside, mediumInterface->outside);
        return mediumInterface ? mediumInterface->inside : medium;
    }

    // Interaction Public Members

    //All interactions have a point  associated with them. And account for errors(using Interval)
    Point3fi pi;
    //All interactions also have a time associated with them. Among other uses, this value is necessary for setting the time of a spawned ray leaving the interaction.
    Float time = 0;
    //For interactions that lie along a ray (either from a ray–shape intersection or from a ray passing through participating media), 
    //the negative ray direction is stored in the wo member variable, which corresponds to w_o, the notation we use for the outgoing direction when computing lighting at points.
    Vector3f wo;
    //For interactions on surfaces, n stores the surface normal at the point and uv stores its  parametric coordinates. 
    Normal3f n;
    Point2f uv;
    //For interactions on MediumSurfaces
    const MediumInterface *mediumInterface = nullptr;
    Medium medium = nullptr;
};

// SurfaceInteraction Definition
class SurfaceInteraction : public Interaction {
  public:
    // SurfaceInteraction Public Methods
    SurfaceInteraction() = default;

    //This representation implicitly assumes that shapes have a parametric description
    DFPBRT_CPU_GPU
    SurfaceInteraction(Point3fi pi, Point2f uv, Vector3f wo, Vector3f dpdu, Vector3f dpdv,
                       Normal3f dndu, Normal3f dndv, Float time, bool flipNormal)
        : Interaction(pi, Normal3f(Normalize(Cross(dpdu, dpdv))), uv, wo, time),
          dpdu(dpdu),
          dpdv(dpdv),
          dndu(dndu),
          dndv(dndv) {
        // Initialize shading geometry from true geometry
        shading.n = n;
        shading.dpdu = dpdu;
        shading.dpdv = dpdv;
        shading.dndu = dndu;
        shading.dndv = dndv;

        // Adjust normal based on orientation and handedness
        if (flipNormal) {
            n *= -1;
            shading.n *= -1;
        }
    }
    //Another constructor that is able to set faceIndex(also uses constructor delegation)
    PBRT_CPU_GPU
    SurfaceInteraction(Point3fi pi, Point2f uv, Vector3f wo, Vector3f dpdu, Vector3f dpdv,
                       Normal3f dndu, Normal3f dndv, Float time, bool flipNormal,
                       int faceIndex)
        : SurfaceInteraction(pi, uv, wo, dpdu, dpdv, dndu, dndv, time, flipNormal) {
        this->faceIndex = faceIndex;
    }

    PBRT_CPU_GPU
    void SetShadingGeometry(Normal3f ns, Vector3f dpdus, Vector3f dpdvs, Normal3f dndus,
                            Normal3f dndvs, bool orientationIsAuthoritative) {
        // Compute _shading.n_ for _SurfaceInteraction_
        shading.n = ns;
        DCHECK(shading.n!= Normal3f(0, 0, 0));
        if (orientationIsAuthoritative)//means the ns is more authoritative in orientation
            n = FaceForward(n, shading.n);
        else //means n is more authoritative in orientation
            shading.n = FaceForward(shading.n, n);

        // Initialize _shading_ partial derivative values
        shading.dpdu = dpdus;
        shading.dpdv = dpdvs;
        shading.dndu = dndus;
        shading.dndv = dndvs;
        while (LengthSquared(shading.dpdu) > 1e16f ||
               LengthSquared(shading.dpdv) > 1e16f) {
            shading.dpdu /= 1e8f;
            shading.dpdv /= 1e8f;
        }
    }

    std::string ToString() const;

    void SetIntersectionProperties(Material mtl, Light area,
                                   const MediumInterface *primMediumInterface,
                                   Medium rayMedium) {
        material = mtl;
        areaLight = area;
        CHECK_GE(Dot(n, shading.n), 0.);
        // Set medium properties at surface intersection
        if (primMediumInterface && primMediumInterface->IsMediumTransition())
            mediumInterface = primMediumInterface;
        else
            medium = rayMedium;
    }

    PBRT_CPU_GPU
    void ComputeDifferentials(const RayDifferential &r, Camera camera,
                              int samplesPerPixel);

    PBRT_CPU_GPU
    void SkipIntersection(RayDifferential *ray, Float t) const;

    using Interaction::SpawnRay;
    PBRT_CPU_GPU
    RayDifferential SpawnRay(const RayDifferential &rayi, const BSDF &bsdf, Vector3f wi,
                             int /*BxDFFlags*/ flags, Float eta) const;

    BSDF GetBSDF(const RayDifferential &ray, SampledWavelengths &lambda, Camera camera,
                 ScratchBuffer &scratchBuffer, Sampler sampler);
    BSSRDF GetBSSRDF(const RayDifferential &ray, SampledWavelengths &lambda,
                     Camera camera, ScratchBuffer &scratchBuffer);

    PBRT_CPU_GPU
    SampledSpectrum Le(Vector3f w, const SampledWavelengths &lambda) const;

    // SurfaceInteraction Public Members
    // Local shading geometry
    Vector3f dpdu, dpdv;
    Normal3f dndu, dndv;
    // Is is needed to pay attention that shading store another instance of dpdu dpdv dndu dndv
    // SurfaceInteraction stores a second instance of a surface normal and the various partial derivatives to *represent possibly perturbed values of these quantities—as can be generated by bump mapping or interpolated per-vertex normals with meshes*. 
    // Some parts of the system use this shading geometry, while others need to work with the original quantities.
    struct {
        Normal3f n;
        Vector3f dpdu, dpdv;
        Normal3f dndu, dndv;
    } shading;
    // Able to set faceIndex for mesh geometry
    int faceIndex = 0;
    Material material;
    Light areaLight;
    Vector3f dpdx, dpdy;
    Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;
};

// MediumInteraction Definition
class MediumInteraction : public Interaction {
  public:
    // MediumInteraction Public Methods
    DFPBRT_CPU_GPU
    MediumInteraction() : phase(nullptr) {}
    DFPBRT_CPU_GPU
    MediumInteraction(Point3f p, Vector3f wo, Float time, Medium medium,
                      PhaseFunction phase)
        : Interaction(p, wo, time, medium), phase(phase) {}

    std::string ToString() const;

    // MediumInteraction Public Members
    PhaseFunction phase;
};


}

#endif