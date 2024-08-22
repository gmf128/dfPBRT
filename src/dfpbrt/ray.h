#ifndef DFPBRT_RAY_H
#define DFPBRT_RAY_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/vecmath.h>
#include <dfpbrt/base/medium.h>

namespace dfpbrt {

// Ray Definition
class Ray {
  public:
    // Ray Public Methods
    bool HasNaN() const { return (o.HasNaN() || d.HasNaN()); }

    std::string ToString() const;

    Point3f operator()(Float t) const { return o + d * t; }

    Ray() = default;
    
    Ray(Point3f o, Vector3f d, Float time = 0.f, Medium medium = {})
        : o(o), d(d), time(time), medium(medium) {}

    // Ray Public Members
    Point3f o;
    Vector3f d;
    Float time = 0;
    Medium medium{};
};

// RayDifferential Definition
class RayDifferential : public Ray {
  public:
    // RayDifferential Public Methods
    RayDifferential() = default;
    
    RayDifferential(Point3f o, Vector3f d, Float time = 0.f, Medium medium = {})
        : Ray(o, d, time, medium) {}

    
    explicit RayDifferential(const Ray &ray) : Ray(ray) {}

    void ScaleDifferentials(Float s) {
        rxOrigin = o + (rxOrigin - o) * s;
        ryOrigin = o + (ryOrigin - o) * s;
        rxDirection = d + (rxDirection - d) * s;
        ryDirection = d + (ryDirection - d) * s;
    }

    
    bool HasNaN() const {
        return Ray::HasNaN() ||
               (hasDifferentials && (rxOrigin.HasNaN() || ryOrigin.HasNaN() ||
                                     rxDirection.HasNaN() || ryDirection.HasNaN()));
    }
    std::string ToString() const;

    // RayDifferential Public Members
    bool hasDifferentials = false;
    Point3f rxOrigin, ryOrigin;
    Vector3f rxDirection, ryDirection;
};

// Ray Inline Functions
inline Point3f OffsetRayOrigin(Point3fi pi, Normal3f n, Vector3f w) {
    // Find vector _offset_ to corner of error bounds and compute initial _po_
    Float d = Dot(Abs(n), pi.Error());
    Vector3f offset = d * Vector3f(n);
    if (Dot(w, n) < 0)
        offset = -offset;
    Point3f po = Point3f(pi) + offset;

    // Round offset point _po_ away from _p_
    for (int i = 0; i < 3; ++i) {
        if (offset[i] > 0)
            po[i] = NextFloatUp(po[i]);
        else if (offset[i] < 0)
            po[i] = NextFloatDown(po[i]);
    }

    return po;
}

inline Ray SpawnRay(Point3fi pi, Normal3f n, Float time, Vector3f d) {
    return Ray(OffsetRayOrigin(pi, n, d), d, time);
}

inline Ray SpawnRayTo(Point3fi pFrom, Normal3f n, Float time, Point3f pTo) {
    Vector3f d = pTo - Point3f(pFrom);
    return SpawnRay(pFrom, n, time, d);
}

inline Ray SpawnRayTo(Point3fi pFrom, Normal3f nFrom, Float time,
                                   Point3fi pTo, Normal3f nTo) {
    Point3f pf = OffsetRayOrigin(pFrom, nFrom, Point3f(pTo) - Point3f(pFrom));
    Point3f pt = OffsetRayOrigin(pTo, nTo, pf - Point3f(pTo));
    return Ray(pf, pt - pf, time);
}

}  // namespace pbrt


#endif